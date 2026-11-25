"""Unit tests for `Chemprop` predictor contract, threshold boundaries, gradients, and invariant guards."""

import math

import pytest
import torch
from rdkit import Chem

from src.config import BINARY_PREDICTION_THRESHOLDS, TASKS, TASKS_NUMBER
from src.exceptions import ModelInferenceError
from src.gnn_models.chemprop import Chemprop


@pytest.fixture(scope="module")
def gnn_predictor() -> Chemprop:
    torch.manual_seed(42)
    return Chemprop(
        n_tasks=TASKS_NUMBER,
        device=torch.device("cpu"),
        thresholds=[BINARY_PREDICTION_THRESHOLDS[task] for task in TASKS],
    )


@pytest.fixture(scope="module")
def two_task_predictor() -> Chemprop:
    torch.manual_seed(7)
    return Chemprop(n_tasks=2, device=torch.device("cpu"), thresholds=[0.5, 0.5])


def test_binary_thresholds_cover_exact_tasks_in_unit_interval() -> None:
    assert set(BINARY_PREDICTION_THRESHOLDS.keys()) == set(TASKS)
    for task, threshold in BINARY_PREDICTION_THRESHOLDS.items():
        assert isinstance(threshold, float), f"Threshold for {task} must be float"
        assert math.isfinite(threshold), f"Threshold for {task} must be finite"
        assert 0.0 <= threshold <= 1.0, (
            f"Threshold for {task} out of [0, 1]: {threshold}"
        )


@pytest.mark.parametrize(
    ("smiles", "expected_atoms"),
    [
        ("CC", 2),
        ("CCO.[Na+]", 4),
        ("O=P(O)(O)c1ccc(Br)cc1", 11),
        ("FC(F)(F)S(=O)(=O)NCC[NH3+].[Cl-]", 12),
    ],
)
def test_predict_returns_valid_contract(
    gnn_predictor: Chemprop, smiles: str, expected_atoms: int
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert gnn_predictor.model.training is False

    result = gnn_predictor.predict(mol)

    assert set(result.keys()) == {"probabilities", "predictions", "atom_embeddings"}
    probs, preds, emb = (
        result["probabilities"],
        result["predictions"],
        result["atom_embeddings"],
    )
    thresholds = torch.tensor(
        [BINARY_PREDICTION_THRESHOLDS[task] for task in TASKS],
        dtype=probs.dtype,
        device=probs.device,
    )
    assert probs.shape == (TASKS_NUMBER,) and preds.shape == (TASKS_NUMBER,)
    assert emb.ndim == 2 and emb.shape[0] == expected_atoms and emb.shape[1] > 0
    assert torch.isfinite(probs).all() and torch.isfinite(emb).all()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert torch.equal(preds, (probs > thresholds).int())


def test_threshold_uses_strict_greater_than_at_exact_boundaries() -> None:
    """Verify `prob < t -> 0`, `prob == t -> 0`, `prob > t -> 1`, and `0.0`/`1.0` thresholds unmocked."""
    predictor = Chemprop(
        n_tasks=5,
        device=torch.device("cpu"),
        thresholds=[0.5, 0.5, 0.5, 0.0, 1.0],
    )
    final_linear = [
        m
        for m in predictor.model.predictor.ffn.modules()
        if isinstance(m, torch.nn.Linear)
    ][-1]
    with torch.no_grad():
        final_linear.weight.zero_()
        final_linear.bias.copy_(
            torch.tensor([-1e-4, 0.0, 1e-4, -100.0, 100.0], dtype=torch.float32)
        )
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None

    result = predictor.predict(mol)
    probs = result["probabilities"]

    assert probs[0].item() < 0.5
    assert probs[1].item() == 0.5
    assert probs[2].item() > 0.5
    assert probs[3].item() == 0.0
    assert probs[4].item() == 1.0
    assert torch.equal(
        result["predictions"].cpu(), torch.tensor([0, 0, 1, 0, 0], dtype=torch.int32)
    )


def test_predict_is_deterministic(gnn_predictor: Chemprop) -> None:
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None

    first, second = gnn_predictor.predict(mol), gnn_predictor.predict(mol)

    torch.testing.assert_close(first["probabilities"], second["probabilities"])
    torch.testing.assert_close(first["predictions"], second["predictions"])
    torch.testing.assert_close(first["atom_embeddings"], second["atom_embeddings"])


def test_predict_returns_outputs_without_autograd_graph(
    gnn_predictor: Chemprop,
) -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    result = gnn_predictor.predict(mol)

    assert result["probabilities"].requires_grad is False
    assert result["predictions"].requires_grad is False
    assert result["atom_embeddings"].requires_grad is False


@pytest.mark.parametrize(
    ("n_tasks", "thresholds"),
    [
        (0, []),
        (-1, [0.5]),
        (TASKS_NUMBER, [0.5]),
        (TASKS_NUMBER, [0.5] * (TASKS_NUMBER + 1)),
    ],
)
def test_rejects_invalid_n_tasks_or_mismatched_thresholds(
    n_tasks: int,
    thresholds: list[float],
) -> None:
    with pytest.raises(ValueError):
        Chemprop(n_tasks=n_tasks, device=torch.device("cpu"), thresholds=thresholds)


def test_forward_from_embeddings_matches_predict_and_supports_gradients(
    gnn_predictor: Chemprop,
) -> None:
    mol = Chem.MolFromSmiles("CC(C)O")
    assert mol is not None

    prediction = gnn_predictor.predict(mol)
    embeddings_with_grad = (
        prediction["atom_embeddings"].detach().clone().requires_grad_(True)
    )

    logits = gnn_predictor.forward_from_embeddings(embeddings_with_grad)

    assert logits.shape == (1, TASKS_NUMBER)
    assert torch.isfinite(logits).all() and logits.requires_grad is True
    torch.testing.assert_close(
        logits.sigmoid().squeeze(0), prediction["probabilities"], rtol=1e-5, atol=1e-6
    )

    logits.sum().backward()
    assert embeddings_with_grad.grad is not None
    assert embeddings_with_grad.grad.shape == prediction["atom_embeddings"].shape
    assert torch.isfinite(embeddings_with_grad.grad).all()
    assert embeddings_with_grad.grad.abs().sum() > 0


def test_rejects_none_and_empty_molecules(gnn_predictor: Chemprop) -> None:
    with pytest.raises(ModelInferenceError):
        gnn_predictor.predict(None)
    with pytest.raises(ModelInferenceError):
        gnn_predictor.predict(Chem.Mol())


def test_raises_when_featurizer_returns_none(
    monkeypatch: pytest.MonkeyPatch, two_task_predictor: Chemprop
) -> None:
    monkeypatch.setattr(two_task_predictor, "featurizer", lambda _mol: None)
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None

    with pytest.raises(ModelInferenceError):
        two_task_predictor.predict(mol)


@pytest.mark.parametrize(
    "bad_embeddings",
    [
        torch.ones((3,)),
        torch.ones((2, 4)),
        torch.tensor([[1.0, float("nan")], [1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[1.0, float("inf")], [1.0, 1.0], [1.0, 1.0]]),
    ],
)
def test_rejects_invalid_atom_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    two_task_predictor: Chemprop,
    bad_embeddings: torch.Tensor,
) -> None:
    monkeypatch.setattr(
        two_task_predictor.model.message_passing,
        "forward",
        lambda *_a, **_kw: bad_embeddings,
    )
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    with pytest.raises(ModelInferenceError):
        two_task_predictor.predict(mol)


@pytest.mark.parametrize(
    "bad_logits",
    [
        torch.zeros((2,)),
        torch.zeros((2, 2)),
        torch.zeros((1, 3)),
        torch.tensor([[0.0, float("nan")]]),
        torch.tensor([[0.0, float("inf")]]),
    ],
)
def test_rejects_invalid_logits(
    monkeypatch: pytest.MonkeyPatch,
    two_task_predictor: Chemprop,
    bad_logits: torch.Tensor,
) -> None:
    monkeypatch.setattr(
        two_task_predictor.model.predictor.ffn,
        "forward",
        lambda *_a, **_kw: bad_logits,
    )
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None

    with pytest.raises(ModelInferenceError):
        two_task_predictor.predict(mol)
