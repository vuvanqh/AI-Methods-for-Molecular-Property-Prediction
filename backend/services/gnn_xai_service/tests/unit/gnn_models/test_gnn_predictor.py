import pytest
import torch
from rdkit import Chem

from src.config import BINARY_PREDICTION_THRESHOLDS, TASKS, TASKS_NUMBER
from src.exceptions import ModelInferenceError
from src.gnn_models.chemprop import Chemprop
from src.gnn_models.gnn_interface import GNNPredictor


@pytest.fixture(scope="module")
def gnn_predictor() -> GNNPredictor:
    """Deterministically initialized `Chemprop` instance tested via the `GNNPredictor` interface."""
    torch.manual_seed(42)
    return Chemprop(
        n_tasks=TASKS_NUMBER,
        device=torch.device("cpu"),
        thresholds=[BINARY_PREDICTION_THRESHOLDS[task] for task in TASKS],
    )


def test_gnn_predictor_interface_cannot_be_instantiated_without_implementing_methods() -> None:
    class IncompletePredictor(GNNPredictor):
        pass

    with pytest.raises(TypeError):
        IncompletePredictor()


def test_chemprop_implements_gnn_predictor_interface(gnn_predictor: GNNPredictor) -> None:
    assert isinstance(gnn_predictor, GNNPredictor)


# =====================================================================
# 2. GNNPredictor.predict — Behavioral Contract & Per-Task Thresholds
# =====================================================================


@pytest.mark.parametrize(
    ("smiles", "expected_atoms"),
    [
        ("CCO", 3),
        ("CC(C)O", 4),
        ("c1ccccc1", 6),
        ("C=CC(=O)O", 5),
        ("CC(=O)[O-]", 4),
        ("C[C@H](O)Cl", 4),
        ("[2H]CCO", 4),
    ],
)
def test_gnn_predictor_predict_returns_valid_prediction_contract(
    gnn_predictor: GNNPredictor,
    smiles: str,
    expected_atoms: int,
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None

    result = gnn_predictor.predict(mol)
    assert set(result.keys()) == {"probabilities", "predictions", "atom_embeddings"}

    probabilities = result["probabilities"]
    predictions = result["predictions"]
    atom_embeddings = result["atom_embeddings"]

    configured_thresholds = torch.tensor(
        [BINARY_PREDICTION_THRESHOLDS[task] for task in TASKS],
        dtype=probabilities.dtype,
        device=probabilities.device,
    )

    assert isinstance(probabilities, torch.Tensor)
    assert isinstance(predictions, torch.Tensor)
    assert isinstance(atom_embeddings, torch.Tensor)

    assert probabilities.shape == (TASKS_NUMBER,)
    assert predictions.shape == (TASKS_NUMBER,)
    assert atom_embeddings.ndim == 2
    assert atom_embeddings.shape[0] == expected_atoms
    assert atom_embeddings.shape[1] > 0

    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(atom_embeddings).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert torch.equal(
        predictions,
        (probabilities > configured_thresholds).int(),
    )


def test_gnn_predictor_applies_distinct_threshold_per_task() -> None:
    mol = Chem.MolFromSmiles("CC(C)O")
    assert mol is not None

    torch.manual_seed(42)
    baseline_predictor: GNNPredictor = Chemprop(
        n_tasks=TASKS_NUMBER,
        device=torch.device("cpu"),
        thresholds=[0.5] * TASKS_NUMBER,
    )
    baseline_probs = baseline_predictor.predict(mol)["probabilities"].detach().cpu().tolist()

    # Even tasks get a threshold below prob (expect 1); odd tasks get a threshold above prob (expect 0)
    custom_thresholds = [
        max(0.0, prob - 0.05) if idx % 2 == 0 else min(1.0, prob + 0.05)
        for idx, prob in enumerate(baseline_probs)
    ]

    torch.manual_seed(42)
    custom_predictor: GNNPredictor = Chemprop(
        n_tasks=TASKS_NUMBER,
        device=torch.device("cpu"),
        thresholds=custom_thresholds,
    )
    result = custom_predictor.predict(mol)

    expected_predictions = torch.tensor(
        [1 if idx % 2 == 0 else 0 for idx in range(TASKS_NUMBER)],
        dtype=result["predictions"].dtype,
    )
    assert torch.equal(result["predictions"].cpu(), expected_predictions)


def test_gnn_predictor_is_deterministic_across_repeated_calls(
    gnn_predictor: GNNPredictor,
) -> None:
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None

    first = gnn_predictor.predict(mol)
    second = gnn_predictor.predict(mol)

    torch.testing.assert_close(first["probabilities"], second["probabilities"])
    torch.testing.assert_close(first["predictions"], second["predictions"])
    torch.testing.assert_close(first["atom_embeddings"], second["atom_embeddings"])


@pytest.mark.parametrize(
    ("smiles_a", "smiles_b"),
    [
        ("CCO", "OCC"),
        ("CC(C)O", "OC(C)C"),
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("CC(=O)Oc1ccccc1C(=O)O", "O=C(O)c1ccccc1OC(C)=O"),
        ("F/C=C/F", r"F\C=C\F"),
        ("C[C@H](O)Cl", "Cl[C@H](C)O"),
    ],
)
def test_gnn_predictor_invariance_across_equivalent_smiles(
    gnn_predictor: GNNPredictor,
    smiles_a: str,
    smiles_b: str,
) -> None:
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    assert mol_a is not None and mol_b is not None

    result_a = gnn_predictor.predict(mol_a)
    result_b = gnn_predictor.predict(mol_b)

    torch.testing.assert_close(
        result_a["probabilities"],
        result_b["probabilities"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.equal(result_a["predictions"], result_b["predictions"])


# =====================================================================
# 3. GNNPredictor.forward_from_embeddings — Readout & Differentiability
# =====================================================================


def test_gnn_predictor_forward_from_embeddings_matches_predict_and_supports_gradients(
    gnn_predictor: GNNPredictor,
) -> None:
    mol = Chem.MolFromSmiles("CC(C)O")
    assert mol is not None

    prediction = gnn_predictor.predict(mol)
    embeddings_with_grad = prediction["atom_embeddings"].detach().clone().requires_grad_(True)

    logits = gnn_predictor.forward_from_embeddings(embeddings_with_grad)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, TASKS_NUMBER)
    assert torch.isfinite(logits).all()
    assert logits.requires_grad is True

    torch.testing.assert_close(
        logits.sigmoid().squeeze(0),
        prediction["probabilities"],
        rtol=1e-5,
        atol=1e-6,
    )

    logits.sum().backward()
    assert embeddings_with_grad.grad is not None
    assert embeddings_with_grad.grad.shape == prediction["atom_embeddings"].shape
    assert torch.isfinite(embeddings_with_grad.grad).all()


# =====================================================================
# 4. GNNPredictor.predict — None & Empty Molecule Rejection
# =====================================================================


def test_gnn_predictor_rejects_none_and_empty_molecules(
    gnn_predictor: GNNPredictor,
) -> None:
    with pytest.raises(ModelInferenceError):
        gnn_predictor.predict(None)

    with pytest.raises(ModelInferenceError):
        gnn_predictor.predict(Chem.Mol())
