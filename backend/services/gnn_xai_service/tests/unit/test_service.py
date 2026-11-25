"""Unit tests for `MolecularInferenceService`: SMILES validation, orchestration, bounds, and index invariants."""

import math
from typing import Any

import pytest
import torch
from rdkit import Chem

from src import service as service_module
from src.config import TASKS, TASKS_NUMBER
from src.exceptions import ModelInferenceError, MoleculeValidationError
from src.model import GNNXAIInferenceResponse
from src.service import MolecularInferenceService
from tests.unit.conftest import StubGNNPredictor, StubXAIExplainer

ServiceFixture = tuple[MolecularInferenceService, StubGNNPredictor, StubXAIExplainer]


@pytest.fixture
def service_and_stubs() -> ServiceFixture:
    predictor, explainer = StubGNNPredictor(), StubXAIExplainer()
    return (
        MolecularInferenceService(predictor=predictor, explainer=explainer),
        predictor,
        explainer,
    )


def _pred_output(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    base: dict[str, torch.Tensor] = {
        "probabilities": torch.full((TASKS_NUMBER,), 0.5, dtype=torch.float32),
        "predictions": torch.zeros((TASKS_NUMBER,), dtype=torch.int32),
        "atom_embeddings": torch.ones((3, 2), dtype=torch.float32),
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("smiles", "graph_atoms", "total_atoms"),
    [("CCO", 3, 9), ("O=C=O", 3, 3), ("[2H]CCO", 4, 9)],
)
def test_aligns_tasks_probabilities_and_atomscores(
    service_and_stubs: ServiceFixture,
    smiles: str,
    graph_atoms: int,
    total_atoms: int,
) -> None:
    service, predictor, explainer = service_and_stubs
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)

    result = service.predict_and_explain(smiles)

    assert predictor.predict_call_count == 1
    assert predictor.received_smiles == [canonical]
    assert explainer.explain_call_count == 1
    received_emb, received_targets = explainer.received_args[0]
    assert received_emb.shape == (graph_atoms, 2)
    assert received_targets == TASKS_NUMBER

    validated = GNNXAIInferenceResponse(**result)
    assert list(validated.predictions.keys()) == TASKS
    assert list(validated.probabilities.keys()) == TASKS
    assert list(validated.atomscores.keys()) == TASKS
    assert "V2000" in validated.molblock and "M  END" in validated.molblock
    parsed = Chem.MolFromMolBlock(validated.molblock, removeHs=False)
    assert parsed is not None and parsed.GetNumAtoms() == total_atoms

    expected_probs = torch.linspace(0.05, 0.95, steps=TASKS_NUMBER, dtype=torch.float32)
    expected_sums = received_emb.sum(dim=-1).tolist()
    for task_idx, task in enumerate(TASKS):
        prob = float(expected_probs[task_idx].item())
        assert validated.probabilities[task] == pytest.approx(prob, abs=1e-6)
        assert type(validated.predictions[task]) is bool
        assert validated.predictions[task] is (prob > 0.5)
        scores = validated.atomscores[task]
        assert list(scores.keys()) == list(range(total_atoms))
        for i in range(graph_atoms):
            assert scores[i] == pytest.approx(
                float(expected_sums[i]) + 0.01 * task_idx, abs=1e-6
            )
        assert all(scores[i] is None for i in range(graph_atoms, total_atoms))


@pytest.mark.parametrize(
    ("smiles_a", "smiles_b"),
    [("CCO", "OCC"), ("C[C@H](O)Cl", "O[C@@H](C)Cl")],
)
def test_canonicalizes_equivalent_smiles_identically(
    service_and_stubs: ServiceFixture, smiles_a: str, smiles_b: str
) -> None:
    service, predictor, _ = service_and_stubs
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_a), canonical=True)

    result_a = service.predict_and_explain(smiles_a)
    result_b = service.predict_and_explain(smiles_b)

    assert predictor.received_smiles == [canonical, canonical]
    assert result_a == result_b


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        " \t\n ",
        None,
        123,
        "not_a_smiles",
        "C(C)(C)(C)(C)C",
        "C",
        "[Na+].[Cl-]",
        "[H]C([H])([H])[H]",
    ],
)
def test_rejects_invalid_smiles_without_calling_collaborators(
    invalid_input: Any, service_and_stubs: ServiceFixture
) -> None:
    service, predictor, explainer = service_and_stubs

    with pytest.raises(MoleculeValidationError):
        service.predict_and_explain(invalid_input)

    assert predictor.predict_call_count == 0
    assert explainer.explain_call_count == 0


def test_accepts_exact_boundary_probabilities_zero_and_one() -> None:
    probs = torch.tensor(
        [0.0 if i % 2 == 0 else 1.0 for i in range(TASKS_NUMBER)], dtype=torch.float32
    )
    preds = torch.tensor(
        [0 if i % 2 == 0 else 1 for i in range(TASKS_NUMBER)], dtype=torch.int32
    )
    service = MolecularInferenceService(
        predictor=StubGNNPredictor(
            custom_output=_pred_output(probabilities=probs, predictions=preds)
        ),
        explainer=StubXAIExplainer(),
    )

    result = service.predict_and_explain("CCO")

    for idx, task in enumerate(TASKS):
        assert result["probabilities"][task] == (0.0 if idx % 2 == 0 else 1.0)
        assert math.isfinite(result["probabilities"][task])
        assert result["predictions"][task] is (idx % 2 != 0)


@pytest.mark.parametrize("bad_prob", [-1e-6, 1.0 + 1e-6, float("nan"), float("inf")])
def test_rejects_invalid_probabilities_before_explainer(bad_prob: float) -> None:
    probs = torch.full((TASKS_NUMBER,), 0.5, dtype=torch.float32)
    probs[-1] = bad_prob
    explainer = StubXAIExplainer()
    service = MolecularInferenceService(
        predictor=StubGNNPredictor(custom_output=_pred_output(probabilities=probs)),
        explainer=explainer,
    )

    with pytest.raises(ModelInferenceError):
        service.predict_and_explain("CCO")

    assert explainer.explain_call_count == 0


@pytest.mark.parametrize(
    "faulty_output",
    [
        {k: v for k, v in _pred_output().items() if k != "probabilities"},
        _pred_output(probabilities=torch.full((TASKS_NUMBER - 1,), 0.5)),
        _pred_output(probabilities=torch.full((TASKS_NUMBER, 1), 0.5)),
        {k: v for k, v in _pred_output().items() if k != "predictions"},
        _pred_output(predictions=torch.zeros((TASKS_NUMBER + 1,), dtype=torch.int32)),
        _pred_output(predictions=torch.zeros((TASKS_NUMBER, 1), dtype=torch.int32)),
        {k: v for k, v in _pred_output().items() if k != "atom_embeddings"},
    ],
)
def test_rejects_malformed_predictor_outputs(
    faulty_output: dict[str, torch.Tensor],
) -> None:
    explainer = StubXAIExplainer()
    service = MolecularInferenceService(
        predictor=StubGNNPredictor(custom_output=faulty_output),
        explainer=explainer,
    )

    with pytest.raises(ModelInferenceError):
        service.predict_and_explain("CCO")

    assert explainer.explain_call_count == 0


@pytest.mark.parametrize(
    "faulty_scores",
    [
        [[0.1, 0.2, 0.3] for _ in range(TASKS_NUMBER - 1)],
        [[0.1, 0.2] for _ in range(TASKS_NUMBER)],
        [[0.1, 0.2, 0.3] for _ in range(TASKS_NUMBER - 1)] + [[0.1, 0.2, 0.3, 0.4]],
        [
            [float("nan"), 0.1, 0.2],
            *[[0.1, 0.2, 0.3] for _ in range(TASKS_NUMBER - 1)],
        ],
        [
            [float("inf"), 0.1, 0.2],
            *[[0.1, 0.2, 0.3] for _ in range(TASKS_NUMBER - 1)],
        ],
        [
            [float("-inf"), 0.1, 0.2],
            *[[0.1, 0.2, 0.3] for _ in range(TASKS_NUMBER - 1)],
        ],
    ],
)
def test_rejects_malformed_explainer_outputs(faulty_scores: list[list[float]]) -> None:
    service = MolecularInferenceService(
        predictor=StubGNNPredictor(),
        explainer=StubXAIExplainer(custom_scores=faulty_scores),
    )

    with pytest.raises(ModelInferenceError):
        service.predict_and_explain("CCO")


def test_propagates_predictor_error_without_calling_explainer() -> None:
    explainer = StubXAIExplainer()
    service = MolecularInferenceService(
        predictor=StubGNNPredictor(exc=ModelInferenceError("boom")),
        explainer=explainer,
    )

    with pytest.raises(ModelInferenceError):
        service.predict_and_explain("CCO")

    assert explainer.explain_call_count == 0


def test_propagates_explainer_error() -> None:
    predictor = StubGNNPredictor()
    service = MolecularInferenceService(
        predictor=predictor,
        explainer=StubXAIExplainer(exc=ModelInferenceError("boom")),
    )

    with pytest.raises(ModelInferenceError):
        service.predict_and_explain("CCO")

    assert predictor.predict_call_count == 1


def test_rejects_mol_with_h_having_fewer_atoms_than_heavy_mol(
    monkeypatch: pytest.MonkeyPatch, service_and_stubs: ServiceFixture
) -> None:
    service, _, _ = service_and_stubs
    smaller = Chem.MolFromSmiles("CC")
    assert smaller is not None
    monkeypatch.setattr(
        service_module,
        "mol_block_generation",
        lambda _mol: ("V2000\nM  END\n", smaller),
    )

    with pytest.raises(ModelInferenceError):
        service.predict_and_explain("CCO")


def test_rejects_heavy_atom_type_mismatch(service_and_stubs: ServiceFixture) -> None:
    service, _, _ = service_and_stubs
    mol, reordered = Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("OCC")
    assert mol is not None and reordered is not None

    with pytest.raises(ModelInferenceError):
        service.validate_mol_atomic_indices(
            mol, mol.GetNumAtoms(), reordered, reordered.GetNumAtoms()
        )


def test_rejects_non_hydrogen_appended_atom(service_and_stubs: ServiceFixture) -> None:
    service, _, _ = service_and_stubs
    mol, longer = Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC")
    assert mol is not None and longer is not None

    with pytest.raises(ModelInferenceError):
        service.validate_mol_atomic_indices(
            mol, mol.GetNumAtoms(), longer, longer.GetNumAtoms()
        )
