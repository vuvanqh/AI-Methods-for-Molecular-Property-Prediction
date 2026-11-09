import math
import pytest
import torch

from src.config import BINARY_PREDICTION_THRESHOLD, TASKS, TASKS_NUMBER
from src.exceptions import MoleculeValidationError
from src.model import GNNXAIInferenceResponse
from src.service import MolecularInferenceService


@pytest.mark.random_weights
@pytest.mark.parametrize(
    ("smiles", "expected_graph_atoms", "expected_total_atoms"),
    [
        ("CCO", 3, 9),
        ("CC(C)O", 4, 12),
        ("c1ccccc1", 6, 12),
        ("C=CC(=O)O", 5, 9),
        ("CC(=O)[O-]", 4, 7),
        ("C[C@H](O)Cl", 4, 9),
        ("[2H]CCO", 4, 9),
    ],
)
def test_service_pipeline_end_to_end_contract_for_representative_molecules(
    inference_service: MolecularInferenceService,
    smiles: str,
    expected_graph_atoms: int,
    expected_total_atoms: int,
) -> None:
    result = inference_service.predict_and_explain(smiles)

    # 1. Validate against Pydantic response schema
    validated = GNNXAIInferenceResponse(**result)
    assert set(validated.predictions.keys()) == set(TASKS)
    assert set(validated.probabilities.keys()) == set(TASKS)
    assert set(validated.atomscores.keys()) == set(TASKS)
    assert len(validated.predictions) == TASKS_NUMBER
    assert "V2000" in validated.molblock and "M  END" in validated.molblock

    # 2. Verify threshold consistency and heavy vs added-H atomscores
    for task in TASKS:
        prob = validated.probabilities[task]
        pred = validated.predictions[task]
        assert 0.0 <= prob <= 1.0
        assert not math.isnan(prob)
        assert pred is (prob > BINARY_PREDICTION_THRESHOLD)

        scores_map = validated.atomscores[task]
        assert list(scores_map.keys()) == list(range(expected_total_atoms))
        for idx in range(expected_graph_atoms):
            assert isinstance(scores_map[idx], float)
            assert math.isfinite(scores_map[idx])  # type: ignore[arg-type]
        for idx in range(expected_graph_atoms, expected_total_atoms):
            assert scores_map[idx] is None


@pytest.mark.random_weights
@pytest.mark.parametrize(
    ("smiles_a", "smiles_b"),
    [
        ("CCO", "OCC"),
        ("CC(C)O", "OC(C)C"),
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("C[C@H](O)Cl", "O[C@@H](C)Cl"),
    ],
)
def test_service_pipeline_invariance_on_equivalent_smiles(
    inference_service: MolecularInferenceService,
    smiles_a: str,
    smiles_b: str,
) -> None:
    res_a = inference_service.predict_and_explain(smiles_a)
    res_b = inference_service.predict_and_explain(smiles_b)

    assert res_a["predictions"] == res_b["predictions"]
    assert res_a["molblock"] == res_b["molblock"]

    probs_a = torch.tensor([res_a["probabilities"][t] for t in TASKS], dtype=torch.float32)
    probs_b = torch.tensor([res_b["probabilities"][t] for t in TASKS], dtype=torch.float32)
    torch.testing.assert_close(probs_a, probs_b, rtol=1e-5, atol=1e-6)

    for task in TASKS:
        assert res_a["atomscores"][task] == res_b["atomscores"][task]


# =====================================================================
# Unhappy Paths Across Integrated Service Pipeline
# =====================================================================


@pytest.mark.fast_rdkit
@pytest.mark.parametrize(
    ("invalid_smiles", "expected_error_match"),
    [
        ("", "SMILES must not be empty"),
        ("   ", "SMILES must not be empty"),
        ("not_a_valid_smiles", "RDKit could not parse the provided SMILES"),
        ("C(C)(C)(C)(C)C", "RDKit could not parse the provided SMILES"),
    ],
)
def test_service_pipeline_unhappy_path_invalid_smiles_raises_validation_error(
    inference_service: MolecularInferenceService,
    invalid_smiles: str,
    expected_error_match: str,
) -> None:
    with pytest.raises(MoleculeValidationError, match=expected_error_match):
        inference_service.predict_and_explain(invalid_smiles)
