import math
from typing import Any

import pytest
from rdkit import Chem

from src.config import BINARY_PREDICTION_THRESHOLDS, TASKS
from src.model import GNNXAIInferenceResponse
from src.service import MolecularInferenceService


def _assert_aligned_v2000_response(
    result: dict[str, Any],
    smiles: str,
    expected_graph_atoms: int,
    expected_total_atoms: int,
) -> Chem.Mol:
    """Validate response schema, probabilities, and exact MolBlock <-> atomscores index alignment."""
    validated = GNNXAIInferenceResponse(**result)
    expected_tasks = set(TASKS)
    assert set(validated.predictions) == expected_tasks
    assert set(validated.probabilities) == expected_tasks
    assert set(validated.atomscores) == expected_tasks
    assert "V2000" in validated.molblock and "M  END" in validated.molblock

    parsed_mol = Chem.MolFromMolBlock(validated.molblock, removeHs=False)
    assert parsed_mol is not None and parsed_mol.GetNumAtoms() == expected_total_atoms

    input_mol = Chem.MolFromSmiles(smiles)
    assert input_mol is not None
    canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(input_mol, canonical=True))
    assert (
        canonical_mol is not None
        and canonical_mol.GetNumAtoms() == expected_graph_atoms
    )

    assert all(
        parsed_mol.GetAtomWithIdx(i).GetAtomicNum()
        == canonical_mol.GetAtomWithIdx(i).GetAtomicNum()
        and parsed_mol.GetAtomWithIdx(i).GetFormalCharge()
        == canonical_mol.GetAtomWithIdx(i).GetFormalCharge()
        for i in range(expected_graph_atoms)
    )
    assert all(
        parsed_mol.GetAtomWithIdx(i).GetAtomicNum() == 1
        for i in range(expected_graph_atoms, expected_total_atoms)
    )

    for task in TASKS:
        prob = validated.probabilities[task]
        assert math.isfinite(prob) and 0.0 <= prob <= 1.0
        assert validated.predictions[task] is (
            prob > BINARY_PREDICTION_THRESHOLDS[task]
        )

        scores = validated.atomscores[task]
        assert list(scores) == list(range(expected_total_atoms))
        assert all(
            isinstance(scores[i], float) and math.isfinite(scores[i])
            for i in range(expected_graph_atoms)
        )
        assert all(
            scores[i] is None for i in range(expected_graph_atoms, expected_total_atoms)
        )

    return parsed_mol


@pytest.mark.parametrize(
    ("smiles", "expected_graph_atoms", "expected_total_atoms", "expect_3d_nonplanar"),
    [
        ("C[C@H](O)Cl", 4, 9, True),  # tetrahedral stereocenter -> 3D with non-zero Z
        ("[2H]CCO", 4, 9, False),  # deuterium stays in graph; 5 added H's are None
        ("O=C=O", 3, 3, False),  # hydrogen-free linear molecule: zero None entries
        ("CCO.[Na+]", 4, 10, False),  # multi-fragment salt with unbonded counterion
        ("O=P(O)(O)c1ccc(Br)cc1", 11, 17, False),  # P + Br aryl phosphonic acid
        ("FC(F)(F)S(=O)(=O)NCC[NH3+].[Cl-]", 12, 20, False),  # F, S, N+/Cl- HCl salt
    ],
)
def test_service_aligns_atomscores_with_parsed_v2000_molblock_indices(
    real_inference_service: MolecularInferenceService,
    smiles: str,
    expected_graph_atoms: int,
    expected_total_atoms: int,
    expect_3d_nonplanar: bool,
) -> None:
    """Diverse chemical boundaries produce aligned V2000 MolBlocks, scores, and 3D coords where expected."""
    result = real_inference_service.predict_and_explain(smiles)
    parsed_mol = _assert_aligned_v2000_response(
        result, smiles, expected_graph_atoms, expected_total_atoms
    )

    if expect_3d_nonplanar:
        conf = parsed_mol.GetConformer()
        assert conf.Is3D() is True
        z_coords = [conf.GetAtomPosition(i).z for i in range(parsed_mol.GetNumAtoms())]
        assert any(abs(z) > 1e-4 for z in z_coords)


def test_service_canonicalization_produces_identical_full_payload_for_equivalent_smiles(
    real_inference_service: MolecularInferenceService,
) -> None:
    """Equivalent chiral SMILES spellings yield identical predictions, probabilities, atomscores, and MolBlock."""
    result_a = real_inference_service.predict_and_explain("C[C@H](O)Cl")
    result_b = real_inference_service.predict_and_explain("O[C@@H](C)Cl")

    assert result_a["predictions"] == result_b["predictions"]
    assert result_a["molblock"] == result_b["molblock"]
    for task in TASKS:
        assert result_a["probabilities"][task] == pytest.approx(
            result_b["probabilities"][task], rel=1e-6, abs=1e-7
        )
        scores_a, scores_b = result_a["atomscores"][task], result_b["atomscores"][task]
        assert list(scores_a) == list(scores_b)
        for idx, score_a in scores_a.items():
            if score_a is None:
                assert scores_b[idx] is None
            else:
                assert score_a == pytest.approx(scores_b[idx], rel=1e-6, abs=1e-7)


def test_service_degrades_strained_cage_to_2d_molblock_with_aligned_atomscores(
    real_inference_service: MolecularInferenceService,
) -> None:
    """Strained cage whose 3D embedding fails degrades cleanly to 2D coordinates with aligned atomscores."""
    smiles = "C1=CC2=CC=C1C=C2"
    result = real_inference_service.predict_and_explain(smiles)
    strained_mol = _assert_aligned_v2000_response(result, smiles, 8, 14)

    conf = strained_mol.GetConformer()
    assert conf.Is3D() is False
    assert "2D" in result["molblock"]

    positions = [conf.GetAtomPosition(i) for i in range(strained_mol.GetNumAtoms())]
    assert all(abs(pos.z) < 1e-6 for pos in positions)
    assert max(pos.x for pos in positions) - min(pos.x for pos in positions) > 1.0
    assert max(pos.y for pos in positions) - min(pos.y for pos in positions) > 1.0
