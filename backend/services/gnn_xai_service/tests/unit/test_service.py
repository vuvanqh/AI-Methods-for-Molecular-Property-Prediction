from unittest.mock import MagicMock, patch

import pytest
import torch
from rdkit import Chem

from src.config import TASKS, TASKS_NUMBER
from src.exceptions import ModelInferenceError, MoleculeValidationError
from src.service import MolecularInferenceService

# =====================================================================
# 1. MolecularInferenceService.predict_and_explain — Happy Paths
# =====================================================================


@pytest.mark.fast_rdkit
@pytest.mark.parametrize(
    ("smiles", "expected_graph_atoms"),
    [
        ("CCO", 3),
        ("CC(C)O", 4),
        ("c1ccccc1", 6),
        ("C=CC(=O)O", 5),
        ("CC(=O)[O-]", 4),
        ("C[C@H](O)Cl", 4),
        ("F/C=C/F", 4),
        ("[2H]CCO", 4),
    ],
)
def test_predict_and_explain_happy_path_returns_valid_response_structure(
    smiles: str,
    expected_graph_atoms: int,
) -> None:
    """
    Verifies that `predict_and_explain` parses valid SMILES, invokes the predictor and
    explainer with the canonicalized molecule, maps heavy-atom IG scores to floats and
    added visualization hydrogens to None, and returns a valid V2000 MolBlock.
    """
    service = MolecularInferenceService.__new__(MolecularInferenceService)
    service.predictor = MagicMock()
    service.explainer = MagicMock()

    service.predictor.predict.return_value = {
        "probabilities": torch.full((TASKS_NUMBER,), 0.75),
        "predictions": torch.ones((TASKS_NUMBER,), dtype=torch.int32),
        "atom_embeddings": torch.ones((expected_graph_atoms, 16)),
        "mol_graph_index": torch.zeros((expected_graph_atoms,), dtype=torch.long),
    }
    service.explainer.explain.return_value = [
        [float(i + 1) for i in range(expected_graph_atoms)] for _ in range(TASKS_NUMBER)
    ]

    result = service.predict_and_explain(smiles)

    assert set(result.keys()) == {"predictions", "probabilities", "atomscores", "molblock"}
    assert len(result["predictions"]) == TASKS_NUMBER
    assert len(result["probabilities"]) == TASKS_NUMBER
    assert "V2000" in result["molblock"]

    parsed_molblock = Chem.MolFromMolBlock(result["molblock"], removeHs=False)
    assert parsed_molblock is not None
    total_atoms_with_h = parsed_molblock.GetNumAtoms()
    assert total_atoms_with_h > expected_graph_atoms

    for task in TASKS:
        assert result["predictions"][task] is True
        assert result["probabilities"][task] == pytest.approx(0.75)

        scores_for_task = result["atomscores"][task]
        assert list(scores_for_task.keys()) == list(range(total_atoms_with_h))
        for idx in range(expected_graph_atoms):
            assert isinstance(scores_for_task[idx], float)
        for idx in range(expected_graph_atoms, total_atoms_with_h):
            assert scores_for_task[idx] is None


@pytest.mark.fast_rdkit
@pytest.mark.parametrize(
    ("smiles_a", "smiles_b"),
    [
        ("CCO", "OCC"),
        ("CCO", "[CH3][CH2][OH]"),
        ("CCO", "[H]C([H])([H])C([H])([H])O"),
        ("CC(C)O", "OC(C)C"),
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("C[C@H](O)Cl", "O[C@@H](C)Cl"),
    ],
)
def test_predict_and_explain_canonicalizes_equivalent_smiles_before_calling_predictor(
    smiles_a: str,
    smiles_b: str,
) -> None:
    """
    Verifies that `predict_and_explain` normalizes equivalent SMILES (including reversed
    traversals, explicit hydrogens, Kekule rings, and chiral @/@@ variants) into the same
    canonical RDKit Mol before passing it to `predictor.predict` and `mol_block_generation`.
    """
    service = MolecularInferenceService.__new__(MolecularInferenceService)
    service.predictor = MagicMock()
    service.explainer = MagicMock()

    captured_mols: list[Chem.Mol] = []

    def fake_predict(mol: Chem.Mol) -> dict:
        captured_mols.append(mol)
        n = mol.GetNumAtoms()
        return {
            "probabilities": torch.full((TASKS_NUMBER,), 0.3),
            "predictions": torch.zeros((TASKS_NUMBER,), dtype=torch.int32),
            "atom_embeddings": torch.ones((n, 8)),
            "mol_graph_index": torch.zeros((n,), dtype=torch.long),
        }

    service.predictor.predict.side_effect = fake_predict

    mol_expected = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles_a), canonical=True))
    n_atoms = mol_expected.GetNumAtoms()
    service.explainer.explain.return_value = [[0.1 * i for i in range(n_atoms)] for _ in range(TASKS_NUMBER)]

    res_a = service.predict_and_explain(smiles_a)
    res_b = service.predict_and_explain(smiles_b)

    assert len(captured_mols) == 2
    mol_passed_a, mol_passed_b = captured_mols

    # Both calls received a canonicalized Mol with identical atom ordering and chiral tags
    assert [a.GetSymbol() for a in mol_passed_a.GetAtoms()] == [a.GetSymbol() for a in mol_passed_b.GetAtoms()]
    assert [a.GetChiralTag() for a in mol_passed_a.GetAtoms()] == [a.GetChiralTag() for a in mol_passed_b.GetAtoms()]
    assert res_a == res_b


# =====================================================================
# 2. MolecularInferenceService.predict_and_explain — Unhappy Paths
# =====================================================================


@pytest.mark.fast_rdkit
@pytest.mark.parametrize(
    "empty_or_whitespace",
    [
        "",
        " ",
        "    ",
        "\t",
        "\n",
        " \t\r\n ",
        None,
    ],
)
def test_predict_and_explain_rejects_empty_or_whitespace_smiles(
    empty_or_whitespace: str | None,
    uninitialized_service: MolecularInferenceService,
) -> None:
    with pytest.raises(MoleculeValidationError, match="SMILES must not be empty"):
        uninitialized_service.predict_and_explain(empty_or_whitespace)  # type: ignore[arg-type]


@pytest.mark.fast_rdkit
@pytest.mark.parametrize(
    "invalid_smiles",
    [
        "not_a_smiles",
        "C(C",               # Unclosed branch parenthesis
        "c1cccc1",           # Unkekulizable aromatic ring
        "C1CC",              # Unclosed ring digit
        "C(C)(C)(C)(C)C",    # Pentavalent carbon (fails RDKit valence check)
        "F(F)F",             # Hypervalent fluorine
    ],
)
def test_predict_and_explain_rejects_unparseable_smiles(
    invalid_smiles: str,
    uninitialized_service: MolecularInferenceService,
) -> None:
    # `with pytest.raises(...)` asserts that the code block raises MoleculeValidationError
    # with a message matching the regex; if no error is raised, pytest fails the test.
    with pytest.raises(MoleculeValidationError, match="RDKit could not parse the provided SMILES"):
        uninitialized_service.predict_and_explain(invalid_smiles)


@pytest.mark.fast_rdkit
def test_predict_and_explain_rejects_zero_atom_molecule(
    uninitialized_service: MolecularInferenceService,
) -> None:
    """
    Tests the `if num_atoms_without_h == 0:` guard in `service.py` (line 35).
    We pass a non-empty string ("CCO") so execution passes the initial empty-string check
    on line 26, and mock `Chem.MolFromSmiles` to return an empty 0-atom `Chem.Mol()`.
    """
    with patch("src.service.Chem.MolFromSmiles", return_value=Chem.Mol()):
        with pytest.raises(MoleculeValidationError, match="The molecule must contain at least one atom"):
            uninitialized_service.predict_and_explain("CCO")


@pytest.mark.fast_rdkit
def test_predict_and_explain_rejects_when_canonical_reparse_returns_none(
    uninitialized_service: MolecularInferenceService,
) -> None:
    """Tests the `if mol is None:` guard after `Chem.MolFromSmiles(canonical_smiles)` in `service.py`."""
    valid_mol = Chem.MolFromSmiles("CCO")
    with patch("src.service.Chem.MolFromSmiles", side_effect=[valid_mol, None]):
        with pytest.raises(MoleculeValidationError, match="RDKit could not re-parse the canonicalized SMILES"):
            uninitialized_service.predict_and_explain("CCO")


@pytest.mark.fast_rdkit
def test_predict_and_explain_raises_when_mol_with_h_has_fewer_atoms_than_heavy_atoms() -> None:
    """Tests the `if num_atoms_with_h < num_atoms_without_h:` guard in `service.py`."""
    service = MolecularInferenceService.__new__(MolecularInferenceService)
    service.predictor = MagicMock()
    service.explainer = MagicMock()

    service.predictor.predict.return_value = {
        "probabilities": torch.full((TASKS_NUMBER,), 0.5),
        "predictions": torch.zeros((TASKS_NUMBER,), dtype=torch.int32),
        "atom_embeddings": torch.ones((3, 16)),
        "mol_graph_index": torch.zeros((3,), dtype=torch.long),
    }
    service.explainer.explain.return_value = [[0.1, 0.2, 0.3] for _ in range(TASKS_NUMBER)]

    smaller_mol = Chem.MolFromSmiles("CC")  # 2 atoms < 3 atoms in CCO
    with patch("src.service.mol_block_generation", return_value=("mock_molblock", smaller_mol)):
        with pytest.raises(ModelInferenceError, match="Atom count mismatch: mol_with_h has 2 atoms"):
            service.predict_and_explain("CCO")


@pytest.mark.fast_rdkit
def test_service_predict_and_explain_guardrails_on_corrupted_predictor_or_explainer_outputs() -> None:
    """Tests all intermediate tensor/list invariant checks in `service.predict_and_explain`."""
    service = MolecularInferenceService.__new__(MolecularInferenceService)
    service.predictor = MagicMock()
    service.explainer = MagicMock()

    valid_predictor_output = {
        "probabilities": torch.full((TASKS_NUMBER,), 0.5),
        "predictions": torch.zeros((TASKS_NUMBER,), dtype=torch.int32),
        "atom_embeddings": torch.ones((3, 16)),
        "mol_graph_index": torch.zeros((3,), dtype=torch.long),
    }

    # 1. Missing 'probabilities'
    service.predictor.predict.return_value = {
        k: v for k, v in valid_predictor_output.items() if k != "probabilities"
    }
    with pytest.raises(ModelInferenceError, match="Missing 'probabilities'"):
        service.predict_and_explain("CCO")

    # 2. Wrong probabilities length
    service.predictor.predict.return_value = {
        **valid_predictor_output,
        "probabilities": torch.full((5,), 0.5),
    }
    with pytest.raises(ModelInferenceError, match="Probabilities length mismatch"):
        service.predict_and_explain("CCO")

    # 3. Out-of-bounds probability (> 1.0)
    bad_probs = torch.full((TASKS_NUMBER,), 0.5)
    bad_probs[0] = 1.2
    service.predictor.predict.return_value = {
        **valid_predictor_output,
        "probabilities": bad_probs,
    }
    with pytest.raises(ModelInferenceError, match="Invalid probability value"):
        service.predict_and_explain("CCO")

    # 4. Missing 'predictions'
    service.predictor.predict.return_value = {
        k: v for k, v in valid_predictor_output.items() if k != "predictions"
    }
    with pytest.raises(ModelInferenceError, match="Missing 'predictions'"):
        service.predict_and_explain("CCO")

    # 5. Wrong predictions length
    service.predictor.predict.return_value = {
        **valid_predictor_output,
        "predictions": torch.zeros((3,), dtype=torch.int32),
    }
    with pytest.raises(ModelInferenceError, match="Predictions length mismatch"):
        service.predict_and_explain("CCO")

    # 6. Wrong target count from explainer
    service.predictor.predict.return_value = valid_predictor_output
    service.explainer.explain.return_value = [[0.1, 0.2, 0.3]]  # Only 1 target instead of 18
    with pytest.raises(ModelInferenceError, match="Explainer returned 1 target attributions"):
        service.predict_and_explain("CCO")

    # 7. Wrong per-atom score count from explainer for a task
    service.explainer.explain.return_value = [[0.1, 0.2] for _ in range(len(TASKS))]  # 2 scores for 3-atom CCO
    with pytest.raises(ModelInferenceError, match="attribution score count"):
        service.predict_and_explain("CCO")


# =====================================================================
# 3. MolecularInferenceService.validate_mol_atomic_indices — Happy & Unhappy Paths
# =====================================================================


@pytest.mark.fast_rdkit
@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "CC(C)O",
        "c1ccccc1",
        "C=CC(=O)O",
        "CC(=O)[O-]",
        "C[C@H](O)Cl",
        "[2H]CCO",
    ],
)
def test_validate_mol_atomic_indices_passes_when_heavy_atoms_match_and_hydrogens_appended(
    smiles: str,
    uninitialized_service: MolecularInferenceService,
) -> None:
    """Happy path for `validate_mol_atomic_indices`: heavy atoms match at 0..K-1 and H atoms are at K..N-1."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol_with_h = Chem.AddHs(mol)

    # Should complete without raising ModelInferenceError
    uninitialized_service.validate_mol_atomic_indices(
        mol, mol.GetNumAtoms(), mol_with_h, mol_with_h.GetNumAtoms()
    )


@pytest.mark.fast_rdkit
def test_validate_mol_atomic_indices_raises_on_atom_index_mismatch(
    uninitialized_service: MolecularInferenceService,
) -> None:
    """Unhappy path 1 for `validate_mol_atomic_indices`: `heavy_atom.GetIdx() != atom_in_h_mol.GetIdx()`."""
    mol = Chem.MolFromSmiles("CCO")
    mol_with_h = MagicMock()
    corrupted_atom = MagicMock()
    corrupted_atom.GetIdx.return_value = 99  # Mismatched index for i=0
    mol_with_h.GetAtomWithIdx.return_value = corrupted_atom

    with pytest.raises(ModelInferenceError, match="Atom index mismatch at 0"):
        uninitialized_service.validate_mol_atomic_indices(mol, 3, mol_with_h, 9)


@pytest.mark.fast_rdkit
def test_validate_mol_atomic_indices_raises_on_atomic_num_mismatch(
    uninitialized_service: MolecularInferenceService,
) -> None:
    """Unhappy path 2 for `validate_mol_atomic_indices`: `heavy_atom.GetAtomicNum() != atom_in_h_mol.GetAtomicNum()`."""
    mol = Chem.MolFromSmiles("CCO")  # C, C, O (Z=8 at index 2)
    mol_wrong = Chem.MolFromSmiles("CCN")  # C, C, N (Z=7 at index 2)
    mol_wrong_h = Chem.AddHs(mol_wrong)

    with pytest.raises(ModelInferenceError, match="Atom type mismatch at index 2"):
        uninitialized_service.validate_mol_atomic_indices(
            mol, mol.GetNumAtoms(), mol_wrong_h, mol_wrong_h.GetNumAtoms()
        )


@pytest.mark.fast_rdkit
def test_validate_mol_atomic_indices_raises_when_appended_atom_is_not_hydrogen(
    uninitialized_service: MolecularInferenceService,
) -> None:
    """Unhappy path 3 for `validate_mol_atomic_indices`: appended atom at index >= K has atomic number != 1."""
    mol = Chem.MolFromSmiles("CC")  # 2 carbons (K=2)
    mol_extra_heavy = Chem.MolFromSmiles("CCO")  # Atom at index 2 is Oxygen (Z=8 != 1)

    with pytest.raises(ModelInferenceError, match="Expected Hydrogen at index 2, but got atomic number Z=8"):
        uninitialized_service.validate_mol_atomic_indices(
            mol, mol.GetNumAtoms(), mol_extra_heavy, mol_extra_heavy.GetNumAtoms()
        )
