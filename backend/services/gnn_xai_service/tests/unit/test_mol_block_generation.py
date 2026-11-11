from unittest.mock import patch

import pytest
from rdkit import Chem

from src.exceptions import ModelInferenceError
from src.mol_block_generation import mol_block_generation

# =====================================================================
# 1. mol_block_generation — Happy Paths & Fallbacks
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
        "F/C=C/F",
        "[2H]CCO",
    ],
)
def test_mol_block_generation_returns_valid_molblock_and_preserves_heavy_atom_prefix(
    smiles: str,
) -> None:
    """
    Verifies that `mol_block_generation` returns a valid V2000 MolBlock and `mol_with_h`
    where original heavy atoms stay at indices `0 .. K-1` and added hydrogens at `K .. N-1`.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    num_heavy = mol.GetNumAtoms()

    molblock, mol_with_h = mol_block_generation(mol)

    assert isinstance(molblock, str) and "V2000" in molblock and "M  END" in molblock
    assert mol_with_h.GetNumConformers() == 1
    assert mol_with_h.GetConformer().Is3D() is True

    total_atoms = mol_with_h.GetNumAtoms()
    assert total_atoms > num_heavy

    # Prefix 0 .. num_heavy - 1 matches original atoms; suffix num_heavy .. total_atoms - 1 are H (Z=1)
    for idx in range(num_heavy):
        assert (
            mol_with_h.GetAtomWithIdx(idx).GetAtomicNum()
            == mol.GetAtomWithIdx(idx).GetAtomicNum()
        )
    for idx in range(num_heavy, total_atoms):
        assert mol_with_h.GetAtomWithIdx(idx).GetAtomicNum() == 1


@pytest.mark.fast_rdkit
def test_mol_block_generation_is_deterministic_via_fixed_random_seed() -> None:
    mol_1 = Chem.MolFromSmiles("CC(C)O")
    mol_2 = Chem.MolFromSmiles("CC(C)O")
    assert mol_1 is not None and mol_2 is not None

    molblock_1, _ = mol_block_generation(mol_1)
    molblock_2, _ = mol_block_generation(mol_2)

    assert molblock_1 == molblock_2


@pytest.mark.fast_rdkit
def test_mol_block_generation_uses_random_coords_on_first_embed_failure() -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    real_embed = Chem.AllChem.EmbedMolecule
    call_count = 0

    def side_effect(m: Chem.Mol, params: object) -> int:
        nonlocal call_count
        call_count += 1
        assert getattr(params, "randomSeed", None) == 0xF00D
        if call_count == 1:
            assert getattr(params, "useRandomCoords", False) is False
            return -1
        assert getattr(params, "useRandomCoords", False) is True
        return real_embed(m, params)

    with patch(
        "src.mol_block_generation.AllChem.EmbedMolecule", side_effect=side_effect
    ):
        molblock, mol_with_h = mol_block_generation(mol)
        assert call_count == 2
        assert "V2000" in molblock
        assert mol_with_h.GetNumConformers() == 1


@pytest.mark.fast_rdkit
def test_mol_block_generation_auto_generates_2d_molblock_when_3d_embedding_fails() -> (
    None
):
    """
    When both ETKDGv3 and ETKDGv3(useRandomCoords=True) return -1 (e.g. strained pentalene
    "C1=CC2=CC=C1C=C2"), `Chem.MolToMolBlock` automatically generates 2D coordinates in the
    returned V2000 MolBlock without raising an error.
    """
    strained_mol = Chem.MolFromSmiles("C1=CC2=CC=C1C=C2")
    assert strained_mol is not None

    molblock, mol_with_h = mol_block_generation(strained_mol)
    assert isinstance(molblock, str) and "V2000" in molblock and "M  END" in molblock
    assert mol_with_h.GetNumConformers() == 0
    assert mol_with_h.GetNumAtoms() == 14


# =====================================================================
# 2. mol_block_generation — Unhappy Paths
# =====================================================================


@pytest.mark.fast_rdkit
def test_mol_block_generation_raises_on_none_mol() -> None:
    with pytest.raises(
        ModelInferenceError, match="Cannot generate MolBlock from None molecule"
    ):
        mol_block_generation(None)


@pytest.mark.fast_rdkit
def test_mol_block_generation_raises_when_mol_to_molblock_throws() -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    with patch(
        "src.mol_block_generation.Chem.MolToMolBlock",
        side_effect=RuntimeError("RDKit C++ error"),
    ):
        with pytest.raises(
            ModelInferenceError, match="Failed to generate MolBlock string for molecule"
        ):
            mol_block_generation(mol)
