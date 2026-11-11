import pytest
from rdkit import Chem

from src.exceptions import ModelInferenceError
from src.mol_block_generation import mol_block_generation


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
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    num_heavy = mol.GetNumAtoms()

    molblock, mol_with_h = mol_block_generation(mol)

    # Original molecule is not mutated
    assert mol.GetNumAtoms() == num_heavy

    assert isinstance(molblock, str) and "V2000" in molblock and "M  END" in molblock
    parsed = Chem.MolFromMolBlock(molblock, removeHs=False)
    assert parsed is not None

    total_atoms = mol_with_h.GetNumAtoms()
    assert parsed.GetNumAtoms() == total_atoms
    assert total_atoms > num_heavy

    # Heavy atoms stay at 0 .. num_heavy - 1; appended atoms at num_heavy .. total_atoms - 1 are H (Z=1)
    for idx in range(num_heavy):
        assert (
            mol_with_h.GetAtomWithIdx(idx).GetAtomicNum()
            == mol.GetAtomWithIdx(idx).GetAtomicNum()
        )
    for idx in range(num_heavy, total_atoms):
        assert mol_with_h.GetAtomWithIdx(idx).GetAtomicNum() == 1


def test_mol_block_generation_is_deterministic_across_repeated_calls() -> None:
    mol_1 = Chem.MolFromSmiles("CC(C)O")
    mol_2 = Chem.MolFromSmiles("CC(C)O")
    assert mol_1 is not None and mol_2 is not None

    molblock_1, _ = mol_block_generation(mol_1)
    molblock_2, _ = mol_block_generation(mol_2)

    assert molblock_1 == molblock_2


def test_mol_block_generation_produces_valid_molblock_when_3d_embedding_cannot_converge() -> (
    None
):
    strained_mol = Chem.MolFromSmiles("C1=CC2=CC=C1C=C2")
    assert strained_mol is not None

    molblock, mol_with_h = mol_block_generation(strained_mol)

    assert isinstance(molblock, str) and "V2000" in molblock and "M  END" in molblock
    parsed = Chem.MolFromMolBlock(molblock, removeHs=False)
    assert parsed is not None
    assert parsed.GetNumAtoms() == mol_with_h.GetNumAtoms()


def test_mol_block_generation_raises_on_none_molecule() -> None:
    with pytest.raises(ModelInferenceError):
        mol_block_generation(None)
