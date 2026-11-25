"""Unit tests for `src.mol_block_generation`: V2000 serialization, atom alignment, and 3D->2D fallback."""

from typing import Any

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from src.exceptions import ModelInferenceError
from src.mol_block_generation import mol_block_generation


@pytest.mark.parametrize(
    ("smiles", "expects_added_h"),
    [
        ("CCO", True),
        ("CCO.[Na+]", True),
        ("[2H]CCO", True),
        ("O=C=O", False),
        ("O=P(O)(O)c1ccc(Br)cc1", True),
        ("FC(F)(F)S(=O)(=O)NCC[NH3+].[Cl-]", True),
    ],
)
def test_preserves_heavy_atom_prefix_and_round_trips_v2000(
    smiles: str, expects_added_h: bool
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    num_heavy, orig_confs = mol.GetNumAtoms(), mol.GetNumConformers()

    molblock, mol_with_h = mol_block_generation(mol)

    assert mol.GetNumAtoms() == num_heavy and mol.GetNumConformers() == orig_confs
    assert "V2000" in molblock and "M  END" in molblock

    parsed = Chem.MolFromMolBlock(molblock, removeHs=False)
    assert parsed is not None
    total = mol_with_h.GetNumAtoms()
    assert parsed.GetNumAtoms() == total
    assert (total > num_heavy) is expects_added_h

    heavy_z = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_heavy)]
    assert [
        mol_with_h.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_heavy)
    ] == heavy_z
    assert [
        parsed.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_heavy)
    ] == heavy_z
    assert all(
        mol_with_h.GetAtomWithIdx(i).GetAtomicNum() == 1
        and parsed.GetAtomWithIdx(i).GetAtomicNum() == 1
        for i in range(num_heavy, total)
    )


def test_embeds_deterministic_3d_conformer() -> None:
    mol_1, mol_2 = Chem.MolFromSmiles("CC(C)O"), Chem.MolFromSmiles("CC(C)O")
    assert mol_1 is not None and mol_2 is not None

    molblock_1, mol_with_h = mol_block_generation(mol_1)
    molblock_2, _ = mol_block_generation(mol_2)

    assert mol_with_h.GetNumConformers() == 1
    conf = mol_with_h.GetConformer()
    assert conf.Is3D() is True and "3D" in molblock_1
    assert any(
        abs(conf.GetAtomPosition(i).z) > 1e-4 for i in range(mol_with_h.GetNumAtoms())
    )
    assert molblock_1 == molblock_2


def test_retries_with_random_coords_after_standard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    random_coords_flags: list[bool] = []
    real_embed = AllChem.EmbedMolecule

    def _fail_first(target_mol: Chem.Mol, params: Any) -> int:
        random_coords_flags.append(params.useRandomCoords)
        return -1 if len(random_coords_flags) == 1 else real_embed(target_mol, params)

    monkeypatch.setattr(AllChem, "EmbedMolecule", _fail_first)

    molblock, mol_with_h = mol_block_generation(mol)

    assert random_coords_flags == [False, True]
    assert mol_with_h.GetNumConformers() == 1 and mol_with_h.GetConformer().Is3D()
    assert "V2000" in molblock and "3D" in molblock
    parsed = Chem.MolFromMolBlock(molblock, removeHs=False)
    assert parsed is not None and parsed.GetNumAtoms() == mol_with_h.GetNumAtoms()


def test_falls_back_to_2d_organically_for_strained_molecule() -> None:
    mol = Chem.MolFromSmiles("C1=CC2=CC=C1C=C2")
    assert mol is not None
    num_heavy = mol.GetNumAtoms()

    molblock, mol_with_h = mol_block_generation(mol)

    assert mol_with_h.GetNumConformers() == 0
    assert "V2000" in molblock and "M  END" in molblock and "2D" in molblock
    parsed = Chem.MolFromMolBlock(molblock, removeHs=False)
    assert parsed is not None
    total = mol_with_h.GetNumAtoms()
    assert parsed.GetNumAtoms() == total
    conf = parsed.GetConformer()
    assert conf.Is3D() is False
    assert all(conf.GetAtomPosition(i).z == 0.0 for i in range(total))
    assert [parsed.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_heavy)] == [
        mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_heavy)
    ]
    assert all(
        parsed.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in range(num_heavy, total)
    )


def test_wraps_mol_to_molblock_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    root_cause = RuntimeError("rdkit serialization failure")
    monkeypatch.setattr(
        Chem, "MolToMolBlock", lambda _mol: (_ for _ in ()).throw(root_cause)
    )

    with pytest.raises(ModelInferenceError) as exc_info:
        mol_block_generation(mol)

    assert exc_info.value.__cause__ is root_cause


def test_raises_on_none_molecule() -> None:
    with pytest.raises(ModelInferenceError):
        mol_block_generation(None)  # type: ignore[arg-type]
