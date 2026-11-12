import pytest
from rdkit import DataStructs

from src.descriptors import describe_molecule
from src.model import MoleculeModel
from src.rag.config import MORGAN_FINGERPRINT_SIZE
from src.rag.morgan_fingerprint_helper import (
    bitstring_to_fingerprint,
    fingerprint_to_bitstring,
    generate_morgan_fingerprint,
    get_canonical_smiles,
)


@pytest.mark.parametrize(
    ("smiles", "canonical", "formula", "mass", "atoms", "bonds"),
    [
        ("OCC", "CCO", "C2H6O", 46.07, 9, 8),
        ("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O", "C9H8O4", 180.16, 21, 21),
        ("C[C@H](O)Cl", "C[C@H](O)Cl", "C2H5ClO", 80.51, 9, 8),
    ],
)
def test_describe_molecule_computes_expected_descriptors(
    smiles: str, canonical: str, formula: str, mass: float, atoms: int, bonds: int
) -> None:
    result = MoleculeModel(**describe_molecule(smiles))
    assert (result.smiles, result.formula, result.atom_count, result.bond_count) == (
        canonical,
        formula,
        atoms,
        bonds,
    )
    assert result.molar_mass == pytest.approx(mass, abs=1e-2)


def test_equivalent_smiles_produce_identical_descriptors_and_fingerprints() -> None:
    assert describe_molecule("CCO") == describe_molecule("OCC")
    assert get_canonical_smiles("c1ccccc1") == get_canonical_smiles("C1=CC=CC=C1")
    assert fingerprint_to_bitstring(
        generate_morgan_fingerprint("CCO")
    ) == fingerprint_to_bitstring(generate_morgan_fingerprint("OCC"))


def test_fingerprints_preserve_chirality_radius_and_round_trip_bitstrings() -> None:
    r_isomer, s_isomer = "C[C@H](O)Cl", "C[C@@H](O)Cl"
    assert get_canonical_smiles(r_isomer) != get_canonical_smiles(s_isomer)
    fp_r = generate_morgan_fingerprint(r_isomer)
    fp_s = generate_morgan_fingerprint(s_isomer)
    bits_r = fingerprint_to_bitstring(fp_r)
    assert bits_r != fingerprint_to_bitstring(fp_s)
    # 1-butanol and 1-pentanol share identical radius-1 environments and only diverge at radius >= 2
    assert fingerprint_to_bitstring(
        generate_morgan_fingerprint("CCCCO")
    ) != fingerprint_to_bitstring(generate_morgan_fingerprint("CCCCCO"))
    assert len(bits_r) == MORGAN_FINGERPRINT_SIZE
    assert DataStructs.TanimotoSimilarity(fp_r, bitstring_to_fingerprint(bits_r)) == 1.0


@pytest.mark.parametrize("bad_smiles", ["", "   ", "not_a_smiles", "C(C)(C)(C)(C)C"])
def test_invalid_smiles_raises_value_error(bad_smiles: str) -> None:
    with pytest.raises(ValueError):
        describe_molecule(bad_smiles)
    with pytest.raises(ValueError):
        generate_morgan_fingerprint(bad_smiles)
