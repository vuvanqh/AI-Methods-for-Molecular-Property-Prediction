import pytest
from rdkit import DataStructs

from src.rag.morgan_fingerprint_helper import (
    bitstring_to_fingerprint,
    fingerprint_to_bitstring,
    generate_morgan_fingerprint,
    get_canonical_smiles,
    get_molecular_formula,
    smiles_to_mol,
)


@pytest.mark.parametrize(("smiles", "expected_heavy_atoms"), [("CCO", 3), ("[Na+]", 1)])
def test_smiles_to_mol_parses_valid_smiles(
    smiles: str, expected_heavy_atoms: int
) -> None:
    assert smiles_to_mol(smiles).GetNumAtoms() == expected_heavy_atoms


@pytest.mark.parametrize(
    ("bad_smiles", "match"),
    [
        ("", "empty"),
        ("   ", "empty"),
        (None, "empty"),
        ("not_a_smiles", "Invalid SMILES"),
        ("C(C)(C)(C)(C)C", "Invalid SMILES"),
    ],
)
def test_smiles_to_mol_rejects_invalid_smiles(
    bad_smiles: str | None, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        smiles_to_mol(bad_smiles)


def test_get_molecular_formula_handles_valid_and_invalid_smiles() -> None:
    assert get_molecular_formula("CCO") == "C2H6O"
    assert get_molecular_formula("") == ""
    assert get_molecular_formula("invalid_smiles") == ""


def test_get_canonical_smiles_normalizes_encoding_and_preserves_chirality() -> None:
    assert get_canonical_smiles("CCO") == get_canonical_smiles("OCC")
    assert get_canonical_smiles("C[C@H](O)Cl") != get_canonical_smiles("C[C@@H](O)Cl")


def test_morgan_fingerprint_distinguishes_chiral_enantiomers() -> None:
    fp_r = generate_morgan_fingerprint("C[C@H](O)Cl")
    fp_s = generate_morgan_fingerprint("C[C@@H](O)Cl")

    assert fingerprint_to_bitstring(fp_r) != fingerprint_to_bitstring(fp_s)


def test_morgan_fingerprint_captures_radius_two_neighbourhoods() -> None:
    # 1-butanol and 1-pentanol share identical radius-1 environments and diverge at radius >= 2
    fp_butanol = generate_morgan_fingerprint("CCCCO")
    fp_pentanol = generate_morgan_fingerprint("CCCCCO")

    assert fingerprint_to_bitstring(fp_butanol) != fingerprint_to_bitstring(fp_pentanol)


def test_morgan_fingerprint_bitstring_format_equivalence_and_round_trip() -> None:
    fp_cco = generate_morgan_fingerprint("CCO")
    bits_cco = fingerprint_to_bitstring(fp_cco)
    bits_occ = fingerprint_to_bitstring(generate_morgan_fingerprint("OCC"))
    reconstructed = bitstring_to_fingerprint(bits_cco)

    assert len(bits_cco) == 2048
    assert set(bits_cco) <= {"0", "1"}
    assert bits_cco == bits_occ
    assert fingerprint_to_bitstring(reconstructed) == bits_cco
    assert DataStructs.TanimotoSimilarity(fp_cco, reconstructed) == 1.0
