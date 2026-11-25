from collections.abc import Callable

import pytest

from src.descriptors import describe_molecule, standardize_smiles
from src.model import MoleculeModel


@pytest.mark.parametrize(
    ("smiles", "canonical", "formula", "mass", "atoms", "bonds"),
    [
        ("OCC", "CCO", "C2H6O", 46.07, 9, 8),
        ("[Na+]", "[Na+]", "Na+", 22.99, 1, 0),
        ("[H]O[H]", "O", "H2O", 18.02, 3, 2),
        (
            "O=P(O)(O)c1ccc(Br)cc1",
            "O=P(O)(O)c1ccc(Br)cc1",
            "C6H6BrO3P",
            236.99,
            17,
            17,
        ),
        (
            "C[NH3+].FC(F)(F)S(=O)(=O)[O-]",
            "C[NH3+].O=S(=O)([O-])C(F)(F)F",
            "C2H6F3NO3S",
            181.14,
            16,
            14,
        ),
    ],
)
def test_describe_molecule_computes_descriptors_with_hydrogens(
    smiles: str, canonical: str, formula: str, mass: float, atoms: int, bonds: int
) -> None:
    parsed = MoleculeModel(**describe_molecule(smiles))
    assert (parsed.smiles, parsed.formula, parsed.atom_count, parsed.bond_count) == (
        canonical,
        formula,
        atoms,
        bonds,
    )
    assert parsed.molar_mass == pytest.approx(mass, abs=1e-2)


@pytest.mark.parametrize(
    ("raw_smiles", "expected"),
    [
        ("CC(=O)O[Na]", "CC(=O)[O-].[Na+]"),  # Cleanup disconnects metal salt
        ("OCC", "CCO"),  # canonical=True
        ("C[C@H](O)Cl", "C[C@H](O)Cl"),  # isomericSmiles=True
        ("[Cl-].C[NH3+]", "C[NH3+].[Cl-]"),  # organic salt keeps counterion
    ],
)
def test_standardize_smiles_cleans_canonicalizes_and_preserves_chirality(
    raw_smiles: str, expected: str
) -> None:
    assert standardize_smiles(raw_smiles) == expected


@pytest.mark.parametrize("fn", [describe_molecule, standardize_smiles])
@pytest.mark.parametrize("bad_smiles", ["", "not_a_smiles"])
def test_descriptor_functions_reject_invalid_smiles(
    fn: Callable[[str], object], bad_smiles: str
) -> None:
    with pytest.raises(ValueError):
        fn(bad_smiles)
