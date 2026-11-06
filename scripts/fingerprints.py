from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

# ChEMBL triggers thousands of harmless sanitization warnings
rdBase.DisableLog("rdApp.*")

_morgan = rdFingerprintGenerator.GetMorganGenerator(
    radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS
)
_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize_smiles(smiles: str) -> str | None:
    """Clean up and return canonical SMILES; None if RDKit can't parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _largest_fragment.choose(mol)
        mol = _uncharger.uncharge(mol)
    except (
        Exception
    ):  # RDKit raises assorted sanitization errors on exotic structures; skip them
        return None
    return Chem.MolToSmiles(mol)


def morgan_fingerprint(smiles: str) -> str:
    return _morgan.GetFingerprint(Chem.MolFromSmiles(smiles)).ToBitString()


if __name__ == "__main__":
    assert (
        standardize_smiles("CC(=O)[O-].[Na+]") == "CC(=O)O"
    )  # sodium acetate -> acetic acid
    assert standardize_smiles("OCC") == standardize_smiles("CCO") == "CCO"
    assert standardize_smiles("not a smiles") is None
    fp = morgan_fingerprint("CCO")
    assert len(fp) == FINGERPRINT_BITS and set(fp) <= {"0", "1"} and "1" in fp
    assert morgan_fingerprint("CCO") != morgan_fingerprint("c1ccccc1")
    print("fingerprints self-check passed")
