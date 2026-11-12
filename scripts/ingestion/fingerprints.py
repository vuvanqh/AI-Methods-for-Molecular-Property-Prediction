from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from backend.app.src.rag.config import MORGAN_FINGERPRINT_SIZE as FINGERPRINT_BITS
from backend.app.src.rag.morgan_fingerprint_helper import (
    fingerprint_to_bitstring,
    generate_morgan_fingerprint,
)

# ChEMBL triggers thousands of harmless sanitization warnings
rdBase.DisableLog("rdApp.*")


def standardize_smiles(smiles: str) -> str | None:
    """Clean up and return canonical SMILES; None if RDKit can't parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # fingerprints are to serve as a structural similarity indicator so comparing parent molecular structures should be sufficient
    mol = rdMolStandardize.Cleanup(mol)

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def morgan_fingerprint(smiles: str) -> str:
    return fingerprint_to_bitstring(generate_morgan_fingerprint(smiles))


if __name__ == "__main__":
    assert standardize_smiles("CN(=O)=O") == "C[N+](=O)[O-]"  # nitro normalized
    assert (
        standardize_smiles("CC(=O)O[Na]") == "CC(=O)[O-].[Na+]"
    )  # metal disconnected; Cleanup keeps the counter-ion
    assert standardize_smiles("OCC") == standardize_smiles("CCO") == "CCO"
    assert standardize_smiles("not a smiles") is None
    fp = morgan_fingerprint("CCO")
    assert len(fp) == FINGERPRINT_BITS and set(fp) <= {"0", "1"} and "1" in fp
    assert morgan_fingerprint("CCO") != morgan_fingerprint("c1ccccc1")
    print("fingerprints self-check passed")
