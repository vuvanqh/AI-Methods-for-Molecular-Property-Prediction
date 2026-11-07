
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .config import MORGAN_FINGERPRINT_RADIUS, MORGAN_FINGERPRINT_SIZE


def smiles_to_mol(smiles):
    """
    Validate SMILES and return an RDKit molecule.
    """

    if not smiles or not smiles.strip():
        raise ValueError("SMILES cannot be empty.")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if mol.GetNumAtoms() == 0:
        raise ValueError(f"SMILES has no atoms: {smiles}")

    return mol


def get_canonical_smiles(smiles):
    """
    Convert SMILES into canonical isomeric SMILES.
    """

    mol = smiles_to_mol(smiles)

    params = Chem.SmilesWriteParams()

    params.canonical = True
    params.doIsomericSmiles = True

    return Chem.MolToSmiles(mol, params)


def generate_morgan_fingerprint(smiles: str) -> DataStructs.ExplicitBitVect:
    """
    Generate a binary Morgan fingerprint.
    """

    mol = smiles_to_mol(smiles)

    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_FINGERPRINT_RADIUS,
    includeChirality=True,
    fpSize=MORGAN_FINGERPRINT_SIZE
)
    return morgan_generator.GetFingerprint(mol)


def fingerprint_to_bitstring(fingerprint):
    """
    Convert RDKit fingerprint to a string of 0s and 1s for the database.
    """

    return fingerprint.ToBitString()


def bitstring_to_fingerprint(bitstring): # is it used anywhere? todo
    """
    Reconstruct RDKit fingerprint from PostgreSQL BIT.
    """
    return DataStructs.CreateFromBitString(bitstring)


