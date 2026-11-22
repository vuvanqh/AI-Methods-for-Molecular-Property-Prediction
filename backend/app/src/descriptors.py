from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from .rag.morgan_fingerprint_helper import smiles_to_mol

# rdMolStandardize.Cleanup logs normalizer initialization banners at info level on every call
rdBase.DisableLog("rdApp.info")


def standardize_smiles(smiles: str) -> str:
    """Clean up and return canonical isomeric SMILES, matching corpus ingestion."""
    mol = smiles_to_mol(smiles)
    cleaned = rdMolStandardize.Cleanup(mol)
    return Chem.MolToSmiles(cleaned, canonical=True, isomericSmiles=True)


def describe_molecule(smiles: str) -> dict:
    """Structural descriptors shown in the molecule overview card."""
    mol = smiles_to_mol(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("RDKit could not parse the provided SMILES.")

    # counts include hydrogens, to match the atoms drawn from the 3D molblock as rdkit ommits it by default
    mol_with_h = Chem.AddHs(mol)
    return {
        "smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molar_mass": Descriptors.MolWt(mol),
        "atom_count": mol_with_h.GetNumAtoms(),
        "bond_count": mol_with_h.GetNumBonds(),
    }
