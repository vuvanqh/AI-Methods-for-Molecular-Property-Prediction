from rdkit import Chem
from rdkit.Chem import AllChem
from .exceptions import ModelInferenceError

def mol_block_generation(mol: Chem.Mol) -> tuple[str, Chem.Mol]:
    '''
    3d mol block generation for visualization using 3dmol.
    Returns (mol_block_string, mol_with_h).
    https://www.rdkit.org/docs/GettingStartedInPython.html
    '''
    if mol is None:
        raise ModelInferenceError("Cannot generate MolBlock from None molecule.")

    # H atoms should be appended, not added in between
    # representation without H is "the beginning" of this representation
    # original mol is not modified
    mol_with_h = Chem.AddHs(mol) 
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d # optional random seed for reproducibility
    
    res = AllChem.EmbedMolecule(mol_with_h, params)
    if res != 0:
        raise ModelInferenceError("Failed to generate 3D coordinates for molecule.")

    # could not find the failure path return value, only some examples raising errors
    try:
        mol_block = Chem.MolToMolBlock(mol_with_h)
    except Exception as e:
        raise ModelInferenceError("Failed to generate MolBlock string for molecule.")

    return mol_block, mol_with_h

