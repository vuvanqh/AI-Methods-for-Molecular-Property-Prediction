from rdkit import Chem
from rdkit.Chem import AllChem

def mol_block_generation(mol):
    '''
    3d mol block generation for visualization using 3dmol.
    https://www.rdkit.org/docs/GettingStartedInPython.html

    '''
    # H atoms should be appended, not added in between
    # representation without H is "the beginning" of this representation
    # original mol is not modified
    mol_with_h = Chem.AddHs(mol) 
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d # optional random seed for reproducibility
    AllChem.EmbedMolecule(mol_with_h, params)
    
    mol_block = Chem.MolToMolBlock(mol_with_h)
    return mol_block