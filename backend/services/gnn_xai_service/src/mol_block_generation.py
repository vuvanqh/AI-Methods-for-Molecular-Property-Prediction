from rdkit import Chem
from rdkit.Chem import AllChem

from .exceptions import ModelInferenceError


def mol_block_generation(mol: Chem.Mol) -> tuple[str, Chem.Mol]:
    """
    Generate a V2000 MolBlock string and hydrogen-added RDKit Mol for 3Dmol.js visualization.

    Attempts 3D conformer embedding with ETKDGv3 (seeded with 0xf00d), retries with
    random initial coordinates (`useRandomCoords=True`) if standard embedding returns -1,
    and relies on `Chem.MolToMolBlock`'s built-in 2D coordinate fallback if both 3D
    attempts fail. Original heavy-atom indices (0 .. N_heavy - 1) are preserved.
    """
    if mol is None:
        raise ModelInferenceError("Cannot generate MolBlock from None molecule.")

    # Append explicit hydrogens at indices >= mol.GetNumAtoms() without mutating `mol`
    mol_with_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D

    # Try standard ETKDGv3 distance-geometry embedding (returns -1 on failure)
    res = AllChem.EmbedMolecule(mol_with_h, params)

    # Fallback to random 3D box coordinates if standard embedding failed
    if res == -1:
        params.useRandomCoords = True
        AllChem.EmbedMolecule(mol_with_h, params)

    # Serialize to V2000 MolBlock (auto-generates 2D coords if 0 conformers exist)
    try:
        mol_block = Chem.MolToMolBlock(mol_with_h)
    except Exception as e:
        raise ModelInferenceError(
            "Failed to generate MolBlock string for molecule."
        ) from e

    return mol_block, mol_with_h
