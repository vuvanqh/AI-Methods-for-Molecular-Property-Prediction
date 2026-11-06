from .morgan_fingerprint_helper import get_canonical_smiles, generate_morgan_fingerprint, fingerprint_to_bitstring
from sqlalchemy.orm import Session
from ..db.models.molecule import Molecule
from sqlalchemy import select, selectinload


def retrieve_similar_molecules_with_sources(smiles: str, top_k: int, db: Session) -> list:
    """
    Retrieve similar molecules from the database based on Morgan fingerprints.
    """
    # Generate the Morgan fingerprint for the input SMILES
    canonical_smiles = get_canonical_smiles(smiles)
    input_fingerprint = generate_morgan_fingerprint(canonical_smiles)

    # Convert the fingerprint to a bitstring for database querying
    input_bitstring = fingerprint_to_bitstring(input_fingerprint)

    # tanimoto similarity = 1 - jaccard distance here 
    dist = Molecule.morgan_fingerprint.jaccard_distance(input_bitstring)

    # label for easier extraction from result
    similarity = (1- dist).label("similarity_score")

    stmt = select(Molecule,similarity).options(selectinload(Molecule.molecule_sources)).order_by(similarity.desc()).limit(top_k)

    similar_molecules = db.execute(stmt).all()

    return similar_molecules


