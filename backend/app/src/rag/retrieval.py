from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..db.models.molecule import Molecule
from .morgan_fingerprint_helper import (
    fingerprint_to_bitstring,
    generate_morgan_fingerprint,
    get_canonical_smiles,
)


def retrieve_similar_molecules_with_sources(
    smiles: str, top_k: int, db: Session
) -> list:
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
    similarity = (1 - dist).label("similarity_score")
    query = (
        select(Molecule, similarity)
        .options(selectinload(Molecule.molecule_sources))
        .order_by(similarity.desc())
        .limit(top_k)
    )

    similar_molecules = db.execute(query).all()

    return list(similar_molecules)
