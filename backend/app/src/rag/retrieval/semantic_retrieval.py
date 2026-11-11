from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ...db.models import Molecule, MoleculeEmbedding
from ..embeddings.base_embeddings import BaseEmbeddings
from ..morgan_fingerprint_helper import get_canonical_smiles
from ..result_types.retrieved_molecules import RetrievedMolecule
from .retrieval_base import Retrieval


class SemanticRetrieval(Retrieval):
    def __init__(self, db: Session, embedding_model: BaseEmbeddings):
        self.db = db
        self.embedding_model = embedding_model

    def retrieve(self, smiles: str, top_k: int) -> list[RetrievedMolecule]:
        """
        Retrieve similar molecules from the database based on semantic embeddings.
        """
        canonical_smiles = get_canonical_smiles(smiles)

        # Generate the embedding for the input SMILES
        input_embedding = self.embedding_model.embedd(smiles)

        dist = MoleculeEmbedding.embedding.cosine_distance(input_embedding)

        similarity_score = (1 - dist).label("similarity_score")

        query = (
            select(Molecule, similarity_score)
            .join(MoleculeEmbedding, Molecule.id == MoleculeEmbedding.molecule_id)
            .where(
                MoleculeEmbedding.embedding_model == self.embedding_model.name,
                MoleculeEmbedding.embedding_model_version
                == self.embedding_model.version,
                Molecule.standardized_smiles != canonical_smiles,
            )
            .options(selectinload(Molecule.molecule_sources))
            .order_by(similarity_score.desc())
            .limit(top_k)
        )

        similar_molecules = self.db.execute(query).all()

        return [
            RetrievedMolecule(molecule=molecule, score=score)
            for molecule, score in similar_molecules
        ]
