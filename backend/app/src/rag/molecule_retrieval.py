from sqlalchemy.orm import Session

from .embeddings.base_embeddings import BaseEmbeddings
from .retrieval.fingerprint_retrieval import FingerprintRetrieval
from .retrieval.hybrid_retrieval import HybridRetrieval
from .retrieval.retrieval_base import Retrieval
from .retrieval.semantic_retrieval import SemanticRetrieval
from .retrieval_type import RetrievalType


# TO-DOOOO!!!!!
class Retriever:
    def __init__(
        self,
        retrieval_type: RetrievalType,
        embedding_model: BaseEmbeddings | None = None,
    ):
        if retrieval_type != RetrievalType.FINGERPRINT and embedding_model is None:
            raise ValueError(
                "Embedding model must be provided for semantic or fusion retrieval."
            )

        self.retrieval_type = retrieval_type
        self.embedding_model = embedding_model

    def create_retriever(
        self,
        db: Session,
    ) -> Retrieval:
        if self.retrieval_type == RetrievalType.FINGERPRINT:
            return FingerprintRetrieval(db)

        semantic = SemanticRetrieval(db, self.embedding_model)
        if self.retrieval_type == RetrievalType.SEMANTIC:
            return semantic

        return HybridRetrieval(FingerprintRetrieval(db), semantic)
