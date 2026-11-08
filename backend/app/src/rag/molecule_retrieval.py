from sqlalchemy.orm import Session
from .embeddings.base_embeddings import BaseEmbeddings
from .retrieval_type import RetrievalType
from .retrieval.fingerprint_retrieval import FingerprintRetrieval
from .retrieval.semantic_retrieval import SemanticRetrieval
from .retrieval.retrieval_base import Retrieval
from .retrieval.hybrid_retrieval import HybridRetrieval


def create_retriever(
    retrieval_type: RetrievalType,
    db: Session,
    embedding_model: BaseEmbeddings | None = None,
) -> Retrieval:
    if retrieval_type == RetrievalType.FINGERPRINT:
        return FingerprintRetrieval(db)

    if embedding_model is None:
        raise ValueError(
            "Embedding model must be provided for semantic or fusion retrieval."
        )

    semantic = SemanticRetrieval(db, embedding_model)
    if retrieval_type == RetrievalType.SEMANTIC:
        return semantic

    return HybridRetrieval(FingerprintRetrieval(db), semantic)
