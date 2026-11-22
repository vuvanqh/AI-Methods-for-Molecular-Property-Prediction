import numpy as np
import pytest
from rdkit import DataStructs
from sqlalchemy.orm import Session, sessionmaker

from src.rag import RetrievalType, Retriever
from src.rag.embeddings.base_embeddings import BaseEmbeddings
from src.rag.morgan_fingerprint_helper import generate_morgan_fingerprint
from src.rag.retrieval import FingerprintRetrieval, SemanticRetrieval
from tests.integration.conftest import seed_molecule

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("isolated_rag_settings")]


class DeterministicEmbeddings(BaseEmbeddings):
    name, version = "unimol", "unimolv1"

    def __init__(self, vectors: dict[str, list[float]]):
        super().__init__(device="cpu")
        self._vectors = vectors

    def load(self):
        pass

    def unload(self):
        pass

    def embedd(self, smiles: str) -> np.ndarray:
        return np.array(self._vectors[smiles], dtype=np.float32)


def _vec(x: float, y: float, dim: int = 16) -> list[float]:
    return [x, y, *([0.0] * (dim - 2))]


def test_fingerprint_retrieval_ranks_by_tanimoto_excludes_query_and_loads_sources(
    pg_session_factory: sessionmaker[Session], clean_db: None
) -> None:
    with pg_session_factory() as db:
        seed_molecule(db, "CCO", sources=[("ChEMBL", "CHEMBL545")])
        seed_molecule(db, "CCCO", sources=[("ChEMBL", "CHEMBL14688")])
        seed_molecule(db, "c1ccccc1", sources=[("ChEMBL", "CHEMBL277500")])

    # Query non-canonical "OCC" -> must still exclude canonical "CCO"
    with pg_session_factory() as db:
        results = FingerprintRetrieval(db).retrieve("OCC", top_k=5)

    assert [r.molecule.standardized_smiles for r in results] == ["CCCO", "c1ccccc1"]
    fp_q = generate_morgan_fingerprint("CCO")
    assert results[0].score == pytest.approx(
        DataStructs.TanimotoSimilarity(fp_q, generate_morgan_fingerprint("CCCO")),
        abs=1e-6,
    )
    assert results[1].score == pytest.approx(
        DataStructs.TanimotoSimilarity(fp_q, generate_morgan_fingerprint("c1ccccc1")),
        abs=1e-6,
    )
    # `selectinload` keeps `molecule_sources` readable after the session closes
    assert results[0].molecule.molecule_sources[0].source_identifier == "CHEMBL14688"


def test_fingerprint_retrieval_empty_corpus_and_top_k_boundary(
    db_session: Session,
) -> None:
    assert FingerprintRetrieval(db_session).retrieve("CCO", top_k=5) == []

    for smiles in ("CCCO", "CCCCO", "c1ccccc1"):
        seed_molecule(db_session, smiles)

    top_two = FingerprintRetrieval(db_session).retrieve("CCO", top_k=2)
    assert len(top_two) == 2 and top_two[0].score >= top_two[1].score


def test_semantic_retrieval_ranks_by_cosine_filters_mixed_dim_models_and_loads_sources(
    pg_session_factory: sessionmaker[Session], clean_db: None
) -> None:
    q = _vec(1.0, 0.0, dim=16)
    with pg_session_factory() as db:
        seed_molecule(db, "CCO", embeddings=[("unimol", "unimolv1", q)])
        seed_molecule(
            db,
            "CCCO",
            sources=[("ChEMBL", "CHEMBL14688")],
            embeddings=[("unimol", "unimolv1", _vec(0.8, 0.6, dim=16))],  # cos = 0.8
        )
        seed_molecule(
            db,
            "c1ccccc1",
            embeddings=[("unimol", "unimolv1", _vec(0.0, 1.0, dim=16))],  # cos = 0.0
        )
        # Different version (same dim) AND different encoder with a DIFFERENT vector dimension (8-d)
        seed_molecule(
            db,
            "CCN",
            embeddings=[
                ("unimol", "unimolv2", q),
                ("other_encoder", "v1", _vec(1.0, 0.0, dim=8)),
            ],
        )

    # Close session before asserting sources to prove `selectinload` eager loading
    with pg_session_factory() as db:
        results = SemanticRetrieval(db, DeterministicEmbeddings({"OCC": q})).retrieve(
            "OCC", top_k=5
        )

    assert [r.molecule.standardized_smiles for r in results] == ["CCCO", "c1ccccc1"]
    assert results[0].score == pytest.approx(0.8, abs=1e-5)
    assert results[1].score == pytest.approx(0.0, abs=1e-5)
    assert results[0].molecule.molecule_sources[0].source_identifier == "CHEMBL14688"


def test_hybrid_retrieval_fuses_ranking_cutoff_self_exclusion_and_partial_embeddings(
    pg_session_factory: sessionmaker[Session], clean_db: None
) -> None:
    q = _vec(1.0, 0.0, dim=16)
    with pg_session_factory() as db:
        # Query molecule itself -> must be excluded
        seed_molecule(db, "CCO", embeddings=[("unimol", "unimolv1", q)])
        # Top in BOTH fingerprint (propanol vs ethanol) and semantic (cos = 0.9) -> unambiguous #1
        seed_molecule(
            db,
            "CCCO",
            sources=[("ChEMBL", "CHEMBL14688")],
            embeddings=[("unimol", "unimolv1", _vec(0.9, 0.43588989, dim=16))],
        )
        # Ranks #3 in fingerprint (Tanimoto = 0.0) but #2 in semantic (cos = 0.5) ->
        # RRF combines both signals (0.5/63 + 0.5/62 > 0.5/62), beating fp-only CCCCCCCCC
        seed_molecule(
            db,
            "c1ccccc1",
            sources=[("ChEMBL", "CHEMBL277500")],
            embeddings=[("unimol", "unimolv1", _vec(0.5, 0.8660254, dim=16))],
        )
        # Fingerprint-only (no active embedding): ranks #2 in fingerprint, #3 after RRF fusion
        seed_molecule(db, "CCCCCCCCC", sources=[("ChEMBL", "CHEMBL_FP_ONLY")])

    with pg_session_factory() as db:
        hybrid = Retriever(
            RetrievalType.FUSION, DeterministicEmbeddings({"CCO": q})
        ).create_retriever(db)
        top_three = hybrid.retrieve("CCO", top_k=3)
        top_two = hybrid.retrieve("CCO", top_k=2)

    assert [r.molecule.standardized_smiles for r in top_three] == [
        "CCCO",
        "c1ccccc1",
        "CCCCCCCCC",
    ]
    assert [r.molecule.standardized_smiles for r in top_two] == ["CCCO", "c1ccccc1"]
    fp_q = generate_morgan_fingerprint("CCO")
    expected_top_score = (
        0.5 * DataStructs.TanimotoSimilarity(fp_q, generate_morgan_fingerprint("CCCO"))
        + 0.5 * 0.9
    )
    expected_fp_only_score = 0.5 * DataStructs.TanimotoSimilarity(
        fp_q, generate_morgan_fingerprint("CCCCCCCCC")
    )
    assert top_three[0].score == pytest.approx(expected_top_score, abs=1e-5)
    assert top_three[2].score == pytest.approx(expected_fp_only_score, abs=1e-5)
    # `selectinload` keeps sources accessible after closing session
    assert top_three[0].molecule.molecule_sources[0].source_identifier == "CHEMBL14688"
    assert (
        top_three[2].molecule.molecule_sources[0].source_identifier == "CHEMBL_FP_ONLY"
    )
