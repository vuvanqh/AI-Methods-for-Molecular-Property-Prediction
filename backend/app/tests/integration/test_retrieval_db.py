import numpy as np
import pytest
from rdkit import DataStructs
from sqlalchemy.exc import DataError
from sqlalchemy.orm import Session, sessionmaker

from src.rag import RetrievalType, Retriever
from src.rag.embeddings.base_embeddings import BaseEmbeddings
from src.rag.morgan_fingerprint_helper import generate_morgan_fingerprint
from src.rag.retrieval import FingerprintRetrieval, SemanticRetrieval
from tests.pg_fixtures import seed_molecule

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


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def _tanimoto(smiles_a: str, smiles_b: str) -> float:
    return DataStructs.TanimotoSimilarity(
        generate_morgan_fingerprint(smiles_a), generate_morgan_fingerprint(smiles_b)
    )


def test_fingerprint_retrieval_ranks_by_exact_tanimoto_against_rdkit_and_caps_top_k(
    db_session: Session,
) -> None:
    candidates = ["CCCO", "CCCCO", "CC(=O)O", "c1ccccc1"]
    for smiles in candidates:
        seed_molecule(db_session, smiles)

    retriever = FingerprintRetrieval(db_session)
    full = retriever.retrieve("CCO", top_k=10)
    capped = retriever.retrieve("CCO", top_k=2)

    expected = sorted(
        ((s, _tanimoto("CCO", s)) for s in candidates), key=lambda p: p[1], reverse=True
    )
    assert [r.molecule.standardized_smiles for r in full] == [s for s, _ in expected]
    assert [r.score for r in full] == pytest.approx([s for _, s in expected], abs=1e-6)
    assert [r.molecule.standardized_smiles for r in capped] == [
        s for s, _ in expected[:2]
    ]
    assert [r.score for r in capped] == pytest.approx(
        [s for _, s in expected[:2]], abs=1e-6
    )


@pytest.mark.parametrize(
    ("stored_smiles", "query_smiles"),
    [("CCO", "OCC"), ("Oc1ccccc1", "c1ccc(O)cc1")],
    ids=["ethanol_reversed", "phenol_ring_permutation"],
)
def test_fingerprint_retrieval_excludes_query_across_canonical_smiles_variants(
    db_session: Session, stored_smiles: str, query_smiles: str
) -> None:
    seed_molecule(db_session, stored_smiles)
    seed_molecule(db_session, "CCCO")

    results = FingerprintRetrieval(db_session).retrieve(query_smiles, top_k=5)

    assert [r.molecule.standardized_smiles for r in results] == ["CCCO"]
    assert results[0].score == pytest.approx(_tanimoto(stored_smiles, "CCCO"), abs=1e-6)


def test_fingerprint_retrieval_eager_loads_all_sources_beyond_session_close(
    pg_session_factory: sessionmaker[Session], clean_db: None
) -> None:
    with pg_session_factory() as db:
        seed_molecule(
            db, "CCCO", sources=[("ChEMBL", "CHEMBL14688"), ("PubChem", "1031")]
        )
    with pg_session_factory() as db:
        results = FingerprintRetrieval(db).retrieve("CCO", top_k=5)

    assert len(results) == 1
    assert {
        (s.provider, s.source_identifier) for s in results[0].molecule.molecule_sources
    } == {("ChEMBL", "CHEMBL14688"), ("PubChem", "1031")}


def test_fingerprint_retrieval_returns_empty_for_empty_or_self_only_corpus(
    db_session: Session,
) -> None:
    assert FingerprintRetrieval(db_session).retrieve("CCO", top_k=5) == []
    seed_molecule(db_session, "CCO")
    assert FingerprintRetrieval(db_session).retrieve("OCC", top_k=5) == []


def test_semantic_retrieval_ranks_by_exact_cosine_and_caps_top_k(
    db_session: Session,
) -> None:
    query_vec = [3.0, 0.0, *([0.0] * 14)]  # Non-unit vector; ||q|| = 3
    v_close = [4.0, 3.0, *([0.0] * 14)]  # cos = 0.8
    v_ortho = [0.0, 5.0, *([0.0] * 14)]  # cos = 0.0
    v_opposite = [-2.0, 0.0, *([0.0] * 14)]  # cos = -1.0

    seed_molecule(db_session, "CCCO", embeddings=[("unimol", "unimolv1", v_close)])
    seed_molecule(db_session, "c1ccccc1", embeddings=[("unimol", "unimolv1", v_ortho)])
    seed_molecule(db_session, "CCN", embeddings=[("unimol", "unimolv1", v_opposite)])

    retriever = SemanticRetrieval(
        db_session, DeterministicEmbeddings({"CCO": query_vec})
    )
    full = retriever.retrieve("CCO", top_k=5)
    top_one = retriever.retrieve("CCO", top_k=1)

    assert [r.molecule.standardized_smiles for r in full] == ["CCCO", "c1ccccc1", "CCN"]
    assert [r.score for r in full] == pytest.approx(
        [_cosine(query_vec, v) for v in (v_close, v_ortho, v_opposite)], abs=1e-5
    )
    assert [r.molecule.standardized_smiles for r in top_one] == ["CCCO"]
    assert top_one[0].score == pytest.approx(_cosine(query_vec, v_close), abs=1e-5)


def test_semantic_retrieval_filters_by_model_version_and_dimension_without_duplicates(
    db_session: Session,
) -> None:
    q = _vec(1.0, 0.0, dim=16)
    # Matching molecule also holds wrong-version and wrong-model/dim embeddings
    # to catch duplicate-join or missing-filter bugs.
    seed_molecule(
        db_session,
        "CCCO",
        embeddings=[
            ("unimol", "unimolv1", _vec(0.8, 0.6, dim=16)),
            ("unimol", "unimolv2", q),
            ("chemberta", "unimolv1", _vec(1.0, 0.0, dim=8)),
        ],
    )
    seed_molecule(
        db_session,
        "CCN",
        embeddings=[
            ("unimol", "unimolv2", q),
            ("chemberta", "unimolv1", _vec(1.0, 0.0, dim=8)),
        ],
    )
    seed_molecule(db_session, "CCCCO")

    results = SemanticRetrieval(
        db_session, DeterministicEmbeddings({"CCO": q})
    ).retrieve("CCO", top_k=5)

    assert [r.molecule.standardized_smiles for r in results] == ["CCCO"]
    assert results[0].score == pytest.approx(0.8, abs=1e-5)


def test_semantic_retrieval_excludes_query_across_canonical_smiles_variants(
    db_session: Session,
) -> None:
    q = _vec(1.0, 0.0, dim=16)
    seed_molecule(db_session, "CCO", embeddings=[("unimol", "unimolv1", q)])
    seed_molecule(
        db_session, "CCCO", embeddings=[("unimol", "unimolv1", _vec(0.6, 0.8, dim=16))]
    )

    results = SemanticRetrieval(
        db_session, DeterministicEmbeddings({"OCC": q})
    ).retrieve("OCC", top_k=5)
    assert [r.molecule.standardized_smiles for r in results] == ["CCCO"]


def test_semantic_retrieval_returns_empty_when_no_matching_embeddings_exist(
    db_session: Session,
) -> None:
    q = _vec(1.0, 0.0, dim=16)
    retriever = SemanticRetrieval(db_session, DeterministicEmbeddings({"CCO": q}))
    assert retriever.retrieve("CCO", top_k=5) == []

    seed_molecule(db_session, "CCCO")
    seed_molecule(db_session, "CCN", embeddings=[("unimol", "unimolv2", q)])
    assert retriever.retrieve("CCO", top_k=5) == []


def test_semantic_retrieval_raises_data_error_on_mismatched_active_dimension(
    db_session: Session,
) -> None:
    seed_molecule(db_session, "CCCO", embeddings=[("unimol", "unimolv1", [0.1] * 4)])
    with pytest.raises(DataError):
        SemanticRetrieval(
            db_session, DeterministicEmbeddings({"CCO": _vec(1.0, 0.0, dim=16)})
        ).retrieve("CCO", top_k=5)
    db_session.rollback()


def test_hybrid_retrieval_matches_exact_rrf_ranking_and_weighted_score_formula(
    pg_session_factory: sessionmaker[Session], clean_db: None
) -> None:
    q = _vec(1.0, 0.0, dim=16)
    with pg_session_factory() as db:
        seed_molecule(db, "CCO", embeddings=[("unimol", "unimolv1", q)])
        seed_molecule(
            db,
            "CCCO",
            sources=[("ChEMBL", "CHEMBL14688")],
            embeddings=[("unimol", "unimolv1", _vec(0.9, 0.43588989, dim=16))],
        )
        seed_molecule(
            db,
            "c1ccccc1",
            sources=[("ChEMBL", "CHEMBL277500")],
            embeddings=[("unimol", "unimolv1", _vec(0.5, 0.8660254, dim=16))],
        )
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
    assert [r.score for r in top_three] == pytest.approx(
        [
            0.5 * _tanimoto("CCO", "CCCO") + 0.5 * 0.9,
            0.5 * _tanimoto("CCO", "c1ccccc1") + 0.5 * 0.5,
            0.5 * _tanimoto("CCO", "CCCCCCCCC"),
        ],
        abs=1e-5,
    )
    # Eager-loaded sources survive session close for both semantic-backed and fp-only hits
    assert top_three[0].molecule.molecule_sources[0].source_identifier == "CHEMBL14688"
    assert (
        top_three[2].molecule.molecule_sources[0].source_identifier == "CHEMBL_FP_ONLY"
    )
