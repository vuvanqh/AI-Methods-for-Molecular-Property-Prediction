from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy.dialects import postgresql

from src.rag import Retriever
from src.rag.embeddings import unimol_embeddings as unimol_module
from src.rag.embeddings.base_embeddings import BaseEmbeddings
from src.rag.embeddings.unimol_embeddings import UniMolEmbeddings
from src.rag.result_types.retrieved_molecules import RetrievedMolecule
from src.rag.retrieval import (
    FingerprintRetrieval,
    HybridRetrieval,
    Retrieval,
    SemanticRetrieval,
    hybrid_retrieval,
)
from src.rag.retrieval_type import RetrievalType


def mol(mol_id: str, smiles: str = "CC") -> SimpleNamespace:
    return SimpleNamespace(id=mol_id, standardized_smiles=smiles)


def fake_db(rows: list[tuple]) -> MagicMock:
    db = MagicMock()
    db.execute.return_value.all.return_value = rows
    return db


class FakeRetrieval(Retrieval):
    def __init__(self, results: list[RetrievedMolecule]):
        self.results = results
        self.requested_k = None

    def retrieve(self, smiles: str, top_k: int) -> list[RetrievedMolecule]:
        self.requested_k = top_k
        return self.results[:top_k]


class FakeEmbeddings(BaseEmbeddings):
    name = "fake"
    version = "1"

    def load(self):
        pass

    def unload(self):
        pass

    def embedd(self, smiles: str) -> np.ndarray:
        return np.zeros(512, dtype=np.float32)


@pytest.fixture
def equal_weights(monkeypatch):
    monkeypatch.setattr(
        hybrid_retrieval,
        "retrievalSettings",
        SimpleNamespace(fingerprint_weight=0.5, semantic_weight=0.5),
    )


@pytest.fixture
def fake_unimol(monkeypatch):
    model = MagicMock()
    model.get_repr.return_value = [[0.0] * 512]
    monkeypatch.setattr(unimol_module, "UniMolRepr", MagicMock(return_value=model))
    monkeypatch.setattr(
        unimol_module,
        "unimolSettings",
        SimpleNamespace(
            model_name="unimolv1",
            remove_hydrogens=False,
            batch_size=32,
            idle_unload_seconds=0.01,
        ),
    )


@pytest.mark.usefixtures("equal_weights")
def test_hybrid_sums_scores_for_molecule_found_by_both():
    a, b, c = mol("a"), mol("b"), mol("c")
    fingerprint = FakeRetrieval([RetrievedMolecule(a, 0.9), RetrievedMolecule(b, 0.8)])
    semantic = FakeRetrieval([RetrievedMolecule(b, 0.7), RetrievedMolecule(c, 0.6)])

    results = HybridRetrieval(fingerprint, semantic).retrieve("CC", 3)

    assert [r.molecule.id for r in results] == ["b", "a", "c"]
    assert results[0].score == pytest.approx(0.5 / 62 + 0.5 / 61)


@pytest.mark.usefixtures("equal_weights")
def test_hybrid_fetches_three_times_top_k_and_returns_top_k():
    molecules = [RetrievedMolecule(mol(str(i)), 1.0) for i in range(10)]
    fingerprint = FakeRetrieval(molecules)
    semantic = FakeRetrieval(molecules)

    results = HybridRetrieval(fingerprint, semantic).retrieve("CC", 2)

    assert len(results) == 2
    assert (fingerprint.requested_k, semantic.requested_k) == (6, 6)


@pytest.mark.parametrize(
    "make_retriever",
    [
        lambda db: FingerprintRetrieval(db),
        lambda db: SemanticRetrieval(db, FakeEmbeddings()),
    ],
    ids=["fingerprint", "semantic"],
)
def test_retrievers_exclude_the_input_molecule(make_retriever):
    db = fake_db([(mol("b", "CCC"), 0.4)])

    results = make_retriever(db).retrieve("OCC", 5)

    compiled = db.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert "molecules.standardized_smiles !=" in str(compiled)
    assert "CCO" in [v for v in compiled.params.values() if isinstance(v, str)]
    assert results[0].score == 0.4


def test_create_retriever_fingerprint_needs_no_model():
    retriever = Retriever(RetrievalType.FINGERPRINT).create_retriever(MagicMock())

    assert isinstance(retriever, FingerprintRetrieval)


@pytest.mark.parametrize(
    "retrieval_type", [RetrievalType.SEMANTIC, RetrievalType.FUSION]
)
def test_retriever_requires_model(retrieval_type):
    with pytest.raises(ValueError):
        Retriever(retrieval_type)


@pytest.mark.parametrize(
    ("retrieval_type", "expected"),
    [
        (RetrievalType.SEMANTIC, SemanticRetrieval),
        (RetrievalType.FUSION, HybridRetrieval),
    ],
)
def test_create_retriever_returns_matching_strategy(retrieval_type, expected):
    retriever = Retriever(retrieval_type, FakeEmbeddings()).create_retriever(
        MagicMock()
    )

    assert isinstance(retriever, expected)


@pytest.mark.usefixtures("fake_unimol")
def test_embedd_returns_flat_vector_and_unloads_when_idle():
    embeddings = UniMolEmbeddings(device="cpu")

    vector = embeddings.embedd("CCO")
    assert vector.shape == (512,)

    embeddings._unload_timer.join()
    assert embeddings.model is None


@pytest.mark.usefixtures("fake_unimol")
def test_new_embedding_cancels_previous_unload_timer():
    embeddings = UniMolEmbeddings(device="cpu")

    embeddings.embedd("CCO")
    first_timer = embeddings._unload_timer
    embeddings.embedd("CCC")

    assert embeddings._unload_timer is not first_timer
    assert first_timer.finished.is_set()
    embeddings._unload_timer.join()
