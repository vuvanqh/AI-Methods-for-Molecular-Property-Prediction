from types import SimpleNamespace

import numpy as np
import pytest

from src.rag import RetrievalType, Retriever
from src.rag.embeddings import unimol_embeddings as unimol_module
from src.rag.embeddings.base_embeddings import BaseEmbeddings
from src.rag.embeddings.unimol_embeddings import UniMolEmbeddings
from src.rag.result_types.retrieved_molecules import RetrievedMolecule
from src.rag.retrieval import (
    FingerprintRetrieval,
    HybridRetrieval,
    SemanticRetrieval,
    hybrid_retrieval,
)
from tests.unit.conftest import FakeRetrieval, make_molecule


class _FakeEmbeddings(BaseEmbeddings):
    name, version = "fake-encoder", "v1"

    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.embedded_smiles: list[str] = []

    def load(self) -> None:
        self.model = object()

    def unload(self) -> None:
        self.model = None

    def embedd(self, smiles: str) -> np.ndarray:
        self.embedded_smiles.append(smiles)
        return np.array([0.25] * 8, dtype=np.float32)


class _SentinelDB:
    def execute(self, *_args, **_kwargs) -> None:
        raise AssertionError("DB must not be queried for invalid SMILES")


class _FakeUniMolRepr:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs

    def get_repr(self, batch: list[str]) -> list[list[float]]:
        return [[float(i) for i in range(512)] for _ in batch]


class _FakeTimer:
    def __init__(self, interval: float, callback) -> None:
        self.interval, self.callback = interval, callback
        self.daemon = self.started = self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback()


def _cands(*pairs: tuple[str, float]) -> FakeRetrieval:
    return FakeRetrieval([RetrievedMolecule(make_molecule(i), s) for i, s in pairs])


@pytest.fixture
def set_hybrid_weights(monkeypatch: pytest.MonkeyPatch):
    def _set(fp_weight: float = 0.5, sem_weight: float = 0.5) -> None:
        monkeypatch.setattr(
            hybrid_retrieval,
            "retrievalSettings",
            SimpleNamespace(fingerprint_weight=fp_weight, semantic_weight=sem_weight),
        )

    return _set


@pytest.fixture
def fake_unimol_env(monkeypatch: pytest.MonkeyPatch):
    models: list[_FakeUniMolRepr] = []
    timers: list[_FakeTimer] = []
    cuda = {"available": False}

    monkeypatch.setattr(
        unimol_module,
        "UniMolRepr",
        lambda **kw: models.append(_FakeUniMolRepr(**kw)) or models[-1],
    )
    monkeypatch.setattr(
        unimol_module.threading,
        "Timer",
        lambda iv, cb: timers.append(_FakeTimer(iv, cb)) or timers[-1],
    )
    monkeypatch.setattr(
        unimol_module.torch.cuda, "is_available", lambda: cuda["available"]
    )
    monkeypatch.setattr(
        unimol_module,
        "unimolSettings",
        SimpleNamespace(
            model_name="unimolv1",
            remove_hydrogens=False,
            batch_size=32,
            idle_unload_seconds=600.0,
        ),
    )
    return SimpleNamespace(models=models, timers=timers, cuda=cuda)


@pytest.mark.parametrize(
    ("retrieval_type", "with_model", "expected_cls"),
    [
        (RetrievalType.FINGERPRINT, False, FingerprintRetrieval),
        (RetrievalType.SEMANTIC, True, SemanticRetrieval),
        (RetrievalType.FUSION, True, HybridRetrieval),
    ],
)
def test_retriever_factory_creates_expected_strategy(
    retrieval_type: RetrievalType, with_model: bool, expected_cls: type
) -> None:
    model = _FakeEmbeddings() if with_model else None
    strategy = Retriever(retrieval_type, model).create_retriever(_SentinelDB())  # type: ignore[arg-type]
    assert isinstance(strategy, expected_cls)


@pytest.mark.parametrize(
    "retrieval_type", [RetrievalType.SEMANTIC, RetrievalType.FUSION]
)
def test_retriever_factory_requires_embedding_model(
    retrieval_type: RetrievalType,
) -> None:
    with pytest.raises(ValueError, match="Embedding model must be provided"):
        Retriever(retrieval_type, None)


def test_retrievers_reject_invalid_smiles_before_embedding_or_db() -> None:
    db, embeddings = _SentinelDB(), _FakeEmbeddings()
    with pytest.raises(ValueError):
        FingerprintRetrieval(db).retrieve("not_a_smiles", top_k=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SemanticRetrieval(db, embeddings).retrieve("not_a_smiles", top_k=5)  # type: ignore[arg-type]
    assert embeddings.embedded_smiles == []


def test_hybrid_retrieval_promotes_consensus_candidate_from_beyond_top_k(
    set_hybrid_weights,
) -> None:
    set_hybrid_weights(0.5, 0.5)
    # "shared" ranks #3 (outside top_k=2) in each sub-list; consensus promotes it to #1
    fp = _cands(("fp1", 0.95), ("fp2", 0.90), ("shared", 0.80))
    sem = _cands(("sem1", 0.95), ("sem2", 0.90), ("shared", 0.70))

    results = HybridRetrieval(fp, sem, rrf_k=60).retrieve("CC", top_k=2)

    assert len(results) == 2 and results[0].molecule.id == "shared"
    assert results[0].score == pytest.approx(0.5 * 0.80 + 0.5 * 0.70)
    assert HybridRetrieval(fp, sem, rrf_k=60).retrieve("CC", top_k=0) == []


def test_hybrid_retrieval_breaks_ties_by_id_and_handles_empty_lists(
    set_hybrid_weights,
) -> None:
    set_hybrid_weights(0.5, 0.5)
    tied = HybridRetrieval(_cands(("z", 0.9)), _cands(("a", 0.9)), rrf_k=60).retrieve(
        "CCO", top_k=2
    )
    assert [r.molecule.id for r in tied] == ["a", "z"]

    assert HybridRetrieval(_cands(), _cands()).retrieve("CCO", top_k=5) == []
    assert [
        r.molecule.id
        for r in HybridRetrieval(_cands(("a", 0.7), ("b", 0.6)), _cands()).retrieve(
            "CCO", 5
        )
    ] == ["a", "b"]
    assert [
        r.molecule.id
        for r in HybridRetrieval(_cands(), _cands(("c", 0.7), ("d", 0.6))).retrieve(
            "CCO", 5
        )
    ] == ["c", "d"]


@pytest.mark.parametrize(
    ("fp_w", "sem_w", "rrf_k", "expected_first"),
    [
        (1.0, 0.0, 60, "fp-top"),
        (0.8, 0.2, 1, "fp-top"),  # small rrf_k favors single-list #1
        (0.8, 0.2, 10, "shared"),  # larger rrf_k favors dual-list consensus #2
    ],
)
def test_hybrid_retrieval_weight_and_rrf_k_crossover(
    set_hybrid_weights, fp_w: float, sem_w: float, rrf_k: int, expected_first: str
) -> None:
    set_hybrid_weights(fp_w, sem_w)
    fp = _cands(("fp-top", 0.95), ("shared", 0.80))
    sem = _cands(("sem-top", 0.95), ("shared", 0.80))
    assert (
        HybridRetrieval(fp, sem, rrf_k=rrf_k).retrieve("CCO", top_k=3)[0].molecule.id
        == expected_first
    )


@pytest.mark.parametrize(
    ("device", "cuda_available", "expected_use_cuda"),
    [("cuda", True, True), ("cuda", False, False), ("cpu", True, False)],
)
def test_unimol_load_selects_cuda_only_when_requested_and_available(
    fake_unimol_env, device: str, cuda_available: bool, expected_use_cuda: bool
) -> None:
    fake_unimol_env.cuda["available"] = cuda_available
    model = UniMolEmbeddings(device=device).load()
    assert model is fake_unimol_env.models[0]
    assert model.init_kwargs["use_cuda"] is expected_use_cuda


def test_unimol_unload_noop_and_lazy_load_idle_reload_lifecycle(
    fake_unimol_env,
) -> None:
    embeddings = UniMolEmbeddings(device="cpu")
    embeddings.unload()
    assert embeddings.model is None and fake_unimol_env.models == []

    vec = embeddings.embedd("CCO")
    assert vec.shape == (512,) and vec.dtype == np.float32
    assert len(fake_unimol_env.models) == 1
    t1 = fake_unimol_env.timers[0]
    assert t1.daemon and t1.started and t1.interval == 600.0

    embeddings.embedd("CCC")
    assert len(fake_unimol_env.models) == 1 and t1.cancelled is True
    t2 = fake_unimol_env.timers[1]
    assert t2.started and not t2.cancelled

    t2.fire()
    assert embeddings.model is None

    embeddings.embedd("c1ccccc1")
    assert (
        len(fake_unimol_env.models) == 2
        and embeddings.model is fake_unimol_env.models[1]
    )
