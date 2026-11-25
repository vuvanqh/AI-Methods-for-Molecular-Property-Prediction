import asyncio
import json
from collections.abc import Callable
from contextlib import aclosing
from typing import Any

import httpx
import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.clients import GNNClient, LLMClient
from src.db.analysis_status import AnalysisStatus
from src.db.models import Analysis
from src.model import PredictionResponse
from src.rag import RetrievalType
from src.rag.config import RAG_TOP_K
from src.rag.result_types.retrieved_molecules import RetrievedMolecule
from src.service import predict_properties
from tests.fakes import LLM_SUMMARY, SAMPLE_EVIDENCE, corpus_mol, gnn_payload
from tests.unit.conftest import FakeEvidenceSource, FakeRetrieval

HttpMocker = Callable[[Callable[[httpx.Request], httpx.Response]], None]
GNN_URL, LLM_URL = "http://gnn.test/gnn-xai", "http://llm.test/api/v1/chat/completions"


class _StubRetriever:
    retrieval_type = RetrievalType.FUSION

    def __init__(self, retrieval: FakeRetrieval) -> None:
        self._retrieval = retrieval

    def create_retriever(self, _db: Any) -> FakeRetrieval:
        return self._retrieval


class _RecordingSession:
    def __init__(self, *, commit_error: Exception | None = None) -> None:
        self.events: list[str] = []
        self.added: list[Analysis] = []
        self.commit_error = commit_error

    def rollback(self) -> None:
        self.events.append("rollback")

    def add(self, instance: Analysis) -> None:
        self.events.append("add")
        self.added.append(instance)

    def commit(self) -> None:
        self.events.append("commit")
        if self.commit_error is not None:
            raise self.commit_error


def _install_http(
    mock_http: HttpMocker,
    *,
    gnn_handler: Callable[[httpx.Request], httpx.Response] | None = None,
    llm_handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> dict[str, list[Any]]:
    seen: dict[str, list[Any]] = {"gnn_smiles": [], "llm_prompts": []}

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if req.url.host == "gnn.test":
            seen["gnn_smiles"].append(body["smiles"])
            return (
                gnn_handler(req)
                if gnn_handler
                else httpx.Response(200, json=gnn_payload(body["smiles"]))
            )
        seen["llm_prompts"].append(body["messages"][-1]["content"])
        if llm_handler:
            return llm_handler(req)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": LLM_SUMMARY}}]}
        )

    mock_http(handler)
    return seen


def _neighbours() -> list[RetrievedMolecule]:
    return [
        RetrievedMolecule(
            corpus_mol("O=C(O)c1ccccc1O", [("ChEMBL", "CHEMBL424"), ("Unknown", "X1")]),
            0.88,
        )
    ]


def _run_predict(
    smiles: str,
    *,
    retrieval: FakeRetrieval | None = None,
    source: FakeEvidenceSource | None = None,
    db: _RecordingSession | None = None,
) -> tuple[dict[str, Any], _RecordingSession, FakeEvidenceSource, FakeRetrieval]:
    ret = retrieval or FakeRetrieval(_neighbours())
    src = source or FakeEvidenceSource(
        "ChEMBL",
        evidence_by_query={"CCCN": [SAMPLE_EVIDENCE], "CHEMBL424": [SAMPLE_EVIDENCE]},
    )
    session = db or _RecordingSession()

    async def _execute() -> dict[str, Any]:
        async with (
            aclosing(GNNClient(GNN_URL)) as gnn,
            aclosing(LLMClient(LLM_URL)) as llm,
        ):
            return await predict_properties(
                smiles,
                _StubRetriever(ret),
                [src],
                llm,
                gnn,
                session,  # type: ignore[arg-type]
            )

    return asyncio.run(_execute()), session, src, ret


def test_predict_properties_orchestrates_pipeline_and_preserves_raw_vs_canonical(
    mock_http: HttpMocker,
) -> None:
    def _gnn_atom0_highest(req: httpx.Request) -> httpx.Response:
        payload = gnn_payload(json.loads(req.content)["smiles"])
        payload["atomscores"]["hERG_Karim"]["0"] = 0.95
        return httpx.Response(200, json=payload)

    seen = _install_http(mock_http, gnn_handler=_gnn_atom0_highest)
    raw, db, source, retrieval = _run_predict("NCCC")
    res = PredictionResponse(**raw)

    assert retrieval.calls == [("NCCC", RAG_TOP_K)] and seen["gnn_smiles"] == ["NCCC"]
    assert source.queries == ["CCCN", "CHEMBL424"]
    assert res.molecule.smiles == "CCCN" and res.molecule.formula == "C3H9N"
    assert res.similar_compounds[0].sources == [
        "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL424"
    ]
    assert res.llm_summary == LLM_SUMMARY
    # Prompt indexes atoms against raw "NCCC" (nitrogen at N0, not canonical N3)
    assert (
        "N0" in seen["llm_prompts"][0]
        and "[1] O=C(O)c1ccccc1O" in seen["llm_prompts"][0]
    )
    assert db.events == ["add", "commit"]
    assert db.added[0].status == AnalysisStatus.COMPLETED and db.added[0].results == raw
    assert db.added[0].config == {
        "input_smiles": "NCCC",
        "retrieval_type": RetrievalType.FUSION.value,
        "top_k": RAG_TOP_K,
    }


def test_predict_properties_logs_standardized_smiles_using_cleanup(
    mock_http: HttpMocker,
) -> None:
    _install_http(mock_http)
    _, db, _, _ = _run_predict("CC(=O)O[Na]")
    # describe_molecule keeps covalent "CC(=O)O[Na]"; standardize_smiles disconnects salt
    assert db.added[0].standardized_input_smiles == "CC(=O)[O-].[Na+]"


def test_predict_properties_rejects_invalid_smiles_before_collaborators(
    mock_http: HttpMocker,
) -> None:
    seen = _install_http(mock_http)
    db, source, retrieval = (
        _RecordingSession(),
        FakeEvidenceSource("ChEMBL"),
        FakeRetrieval(_neighbours()),
    )

    with pytest.raises(ValueError):
        _run_predict("not_a_smiles", retrieval=retrieval, source=source, db=db)

    assert retrieval.calls == [] and source.queries == [] and db.events == []
    assert seen == {"gnn_smiles": [], "llm_prompts": []}


@pytest.mark.parametrize(
    "llm_response", [httpx.Response(503), httpx.Response(200, json={"choices": []})]
)
def test_predict_properties_degrades_to_empty_summary_on_llm_failure(
    mock_http: HttpMocker, llm_response: httpx.Response
) -> None:
    _install_http(mock_http, llm_handler=lambda _req: llm_response)
    raw, db, _, _ = _run_predict("CCO")
    assert PredictionResponse(**raw).llm_summary == ""
    assert db.added[0].status == AnalysisStatus.COMPLETED


def test_predict_properties_degrades_when_evidence_source_unreachable(
    mock_http: HttpMocker,
) -> None:
    seen = _install_http(mock_http)
    raw, db, _, _ = _run_predict(
        "CCO", source=FakeEvidenceSource("ChEMBL", error=ConnectionError("down"))
    )
    assert PredictionResponse(**raw).llm_summary == LLM_SUMMARY
    assert "ChEMBL could not be reached" in seen["llm_prompts"][0]
    assert db.added[0].status == AnalysisStatus.COMPLETED


def test_predict_properties_propagates_gnn_422_without_logging_or_llm(
    mock_http: HttpMocker,
) -> None:
    seen = _install_http(mock_http, gnn_handler=lambda _req: httpx.Response(422))
    db = _RecordingSession()
    with pytest.raises(ValueError):
        _run_predict("[Na+]", db=db)
    assert seen["llm_prompts"] == [] and db.events == []


@pytest.mark.parametrize(
    ("failure_mode", "expected_exc"),
    [
        ("gnn_503", ConnectionError),
        ("retrieval_error", RuntimeError),
        ("evidence_bug", RuntimeError),
        ("llm_unexpected", TypeError),
    ],
)
def test_predict_properties_logs_failed_and_propagates_downstream_errors(
    mock_http: HttpMocker, failure_mode: str, expected_exc: type[Exception]
) -> None:
    def _boom(_req: httpx.Request) -> httpx.Response:
        raise TypeError("unexpected llm bug")

    _install_http(
        mock_http,
        gnn_handler=(lambda _req: httpx.Response(503))
        if failure_mode == "gnn_503"
        else None,
        llm_handler=_boom if failure_mode == "llm_unexpected" else None,
    )
    retrieval = FakeRetrieval(_neighbours())
    if failure_mode == "retrieval_error":
        retrieval.retrieve = lambda _s, _k: (_ for _ in ()).throw(
            RuntimeError("pgvector error")
        )  # type: ignore[method-assign]
    source = FakeEvidenceSource(
        "ChEMBL",
        error=RuntimeError("parse bug") if failure_mode == "evidence_bug" else None,
    )
    db = _RecordingSession()

    with pytest.raises(expected_exc):
        _run_predict("OCC", retrieval=retrieval, source=source, db=db)

    assert db.events == ["rollback", "add", "commit"]
    assert db.added[0].standardized_input_smiles == "CCO"
    assert db.added[0].status == AnalysisStatus.FAILED and db.added[0].results is None


def test_predict_properties_propagates_original_error_when_failed_logging_hits_db_error(
    mock_http: HttpMocker,
) -> None:
    _install_http(mock_http, gnn_handler=lambda _req: httpx.Response(503))
    db = _RecordingSession(commit_error=SQLAlchemyError("db down"))

    with pytest.raises(ConnectionError, match="503"):
        _run_predict("CCO", db=db)

    assert db.events[-1] == "rollback"
