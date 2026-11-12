import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, create_autospec

import pytest
from sqlalchemy.orm import Session

from src.clients import GNNClient, LLMClient
from src.model import PredictionResponse
from src.rag import Retriever
from src.rag.augmentation import ClientStrategyBase
from src.rag.config import RAG_TOP_K
from src.rag.result_types.retrieved_molecules import RetrievedMolecule
from src.rag.retrieval.retrieval_base import Retrieval
from src.service import predict_properties
from tests.fakes import LLM_SUMMARY, SAMPLE_EVIDENCE, corpus_mol, gnn_payload


@dataclass
class _ServiceDeps:
    retriever: Retriever
    retrieval: Retrieval
    source: ClientStrategyBase
    llm: LLMClient
    gnn: GNNClient
    db: Session


def _default_neighbours() -> list[RetrievedMolecule]:
    return [
        RetrievedMolecule(
            corpus_mol("O=C(O)c1ccccc1O", [("ChEMBL", "CHEMBL424"), ("Unknown", "X1")]),
            0.88,
        )
    ]


def _make_deps(
    *,
    neighbours: list[RetrievedMolecule] | None = None,
    gnn_exc: Exception | None = None,
    llm_exc: Exception | None = None,
    source_exc: Exception | None = None,
) -> _ServiceDeps:
    stored = _default_neighbours() if neighbours is None else neighbours

    retrieval = create_autospec(Retrieval, instance=True)
    retrieval.retrieve.side_effect = lambda _smiles, k: stored[:k]

    retriever = create_autospec(Retriever, instance=True)
    retriever.create_retriever.return_value = retrieval

    source = create_autospec(ClientStrategyBase, instance=True)
    source.provider = "ChEMBL"
    source.retrieve_molecular_evidence = AsyncMock(
        side_effect=source_exc, return_value=[SAMPLE_EVIDENCE]
    )

    gnn = create_autospec(GNNClient, instance=True)
    gnn.get_prediction_and_attribution.side_effect = gnn_exc or gnn_payload

    llm = create_autospec(LLMClient, instance=True)
    llm.get_summary.side_effect = llm_exc
    llm.get_summary.return_value = LLM_SUMMARY

    db = create_autospec(Session, instance=True)
    return _ServiceDeps(retriever, retrieval, source, llm, gnn, db)


def test_predict_properties_builds_response_and_passes_predictions_to_llm() -> None:
    deps = _make_deps()
    raw = asyncio.run(
        predict_properties(
            "OCC", deps.retriever, [deps.source], deps.llm, deps.gnn, deps.db
        )
    )
    res = PredictionResponse(**raw)

    deps.retriever.create_retriever.assert_called_once_with(deps.db)
    deps.retrieval.retrieve.assert_called_once_with("OCC", RAG_TOP_K)
    assert [
        call.args[0] for call in deps.source.retrieve_molecular_evidence.await_args_list
    ] == ["CCO", "CHEMBL424"]

    assert res.molecule.smiles == "CCO"
    assert len(res.similar_compounds) == 1
    assert res.similar_compounds[0].smiles == "O=C(O)c1ccccc1O"
    assert res.similar_compounds[0].similarity_score == pytest.approx(0.88)
    assert res.similar_compounds[0].sources == [
        "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL424"
    ]
    assert res.llm_summary == LLM_SUMMARY

    prompt_sent = deps.llm.get_summary.await_args.args[0]
    assert "hERG_Karim: positive" in prompt_sent
    assert "[1] O=C(O)c1ccccc1O" in prompt_sent
    assert "KCNH2 (ChEMBL): IC50 = 14 nM" in prompt_sent


def test_predict_properties_returns_empty_similar_compounds_when_corpus_empty() -> None:
    deps = _make_deps(neighbours=[])
    raw = asyncio.run(
        predict_properties(
            "CCO", deps.retriever, [deps.source], deps.llm, deps.gnn, deps.db
        )
    )
    res = PredictionResponse(**raw)

    assert res.similar_compounds == []
    assert res.llm_summary == LLM_SUMMARY


def test_predict_properties_rejects_invalid_smiles_before_calling_collaborators() -> (
    None
):
    deps = _make_deps()
    with pytest.raises(ValueError):
        asyncio.run(
            predict_properties(
                "not_a_smiles",
                deps.retriever,
                [deps.source],
                deps.llm,
                deps.gnn,
                deps.db,
            )
        )

    deps.retriever.create_retriever.assert_not_called()
    deps.gnn.get_prediction_and_attribution.assert_not_awaited()
    deps.llm.get_summary.assert_not_awaited()


@pytest.mark.parametrize("llm_err_cls", [ConnectionError, RuntimeError])
def test_predict_properties_returns_empty_summary_when_llm_fails(
    llm_err_cls: type[Exception],
) -> None:
    deps = _make_deps(llm_exc=llm_err_cls("LLM failure"))
    raw = asyncio.run(
        predict_properties(
            "CCO", deps.retriever, [deps.source], deps.llm, deps.gnn, deps.db
        )
    )
    assert PredictionResponse(**raw).llm_summary == ""


def test_predict_properties_still_summarizes_when_evidence_source_fails() -> None:
    deps = _make_deps(source_exc=ConnectionError("ChEMBL down"))
    raw = asyncio.run(
        predict_properties(
            "CCO", deps.retriever, [deps.source], deps.llm, deps.gnn, deps.db
        )
    )
    res = PredictionResponse(**raw)
    assert res.llm_summary == LLM_SUMMARY
    assert "ChEMBL could not be reached" in deps.llm.get_summary.await_args.args[0]


@pytest.mark.parametrize("gnn_err_cls", [ValueError, ConnectionError])
def test_predict_properties_propagates_gnn_errors_without_calling_llm(
    gnn_err_cls: type[Exception],
) -> None:
    deps = _make_deps(gnn_exc=gnn_err_cls("GNN failure"))
    with pytest.raises(gnn_err_cls):
        asyncio.run(
            predict_properties(
                "CCO", deps.retriever, [deps.source], deps.llm, deps.gnn, deps.db
            )
        )
    deps.llm.get_summary.assert_not_awaited()
