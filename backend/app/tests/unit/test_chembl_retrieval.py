from collections.abc import Callable
from typing import Any

import httpx
import pytest

from src.clients import ChEMBLClient
from src.rag.augmentation.chembl_retrieval import ChemblRetrieval
from src.rag.result_types.molecular_evidence import MolecularEvidence
from tests.unit.conftest import run_client

HttpMocker = Callable[[Callable[[httpx.Request], httpx.Response]], None]
BASE_URL = "http://chembl.test/api/data"


def _activity(document: str | None = "DOC1", **overrides: Any) -> dict[str, Any]:
    return {
        "molecule_chembl_id": "CHEMBL1",
        "target_chembl_id": "CHEMBL240",
        "target_pref_name": "KCNH2",
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "14.0",
        "standard_units": "nM",
        "pchembl_value": "7.85",
        "activity_comment": None,
        "assay_description": "hERG patch clamp",
        "document_chembl_id": document,
        **overrides,
    }


def _install_api(
    mock_http: HttpMocker,
    *,
    molecules: list[dict[str, Any]] | None = None,
    activities: list[dict[str, Any]] | None = None,
    documents: list[dict[str, Any]] | None = None,
    overrides: dict[str, httpx.Response] | None = None,
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []
    defaults = {
        "molecule.json": ("molecules", molecules or []),
        "activity.json": ("activities", activities or []),
        "document.json": ("documents", documents or []),
    }

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        endpoint = req.url.path.rsplit("/", 1)[-1]
        if overrides and endpoint in overrides:
            return overrides[endpoint]
        key, rows = defaults[endpoint]
        return httpx.Response(200, json={key: rows, "page_meta": {"next": None}})

    mock_http(handler)
    return requests


def _retrieve(query: str) -> list[MolecularEvidence]:
    return run_client(
        ChEMBLClient(BASE_URL),
        lambda c: ChemblRetrieval(c).retrieve_molecular_evidence(query),
    )


def test_chembl_id_query_skips_smiles_lookup(mock_http: HttpMocker) -> None:
    reqs = _install_api(mock_http)
    assert _retrieve("CHEMBL25") == []
    assert [r.url.path for r in reqs] == ["/api/data/activity.json"]
    assert reqs[0].url.params["molecule_chembl_id"] == "CHEMBL25"


@pytest.mark.parametrize("query", ["CCO", "CHEMBL", "CHEMBL25a", "chembl25"])
def test_non_chembl_id_query_resolves_via_smiles_lookup(
    mock_http: HttpMocker, query: str
) -> None:
    reqs = _install_api(mock_http, molecules=[{"molecule_chembl_id": "CHEMBL25"}])
    _retrieve(query)
    assert [r.url.path for r in reqs] == [
        "/api/data/molecule.json",
        "/api/data/activity.json",
    ]
    assert (
        reqs[0].url.params["molecule_structures__canonical_smiles__flexmatch"] == query
    )
    assert reqs[1].url.params["molecule_chembl_id"] == "CHEMBL25"


def test_unknown_smiles_returns_empty_without_fetching_activities(
    mock_http: HttpMocker,
) -> None:
    reqs = _install_api(mock_http)
    assert _retrieve("CCO") == []
    assert [r.url.path for r in reqs] == ["/api/data/molecule.json"]


def test_multi_activity_join_deduplicates_docs_and_maps_each_citation(
    mock_http: HttpMocker,
) -> None:
    reqs = _install_api(
        mock_http,
        activities=[
            _activity(
                "DOC_B",
                standard_relation=">",
                standard_value=None,
                pchembl_value=None,
                activity_comment="inactive",
            ),
            _activity(
                "DOC_A",
                standard_value="14.5",
                pchembl_value="7.84",
                activity_comment="Active",
            ),
            _activity("DOC_A", standard_value="50.0", pchembl_value="7.3"),
        ],
        documents=[
            {
                "document_chembl_id": "DOC_A",
                "title": "Paper A",
                "abstract": "Patch clamp study.",
                "year": 2001,
                "doi": "10.1021/jm001",
            },
            {
                "document_chembl_id": "DOC_B",
                "title": "Paper B",
                "abstract": "",
                "year": 1995,
                "doi": None,
            },
        ],
    )

    ev_b, ev_a1, ev_a2 = _retrieve("CHEMBL1")

    doc_req = next(r for r in reqs if r.url.path.endswith("/document.json"))
    doc_ids = doc_req.url.params["document_chembl_id__in"].split(",")
    assert len(doc_ids) == 2 and set(doc_ids) == {"DOC_A", "DOC_B"}

    assert ev_a1 == MolecularEvidence(
        provider="ChEMBL",
        molecule_identifier="CHEMBL1",
        target_identifier="CHEMBL240",
        target_name="KCNH2",
        measurement="IC50",
        relation="=",
        value=14.5,
        units="nM",
        log_potency=7.84,
        outcome="Active",
        assay_description="hERG patch clamp",
        document_title="Paper A",
        document_abstract="Patch clamp study.",
        document_year=2001,
        document_url="https://doi.org/10.1021/jm001",
    )
    assert (ev_b.value, ev_b.log_potency, ev_b.outcome, ev_b.document_abstract) == (
        None,
        None,
        "inactive",
        None,
    )
    assert (ev_b.document_title, ev_b.document_year, ev_b.document_url) == (
        "Paper B",
        1995,
        "https://www.ebi.ac.uk/chembl/explore/document/DOC_B",
    )
    assert (ev_a2.document_title, ev_a2.document_url) == (
        "Paper A",
        "https://doi.org/10.1021/jm001",
    )


@pytest.mark.parametrize(
    ("doi", "expected_url"),
    [
        ("10.1/abc", "https://doi.org/10.1/abc"),
        (None, "https://www.ebi.ac.uk/chembl/explore/document/CHEMBL_DOC1"),
        ("", "https://www.ebi.ac.uk/chembl/explore/document/CHEMBL_DOC1"),
    ],
)
def test_document_url_prefers_doi_then_falls_back_to_chembl_url(
    mock_http: HttpMocker, doi: str | None, expected_url: str
) -> None:
    _install_api(
        mock_http,
        activities=[_activity("CHEMBL_DOC1")],
        documents=[{"document_chembl_id": "CHEMBL_DOC1", "title": "T", "doi": doi}],
    )
    assert _retrieve("CHEMBL1")[0].document_url == expected_url


def test_null_or_missing_document_yields_none_citations(mock_http: HttpMocker) -> None:
    reqs = _install_api(mock_http, activities=[_activity(None)])
    [ev_null] = _retrieve("CHEMBL1")
    assert not any(r.url.path.endswith("/document.json") for r in reqs)
    assert (
        ev_null.document_title,
        ev_null.document_abstract,
        ev_null.document_year,
        ev_null.document_url,
    ) == (None, None, None, None)

    _install_api(mock_http, activities=[_activity("DOC_MISSING")], documents=[])
    [ev_missing] = _retrieve("CHEMBL1")
    assert (
        ev_missing.document_title,
        ev_missing.document_abstract,
        ev_missing.document_year,
        ev_missing.document_url,
    ) == (None, None, None, None)


def test_activities_survive_document_connection_error_but_propagate_other_errors(
    mock_http: HttpMocker,
) -> None:
    _install_api(
        mock_http,
        activities=[_activity("DOC1", standard_value="25.0")],
        overrides={"document.json": httpx.Response(503)},
    )
    [ev] = _retrieve("CHEMBL1")
    assert ev.value == pytest.approx(25.0) and ev.document_url is None

    _install_api(
        mock_http,
        activities=[_activity("DOC1")],
        overrides={"document.json": httpx.Response(200, json={"unexpected": []})},
    )
    with pytest.raises(KeyError):
        _retrieve("CHEMBL1")


@pytest.mark.parametrize(
    ("query", "endpoint"), [("CCO", "molecule.json"), ("CHEMBL1", "activity.json")]
)
def test_connection_error_in_molecule_or_activity_lookup_propagates(
    mock_http: HttpMocker, query: str, endpoint: str
) -> None:
    _install_api(mock_http, overrides={endpoint: httpx.Response(503)})
    with pytest.raises(ConnectionError, match="503"):
        _retrieve(query)
