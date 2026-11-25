import asyncio
import json
from collections.abc import Callable
from contextlib import aclosing

import httpx
import pytest

from src.clients import ChEMBLClient, GNNClient, LLMClient
from src.core.config import settings
from tests.fakes import LLM_SUMMARY, gnn_payload
from tests.unit.conftest import run_client

HttpMocker = Callable[[Callable[[httpx.Request], httpx.Response]], None]

REQUIRED_ACTIVITY_FIELDS = {
    "molecule_chembl_id",
    "target_chembl_id",
    "target_pref_name",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "activity_comment",
    "assay_description",
    "document_chembl_id",
}
REQUIRED_DOCUMENT_FIELDS = {"document_chembl_id", "title", "abstract", "year", "doi"}


def _page(key: str, rows: list[dict], next_url: str | None = None) -> httpx.Response:
    return httpx.Response(200, json={key: rows, "page_meta": {"next": next_url}})


def test_clients_resolve_default_base_url_lazily_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gnn_service_url", "http://runtime-gnn.test/")
    monkeypatch.setattr(
        settings, "llm_service_url", "http://runtime-llm.test/v1/chat/completions"
    )
    monkeypatch.setattr(
        settings, "chembl_api_url", "http://runtime-chembl.test/api/data"
    )

    cases = [
        (GNNClient, None, "http://runtime-gnn.test/prediction_and_attribution"),
        (LLMClient, None, "http://runtime-llm.test/v1/chat/completions"),
        (ChEMBLClient, None, "http://runtime-chembl.test/api/data/"),
        (
            GNNClient,
            "http://explicit-gnn.test/",
            "http://explicit-gnn.test/prediction_and_attribution",
        ),
        (
            LLMClient,
            "http://explicit-llm.test/completions",
            "http://explicit-llm.test/completions",
        ),
        (
            ChEMBLClient,
            "http://explicit-chembl.test/data",
            "http://explicit-chembl.test/data/",
        ),
    ]

    async def _check() -> None:
        for client_cls, base_url, expected_url in cases:
            client = client_cls() if base_url is None else client_cls(base_url=base_url)
            async with aclosing(client):
                assert str(client.url) == expected_url

    asyncio.run(_check())


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("http://gnn", "http://gnn/prediction_and_attribution"),
        ("http://gnn/gnn-xai/", "http://gnn/gnn-xai/prediction_and_attribution"),
    ],
)
def test_gnn_client_posts_smiles_to_normalized_url(
    mock_http: HttpMocker, base_url: str, expected_url: str
) -> None:
    captured: list[httpx.Request] = []
    expected_payload = gnn_payload("CCO")

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json=expected_payload)

    mock_http(handler)
    res = run_client(
        GNNClient(base_url), lambda c: c.get_prediction_and_attribution("CCO")
    )

    assert captured[0].method == "POST"
    assert str(captured[0].url) == expected_url
    assert json.loads(captured[0].content) == {"smiles": "CCO"}
    assert res == expected_payload


def test_gnn_client_maps_422_to_value_error(mock_http: HttpMocker) -> None:
    mock_http(lambda _req: httpx.Response(422, json={"message": "unbonded"}))
    with pytest.raises(ValueError, match="could not be processed"):
        run_client(
            GNNClient("http://gnn"), lambda c: c.get_prediction_and_attribution("[Na+]")
        )


def test_llm_client_sends_auth_config_and_prompt(
    monkeypatch: pytest.MonkeyPatch, mock_http: HttpMocker
) -> None:
    captured_body: dict = {}
    captured_headers: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        captured_headers.update(req.headers)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": LLM_SUMMARY}}]}
        )

    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "secret-token")
    mock_http(handler)
    summary = run_client(
        LLMClient("http://llm/completions"), lambda c: c.get_summary("Explain ethanol")
    )

    assert summary == LLM_SUMMARY
    assert captured_headers["authorization"] == "Bearer secret-token"
    assert captured_body["model"] == settings.llm_model
    assert captured_body["max_tokens"] == settings.llm_max_tokens
    assert captured_body["temperature"] == settings.llm_temp
    msgs = captured_body["messages"]
    assert any(m["role"] == "system" and m["content"] for m in msgs)
    assert any(m["role"] == "user" and "Explain ethanol" in m["content"] for m in msgs)


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b"{}",
        b'{"choices": []}',
        b'{"choices": [{"message": {"content": ""}}]}',
    ],
)
def test_llm_client_rejects_malformed_or_empty_response(
    mock_http: HttpMocker, body: bytes
) -> None:
    mock_http(lambda _req: httpx.Response(200, content=body))
    with pytest.raises(RuntimeError):
        run_client(
            LLMClient("http://llm/completions"), lambda c: c.get_summary("Prompt")
        )


def test_chembl_client_find_molecule_id_returns_first_match_or_none(
    mock_http: HttpMocker,
) -> None:
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        smiles = req.url.params["molecule_structures__canonical_smiles__flexmatch"]
        rows = (
            [{"molecule_chembl_id": "CHEMBL25"}, {"molecule_chembl_id": "CHEMBL999"}]
            if smiles == "CCO"
            else []
        )
        return _page("molecules", rows)

    mock_http(handler)
    found = run_client(
        ChEMBLClient("http://chembl/api/data"), lambda c: c.find_molecule_id("CCO")
    )
    missing = run_client(
        ChEMBLClient("http://chembl/api/data/"), lambda c: c.find_molecule_id("CCC")
    )

    assert found == "CHEMBL25" and missing is None
    assert captured[0].url.path == "/api/data/molecule.json"
    assert captured[0].url.params["only"] == "molecule_chembl_id"
    assert int(captured[0].url.params["limit"]) <= 1000


def test_chembl_client_get_activities_applies_filters_and_required_fields(
    mock_http: HttpMocker,
) -> None:
    params: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        params.update(req.url.params)
        return _page("activities", [{"molecule_chembl_id": "CHEMBL25"}])

    mock_http(handler)
    rows = run_client(
        ChEMBLClient("http://chembl/api/data"), lambda c: c.get_activities("CHEMBL25")
    )

    assert rows == [{"molecule_chembl_id": "CHEMBL25"}]
    assert params["molecule_chembl_id"] == "CHEMBL25"
    assert params["target_organism"] == "Homo sapiens"
    assert params["data_validity_comment__isnull"] == "true"
    assert params["potential_duplicate"] == "0"
    assert int(params["limit"]) <= 1000
    assert REQUIRED_ACTIVITY_FIELDS <= set(params["only"].split(","))


def test_chembl_client_paginates_and_propagates_next_page_error(
    mock_http: HttpMocker,
) -> None:
    seen: list[dict[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(dict(req.url.params))
        if req.url.params.get("offset") == "1":
            return _page("activities", [{"id": 2}])
        return _page("activities", [{"id": 1}], "/api/data/activity.json?offset=1")

    mock_http(handler)
    rows = run_client(
        ChEMBLClient("http://chembl/api/data"), lambda c: c.get_activities("CHEMBL1")
    )
    assert rows == [{"id": 1}, {"id": 2}] and seen[1]["offset"] == "1"

    mock_http(
        lambda req: (
            httpx.Response(503)
            if req.url.params.get("offset") == "1"
            else _page("activities", [{"id": 1}], "/api/data/activity.json?offset=1")
        )
    )
    with pytest.raises(ConnectionError, match="503"):
        run_client(
            ChEMBLClient("http://chembl/api/data"),
            lambda c: c.get_activities("CHEMBL1"),
        )


@pytest.mark.parametrize(
    ("num_ids", "expected_requests"), [(0, 0), (1, 1), (50, 1), (51, 2), (120, 3)]
)
def test_chembl_client_get_documents_batches_within_limit(
    mock_http: HttpMocker, num_ids: int, expected_requests: int
) -> None:
    batches: list[int] = []
    seen_only: list[set[str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        ids = req.url.params["document_chembl_id__in"].split(",")
        batches.append(len(ids))
        seen_only.append(set(req.url.params["only"].split(",")))
        return _page("documents", [{"document_chembl_id": d} for d in ids])

    mock_http(handler)
    docs = run_client(
        ChEMBLClient("http://chembl/api/data"),
        lambda c: c.get_documents([f"DOC{i}" for i in range(num_ids)]),
    )

    assert len(batches) == expected_requests
    assert all(1 <= size <= 50 for size in batches)
    assert len(docs) == num_ids
    assert all(REQUIRED_DOCUMENT_FIELDS <= fields for fields in seen_only)


def test_chembl_client_get_documents_propagates_second_batch_failure(
    mock_http: HttpMocker,
) -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            return httpx.Response(503)
        ids = req.url.params["document_chembl_id__in"].split(",")
        return _page("documents", [{"document_chembl_id": d} for d in ids])

    mock_http(handler)
    with pytest.raises(ConnectionError, match="503"):
        run_client(
            ChEMBLClient("http://chembl/api/data"),
            lambda c: c.get_documents([f"DOC{i}" for i in range(51)]),
        )


@pytest.mark.parametrize("client_name", ["gnn", "llm", "chembl"])
@pytest.mark.parametrize("status_or_exc", [400, 503, httpx.ReadTimeout])
def test_clients_map_http_and_timeout_errors_to_connection_error(
    mock_http: HttpMocker,
    client_name: str,
    status_or_exc: int | type[httpx.RequestError],
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        if isinstance(status_or_exc, int):
            return httpx.Response(status_or_exc, json={"message": "fail"})
        raise status_or_exc("timeout")

    mock_http(handler)
    actions = {
        "gnn": lambda: run_client(
            GNNClient("http://gnn"), lambda c: c.get_prediction_and_attribution("CCO")
        ),
        "llm": lambda: run_client(
            LLMClient("http://llm/completions"), lambda c: c.get_summary("Prompt")
        ),
        "chembl": lambda: run_client(
            ChEMBLClient("http://chembl/api/data"), lambda c: c.find_molecule_id("CCO")
        ),
    }
    with pytest.raises(ConnectionError):
        actions[client_name]()
