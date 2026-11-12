import asyncio
import json
from collections.abc import Callable
from contextlib import aclosing

import httpx
import pytest

from src.clients import ChEMBLClient, GNNClient, LLMClient
from src.core.config import settings
from src.model import GNNXAIInferenceResponse
from tests.fakes import LLM_SUMMARY, gnn_payload

HttpMocker = Callable[[Callable[[httpx.Request], httpx.Response]], None]


def run_client(client, action):
    """Run an async `action(client)` inside an event loop and close the client afterwards."""

    async def _execute():
        async with aclosing(client):
            return await action(client)

    return asyncio.run(_execute())


def test_client_aclose_closes_underlying_http_pool(mock_http: HttpMocker) -> None:
    mock_http(lambda _req: httpx.Response(200, json={}))
    client = GNNClient("http://gnn")
    assert not client.http.is_closed
    asyncio.run(client.aclose())
    assert client.http.is_closed


def test_gnn_client_posts_smiles_and_returns_parsed_payload(
    mock_http: HttpMocker,
) -> None:
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json=gnn_payload("CCO"))

    mock_http(handler)
    res = run_client(
        GNNClient("http://gnn/"),
        lambda c: c.get_prediction_and_attribution("CCO"),
    )

    assert captured[0].method == "POST"
    assert str(captured[0].url) == "http://gnn/prediction_and_attribution"
    assert json.loads(captured[0].content) == {"smiles": "CCO"}
    assert "V2000" in GNNXAIInferenceResponse(**res).molblock


def test_gnn_client_maps_422_to_value_error(mock_http: HttpMocker) -> None:
    mock_http(lambda _req: httpx.Response(422, json={"message": "bad"}))
    with pytest.raises(ValueError):
        run_client(
            GNNClient("http://gnn"),
            lambda c: c.get_prediction_and_attribution("[Na+]"),
        )


def test_llm_client_sends_prompt_and_returns_summary(mock_http: HttpMocker) -> None:
    captured_body: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": LLM_SUMMARY}}]}
        )

    mock_http(handler)
    summary = run_client(
        LLMClient("http://llm/completions"),
        lambda c: c.get_summary("Explain ethanol"),
    )

    assert summary == LLM_SUMMARY
    assert captured_body["model"] == settings.llm_model
    assert captured_body["max_tokens"] == settings.llm_max_tokens
    assert captured_body["temperature"] == settings.llm_temp
    assert captured_body["messages"][0]["role"] == "system"
    assert captured_body["messages"][1] == {
        "role": "user",
        "content": "Explain ethanol",
    }


def test_llm_client_sends_bearer_authorization_header(
    monkeypatch: pytest.MonkeyPatch, mock_http: HttpMocker
) -> None:
    captured_headers: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_headers.update(req.headers)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": LLM_SUMMARY}}]}
        )

    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "secret-token")
    mock_http(handler)
    run_client(LLMClient("http://llm/completions"), lambda c: c.get_summary("Prompt"))

    assert captured_headers["authorization"] == "Bearer secret-token"


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"{}",
        b'{"choices": []}',
        b'{"choices": [{"message": {"content": ""}}]}',
        b'{"choices": [{"message": {"content": null}}]}',
    ],
)
def test_llm_client_raises_runtime_error_on_malformed_or_empty_response(
    mock_http: HttpMocker, body: bytes
) -> None:
    mock_http(lambda _req: httpx.Response(200, content=body))
    with pytest.raises(RuntimeError):
        run_client(
            LLMClient("http://llm/completions"), lambda c: c.get_summary("Prompt")
        )


def test_chembl_client_resolves_molecule_id_or_returns_none(
    mock_http: HttpMocker,
) -> None:
    captured_paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_paths.append(req.url.path)
        smiles = req.url.params["molecule_structures__canonical_smiles__flexmatch"]
        mols = [{"molecule_chembl_id": "CHEMBL25"}] if smiles == "CCO" else []
        return httpx.Response(
            200, json={"molecules": mols, "page_meta": {"next": None}}
        )

    mock_http(handler)

    async def _query_both(c: ChEMBLClient) -> tuple[str | None, str | None]:
        return await c.find_molecule_id("CCO"), await c.find_molecule_id("CCC")

    found, missing = run_client(ChEMBLClient("http://chembl/api/data"), _query_both)
    assert (found, missing) == ("CHEMBL25", None)
    assert captured_paths == ["/api/data/molecule.json", "/api/data/molecule.json"]


def test_chembl_client_filters_activities_to_valid_human_measurements(
    mock_http: HttpMocker,
) -> None:
    captured_params: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_params.update(req.url.params)
        return httpx.Response(
            200,
            json={
                "activities": [{"molecule_chembl_id": "CHEMBL25"}],
                "page_meta": {"next": None},
            },
        )

    mock_http(handler)
    rows = run_client(
        ChEMBLClient("http://chembl/api/data"),
        lambda c: c.get_activities("CHEMBL25"),
    )

    assert rows == [{"molecule_chembl_id": "CHEMBL25"}]
    assert captured_params["molecule_chembl_id"] == "CHEMBL25"
    assert captured_params["target_organism"] == "Homo sapiens"
    assert captured_params["data_validity_comment__isnull"] == "true"
    assert captured_params["potential_duplicate"] == "0"


def test_chembl_client_skips_http_call_when_document_ids_empty(
    mock_http: HttpMocker,
) -> None:
    requests_made = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal requests_made
        requests_made += 1
        return httpx.Response(200, json={})

    mock_http(handler)
    docs = run_client(
        ChEMBLClient("http://chembl/api/data"), lambda c: c.get_documents([])
    )
    assert docs == []
    assert requests_made == 0


def _call_upstream_client(client_name: str) -> object:
    if client_name == "gnn":
        return run_client(
            GNNClient("http://gnn"), lambda c: c.get_prediction_and_attribution("CCO")
        )
    if client_name == "llm":
        return run_client(
            LLMClient("http://llm/completions"), lambda c: c.get_summary("Prompt")
        )
    return run_client(
        ChEMBLClient("http://chembl/api/data"), lambda c: c.find_molecule_id("CCO")
    )


@pytest.mark.parametrize(
    ("client_name", "status_or_exc"),
    [
        ("gnn", 503),
        ("gnn", 404),
        ("gnn", httpx.ConnectError),
        ("gnn", httpx.ReadTimeout),
        ("llm", 503),
        ("llm", 401),
        ("llm", 429),
        ("llm", httpx.ConnectError),
        ("llm", httpx.ReadTimeout),
        ("chembl", 503),
        ("chembl", 404),
        ("chembl", httpx.ConnectError),
        ("chembl", httpx.ReadTimeout),
    ],
)
def test_clients_map_upstream_errors_and_network_failures_to_connection_error(
    mock_http: HttpMocker,
    client_name: str,
    status_or_exc: int | type[httpx.RequestError],
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        if isinstance(status_or_exc, int):
            return httpx.Response(status_or_exc, json={"message": "upstream failure"})
        raise status_or_exc("network failure")

    mock_http(handler)
    with pytest.raises(ConnectionError):
        _call_upstream_client(client_name)
