from collections.abc import Callable
from functools import partial

import httpx
import pytest

from src.core.config import settings
from tests.fakes import FakeExternalServices

# `src/db/session.py` calls `create_engine(settings.database_url)` at import time.
# In CI (where no `.env` exists), `database_url` defaults to `""`, which crashes SQLAlchemy;
# use an in-memory SQLite URL so imports succeed without a running Postgres server.
if not settings.database_url:
    settings.database_url = "sqlite:///:memory:"


@pytest.fixture
def mock_http(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Callable[[httpx.Request], httpx.Response]], None]:
    """Patch `httpx.AsyncClient` so all outbound HTTP calls go to a `MockTransport` handler."""

    def _install(handler: Callable[[httpx.Request], httpx.Response]) -> None:
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)),
        )

    return _install


@pytest.fixture
def external_services(
    monkeypatch: pytest.MonkeyPatch,
    mock_http: Callable[[Callable[[httpx.Request], httpx.Response]], None],
) -> FakeExternalServices:
    """Pin upstream service URLs and route all HTTP traffic to `FakeExternalServices`."""
    monkeypatch.setattr(settings, "gnn_service_url", "http://gnn.test/gnn-xai")
    monkeypatch.setattr(
        settings, "llm_service_url", "http://llm.test/api/v1/chat/completions"
    )
    monkeypatch.setattr(
        settings, "chembl_api_url", "http://chembl.test/chembl/api/data"
    )
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-openrouter-key")

    services = FakeExternalServices()
    mock_http(services.handle)
    return services
