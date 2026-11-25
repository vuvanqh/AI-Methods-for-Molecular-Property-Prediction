import asyncio
import json
import threading
from typing import Any

import httpx
import torch
from fastapi import FastAPI

from src.config import (
    BINARY_PREDICTION_THRESHOLDS,
    DEVICE,
    IG_STEPS,
    TASKS,
    TASKS_NUMBER,
)
from src.controller import gnn_router
from src.gnn_models.chemprop import Chemprop
from src.main import lifespan
from src.model import ErrorResponse, GNNXAIInferenceResponse
from src.service import MolecularInferenceService
from src.xai.integrated_gradients import IntegratedGradientsExplainer
from tests.integration.conftest import create_controller_app

ENDPOINT = "/gnn-xai/prediction_and_attribution"


class _BarrierControlledService:
    """Deterministic barrier wrapper delegating every call to a real `MolecularInferenceService`."""

    def __init__(self, delegate: MolecularInferenceService) -> None:
        self._delegate = delegate
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self.entered_smiles: list[str] = []
        self.active_count = 0
        self.max_active_observed = 0
        self._guard = threading.Lock()

    def predict_and_explain(self, smiles: str) -> dict[str, Any]:
        with self._guard:
            call_index = len(self.entered_smiles)
            self.entered_smiles.append(smiles)
            self.active_count += 1
            self.max_active_observed = max(self.max_active_observed, self.active_count)

        try:
            if call_index == 0:
                self.first_entered.set()
                assert self.release_first.wait(timeout=10.0), (
                    "Timed out waiting for release_first barrier"
                )
            return self._delegate.predict_and_explain(smiles)
        finally:
            with self._guard:
                self.active_count -= 1


class _ArrivalTrackerMiddleware:
    """Pure ASGI middleware signalling request arrivals without wrapping `receive` streams."""

    def __init__(self, app: Any, events: list[asyncio.Event]) -> None:
        self._app = app
        self._events = events
        self._count = 0

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            if self._count < len(self._events):
                self._events[self._count].set()
            self._count += 1
        await self._app(scope, receive, send)


def _setup_tracked_barrier_app(
    service: MolecularInferenceService, num_requests: int
) -> tuple[
    _BarrierControlledService, FastAPI, _ArrivalTrackerMiddleware, list[asyncio.Event]
]:
    barrier = _BarrierControlledService(delegate=service)
    app = create_controller_app(barrier)
    arrivals = [asyncio.Event() for _ in range(num_requests)]
    return barrier, app, _ArrivalTrackerMiddleware(app, arrivals), arrivals


async def _run_overlapping_pair(
    service: MolecularInferenceService, first_smiles: str, second_smiles: str
) -> tuple[httpx.Response, httpx.Response, _BarrierControlledService, FastAPI]:
    barrier, app, tracked_app, arrivals = _setup_tracked_barrier_app(service, 2)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=tracked_app), base_url="http://testserver"
    ) as client:
        first_task = asyncio.create_task(
            client.post(ENDPOINT, json={"smiles": first_smiles})
        )
        assert await asyncio.to_thread(barrier.first_entered.wait, 10.0)
        assert app.state.inference_lock.locked() is True

        second_task = asyncio.create_task(
            client.post(ENDPOINT, json={"smiles": second_smiles})
        )
        await asyncio.wait_for(arrivals[1].wait(), timeout=5.0)
        assert barrier.entered_smiles == [first_smiles]
        assert barrier.max_active_observed == 1

        barrier.release_first.set()
        resp_first, resp_second = await asyncio.gather(first_task, second_task)
    return resp_first, resp_second, barrier, app


def test_controller_serializes_overlapping_real_inferences_without_cross_request_leakage(
    real_inference_service: MolecularInferenceService,
) -> None:
    """Overlapping requests serialize strictly through `inference_lock` and return their own molecule's data."""
    resp_eth, resp_benz, barrier, app = asyncio.run(
        _run_overlapping_pair(real_inference_service, "CCO", "c1ccccc1")
    )

    assert resp_eth.status_code == 200 and resp_benz.status_code == 200
    assert barrier.entered_smiles == ["CCO", "c1ccccc1"]
    assert barrier.max_active_observed == 1
    assert app.state.inference_lock.locked() is False

    eth_payload = GNNXAIInferenceResponse(**resp_eth.json())
    benz_payload = GNNXAIInferenceResponse(**resp_benz.json())
    assert len(eth_payload.atomscores[TASKS[0]]) == 9
    assert len(benz_payload.atomscores[TASKS[0]]) == 12
    assert eth_payload.molblock != benz_payload.molblock


def test_controller_releases_lock_after_real_validation_error_so_queued_request_succeeds(
    real_inference_service: MolecularInferenceService,
) -> None:
    """A real `MoleculeValidationError` (422) releases `inference_lock` so a queued valid request returns 200."""
    resp_bad, resp_ok, barrier, app = asyncio.run(
        _run_overlapping_pair(real_inference_service, "[Na+]", "c1ccccc1")
    )

    assert resp_bad.status_code == 422
    assert ErrorResponse(**resp_bad.json()).message == "Invalid molecular input."
    assert resp_ok.status_code == 200
    assert len(GNNXAIInferenceResponse(**resp_ok.json()).atomscores[TASKS[0]]) == 12
    assert barrier.entered_smiles == ["[Na+]", "c1ccccc1"]
    assert barrier.max_active_observed == 1
    assert app.state.inference_lock.locked() is False


def test_controller_aborts_queued_disconnected_client_with_499_and_runs_next_real_request(
    real_inference_service: MolecularInferenceService,
) -> None:
    """A client disconnecting behind an active real inference gets 499 without running inference, and Request 3 succeeds."""

    async def _run() -> None:
        barrier, app, tracked_app, arrivals = _setup_tracked_barrier_app(
            real_inference_service, 3
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=tracked_app), base_url="http://testserver"
        ) as client:
            task_inflight = asyncio.create_task(
                client.post(ENDPOINT, json={"smiles": "CC"})
            )
            assert await asyncio.to_thread(barrier.first_entered.wait, 10.0)
            assert app.state.inference_lock.locked() is True

            body_bytes = json.dumps({"smiles": "CCO"}).encode("utf-8")
            body_sent = False
            disconnect_ready = asyncio.Event()
            sent_messages: list[dict[str, Any]] = []

            async def _receive() -> dict[str, Any]:
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {
                        "type": "http.request",
                        "body": body_bytes,
                        "more_body": False,
                    }
                if not disconnect_ready.is_set():
                    await disconnect_ready.wait()
                return {"type": "http.disconnect"}

            async def _send(message: dict[str, Any]) -> None:
                sent_messages.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "method": "POST",
                "scheme": "http",
                "path": ENDPOINT,
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body_bytes)).encode("ascii")),
                ],
            }

            task_disc = asyncio.create_task(tracked_app(scope, _receive, _send))
            await asyncio.wait_for(arrivals[1].wait(), timeout=5.0)
            disconnect_ready.set()

            task_next = asyncio.create_task(
                client.post(ENDPOINT, json={"smiles": "c1ccccc1"})
            )
            await asyncio.wait_for(arrivals[2].wait(), timeout=5.0)
            assert barrier.entered_smiles == ["CC"]

            barrier.release_first.set()
            resp_inflight, _, resp_next = await asyncio.gather(
                task_inflight, task_disc, task_next
            )

        assert resp_inflight.status_code == 200
        start_msg = next(m for m in sent_messages if m["type"] == "http.response.start")
        body = b"".join(
            m.get("body", b"")
            for m in sent_messages
            if m["type"] == "http.response.body"
        )
        assert start_msg["status"] == 499 and body == b""

        assert resp_next.status_code == 200
        assert (
            len(GNNXAIInferenceResponse(**resp_next.json()).atomscores[TASKS[0]]) == 12
        )
        assert barrier.entered_smiles == ["CC", "c1ccccc1"]
        assert barrier.max_active_observed == 1
        assert app.state.inference_lock.locked() is False

    asyncio.run(_run())


def test_lifespan_wires_real_service_and_serves_requests_cleanly() -> None:
    """Production `lifespan` boots real `Chemprop` + IG explainer, serves HTTP 200 inference, and cleans up on exit."""
    isolated_app = FastAPI()
    isolated_app.include_router(gnn_router, prefix="/gnn-xai")

    async def _run() -> None:
        async with lifespan(isolated_app):
            service = isolated_app.state.inference_service
            lock = isolated_app.state.inference_lock

            assert isinstance(service, MolecularInferenceService)
            assert isinstance(service.predictor, Chemprop)
            assert service.predictor.n_tasks == TASKS_NUMBER
            assert service.predictor.device == DEVICE
            expected_thresholds = torch.tensor(
                [BINARY_PREDICTION_THRESHOLDS[t] for t in TASKS],
                dtype=torch.float32,
                device=DEVICE,
            )
            assert torch.equal(service.predictor.thresholds, expected_thresholds)
            assert isinstance(service.explainer, IntegratedGradientsExplainer)
            assert service.explainer.predictor is service.predictor
            assert service.explainer.steps == IG_STEPS
            assert isinstance(lock, asyncio.Lock) and not lock.locked()

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=isolated_app),
                base_url="http://testserver",
            ) as client:
                response = await client.post(ENDPOINT, json={"smiles": "CCO"})

            assert response.status_code == 200
            assert (
                len(GNNXAIInferenceResponse(**response.json()).atomscores[TASKS[0]])
                == 9
            )
            assert lock.locked() is False

        assert not hasattr(isolated_app.state, "inference_service")
        assert not hasattr(isolated_app.state, "inference_lock")

    asyncio.run(_run())
