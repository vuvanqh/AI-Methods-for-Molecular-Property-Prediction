import asyncio
from typing import Any

import pytest
import torch
from fastapi import FastAPI

from src.config import BINARY_PREDICTION_THRESHOLDS, IG_STEPS, TASKS, TASKS_NUMBER
from src.controller import gnn_router
from src.gnn_models.chemprop import Chemprop
from src.service import MolecularInferenceService
from src.xai.integrated_gradients import IntegratedGradientsExplainer

FAST_IG_STEPS = 20


def create_controller_app(service: Any) -> FastAPI:
    """Build a FastAPI app mounting `gnn_router` with an isolated `asyncio.Lock`."""
    test_app = FastAPI()
    test_app.state.inference_service = service
    test_app.state.inference_lock = asyncio.Lock()
    test_app.include_router(gnn_router, prefix="/gnn-xai")
    return test_app


@pytest.fixture(scope="session")
def real_chemprop() -> Chemprop:
    """Deterministically seeded real `Chemprop` model shared read-only across the session."""
    torch.manual_seed(42)
    thresholds = [BINARY_PREDICTION_THRESHOLDS[task] for task in TASKS]
    return Chemprop(TASKS_NUMBER, torch.device("cpu"), thresholds)


@pytest.fixture(scope="session")
def fast_ig_explainer(real_chemprop: Chemprop) -> IntegratedGradientsExplainer:
    """Real `IntegratedGradientsExplainer` wired to `real_chemprop` with 20 steps."""
    return IntegratedGradientsExplainer(predictor=real_chemprop, steps=FAST_IG_STEPS)


@pytest.fixture(scope="session")
def production_ig_explainer(real_chemprop: Chemprop) -> IntegratedGradientsExplainer:
    """Real `IntegratedGradientsExplainer` using production `IG_STEPS` (100)."""
    return IntegratedGradientsExplainer(predictor=real_chemprop, steps=IG_STEPS)


@pytest.fixture(scope="session")
def real_inference_service(
    real_chemprop: Chemprop, fast_ig_explainer: IntegratedGradientsExplainer
) -> MolecularInferenceService:
    """Real `MolecularInferenceService` wired to real `Chemprop` and IG explainer."""
    return MolecularInferenceService(real_chemprop, fast_ig_explainer)
