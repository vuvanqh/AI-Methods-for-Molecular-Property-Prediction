import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import BINARY_PREDICTION_THRESHOLDS, DEVICE, IG_STEPS, TASKS, TASKS_NUMBER
from .controller import gnn_router
from .gnn_models.chemprop import Chemprop
from .service import MolecularInferenceService
from .xai.integrated_gradients import IntegratedGradientsExplainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up the singleton inference service and async concurrency lock."""
    # Initialize the model and explainer once at startup
    thresholds = [BINARY_PREDICTION_THRESHOLDS[task] for task in TASKS]
    predictor = Chemprop(TASKS_NUMBER, DEVICE, thresholds)
    explainer = IntegratedGradientsExplainer(predictor=predictor, steps=IG_STEPS)
    app.state.inference_service = MolecularInferenceService(predictor=predictor, explainer=explainer)
    # Lock to ensure only 1 request at a time executes the PyTorch model
    app.state.inference_lock = asyncio.Lock()

    try:
        yield
    finally:
        del app.state.inference_service
        del app.state.inference_lock


app = FastAPI(
    title="GNN and XAI Service", docs_url="/gnn-xai/docs", redoc_url="/gnn-xai/redoc", lifespan=lifespan
)

app.include_router(gnn_router, prefix="/gnn-xai")
