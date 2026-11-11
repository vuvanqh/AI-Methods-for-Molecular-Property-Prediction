import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .controller import gnn_router
from .service import MolecularInferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up the singleton inference service and async concurrency lock."""
    # Initialize the model and explainer once at startup
    app.state.inference_service = MolecularInferenceService()
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

