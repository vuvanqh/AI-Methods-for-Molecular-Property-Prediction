from fastapi import FastAPI
from .controller import gnn_router
from .service import MolecularInferenceService
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialize the model and explainer at startup - once, not per request
    app.state.inference_service = MolecularInferenceService()

    try:
        yield
    finally:
        # at shutdown release
        del app.state.inference_service

app = FastAPI(
    title="GNN and XAI Service", docs_url="/gnn-xai/docs", redoc_url="/gnn-xai/redoc", lifespan=lifespan
)

app.include_router(gnn_router, prefix="/gnn-xai")









