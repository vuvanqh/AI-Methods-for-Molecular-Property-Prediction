from fastapi import FastAPI
from .controller import gnn_router



app = FastAPI(
    title="GNN and XAI Service",
    docs_url="/gnn-xai/docs",
    redoc_url="/gnn-xai/redoc")

app.include_router(gnn_router, prefix="/gnn-xai")



