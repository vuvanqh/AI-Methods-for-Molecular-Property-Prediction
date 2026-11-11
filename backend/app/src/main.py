from contextlib import asynccontextmanager

from fastapi import FastAPI
from .chembl_client import chembl_client
from .controller import app_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await chembl_client.aclose()


app = FastAPI(
    title="AI Molecular Property Prediction Service",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.include_router(app_router, prefix="/api")
