from fastapi import FastAPI
from .controller import app_router

app = FastAPI(
    title = "AI Molecular Property Prediction Service",
    docs_url="/api/docs",
    redoc_url="/api/redoc")

app.include_router(app_router, prefix="/api")