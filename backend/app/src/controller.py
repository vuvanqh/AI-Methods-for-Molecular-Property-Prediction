from fastapi import APIRouter, status
from .model import PredictionRequest, PredictionResponse, ErrorResponse
from .service import predict_properties

app_router = APIRouter()

@app_router.post(
    "/predict-properties", 
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    description="Predicts properties using GNN, attributes to substructures using XAI, extracts similar molecules and generates summary",
    responses = {
        422: {
            "model": ErrorResponse,
            "description": "Unprocessable Entity: RDKit failed to parse the given SMILES string."
        },
        503: {
            "model": ErrorResponse,
            "description": "Service Unavailable: The isolated GNN-XAI microservice container is unreachable or timed out."
        }
    }) #todo status code??
async def predict_properties_endpoint(request: PredictionRequest):
    return await predict_properties(request.smiles)
