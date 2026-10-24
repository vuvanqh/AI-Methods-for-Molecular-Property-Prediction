from fastapi import APIRouter, status
from .model import ErrorResponse, GNNXAIInferenceResponse, InferenceRequest
from .service import gnn_xai_inference

gnn_router = APIRouter()


@gnn_router.post(
    "/prediction_and_attribution",
    response_model=GNNXAIInferenceResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Unprocessable Entity: RDKit failed to parse the given SMILES string.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Service Unavailable: The isolated GNN-XAI microservice container is unreachable or timed out.",
        },
    },
    summary="Run GNN prediction and XAI attribution for a given SMILES string",
)
async def prediction_and_attribution_endpoint(request: InferenceRequest):
    return await gnn_xai_inference(request.smiles)
