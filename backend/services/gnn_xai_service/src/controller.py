from fastapi import APIRouter, status
from .model import InferenceRequest, InferenceResponse, ErrorResponse
from .service import gnn_xai_inference

gnn_router = APIRouter()

@gnn_router.post(
    "/prediction_and_attribution",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    responses = {
            422: {
                "model": ErrorResponse,
                "description": "Unprocessable Entity: RDKit failed to parse the given SMILES string."
            },
            503: {
                "model": ErrorResponse,
                "description": "Service Unavailable: The isolated GNN-XAI microservice container is unreachable or timed out."
            }
        },
    summary="Run GNN prediction and XAI attribution for a given SMILES string",
)
async def prediction_and_attribution_endpoint(request: InferenceRequest):
    gnn_xai_result = await gnn_xai_inference(request.smiles)
    
    return InferenceResponse(**gnn_xai_result)