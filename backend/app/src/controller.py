from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .clients import GNNClient, LLMClient
from .db.session import get_db
from .dependencies import (
    get_evidence_sources,
    get_gnn_client,
    get_llm_client,
    get_retriever,
)
from .model import ErrorResponse, InferenceRequest, PredictionResponse
from .rag import Retriever
from .rag.augmentation import ClientStrategyBase
from .service import predict_properties

app_router = APIRouter()


@app_router.post(
    "/predict-properties",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    description="Predicts properties using GNN, attributes to substructures using XAI, extracts similar molecules and generates summary",
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
)
async def predict_properties_endpoint(
    request: InferenceRequest,
    db: Session = Depends(get_db),
    evidence_sources: list[ClientStrategyBase] = Depends(get_evidence_sources),
    retriever: Retriever = Depends(get_retriever),
    llm_client: LLMClient = Depends(get_llm_client),
    gnn_client: GNNClient = Depends(get_gnn_client),
):
    try:
        result = await predict_properties(
            request.smiles, retriever, evidence_sources, llm_client, gnn_client, db
        )
        return result

    except ValueError:
        return JSONResponse(
            status_code=422,
            content={"message": "The molecule could not be processed."},
        )

    except ConnectionError:
        return JSONResponse(
            status_code=503,
            content={"message": "GNN service unavailable."},
        )
