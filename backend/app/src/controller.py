from fastapi import APIRouter, status, Depends
from fastapi.responses import JSONResponse
from .model import InferenceRequest, PredictionResponse, ErrorResponse
from .service import predict_properties
from .db.session import get_db
from sqlalchemy.orm import Session

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
async def predict_properties_endpoint(request: InferenceRequest, db: Session = Depends(get_db)):
    try:
        result = await predict_properties(request.smiles, db)
        return result

    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"message": "The molecule could not be processed."},
        )

    except ConnectionError as exc:
        return JSONResponse(
            status_code=503,
            content={"message": "GNN service unavailable."},
        )
