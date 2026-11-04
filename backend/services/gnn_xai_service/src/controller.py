from fastapi import APIRouter, status, Request
from .model import ErrorResponse, GNNXAIInferenceResponse, InferenceRequest
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

gnn_router = APIRouter()

@gnn_router.post(
    "/prediction_and_attribution",
    response_model=GNNXAIInferenceResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Unprocessable Entity: RDKit failed to parse the given SMILES string.",
        }
    },
    summary="Run GNN prediction and XAI attribution for a given SMILES string",
)
async def prediction_and_attribution_endpoint(request: Request, payload: InferenceRequest):
    inference_service = request.app.state.inference_service

    try:
        result = await run_in_threadpool(
            inference_service.predict_and_explain,
            payload.smiles
        )
        return result

    except ValueError as exc:

        return JSONResponse( 
            status_code=422,
            content={"message": "The molecule could not be processed."},
        )
