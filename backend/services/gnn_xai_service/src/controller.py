import logging
from fastapi import APIRouter, status, Request, Response
from .model import ErrorResponse, GNNXAIInferenceResponse, InferenceRequest
from .exceptions import MoleculeValidationError, ModelInferenceError
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

gnn_router = APIRouter()

@gnn_router.post(
    "/prediction_and_attribution",
    response_model=GNNXAIInferenceResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Unprocessable Entity: RDKit failed to parse the given SMILES string or invalid molecular input.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error: GNN inference, shape assertion, or XAI computation failure.",
        },
    },
    summary="Run GNN prediction and XAI attribution for a given SMILES string",
)
async def prediction_and_attribution_endpoint(request: Request, payload: InferenceRequest):
    inference_service = request.app.state.inference_service
    inference_lock = request.app.state.inference_lock

    # Acquire the single-inference lock asynchronously without blocking thread pool
    async with inference_lock:
        # Check if caller already disconnected/timed out while waiting in queue
        if await request.is_disconnected():
            logger.warning("Client disconnected while waiting for inference lock. Aborting calculation.")
            return Response(status_code=499)

        try:
            result = await run_in_threadpool(
                inference_service.predict_and_explain,
                payload.smiles
            )
            return result

        except MoleculeValidationError as exc:
            logger.info("Molecule validation error: %s", exc)
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"message": str(exc)},
            )

        except (ModelInferenceError, RuntimeError) as exc:
            logger.error("Internal model inference failure: %s", exc, exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"Inference engine failure: {str(exc)}"},
            )

        except Exception as exc:
            logger.error("Unexpected error occurred during inference: %s", exc, exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": "An unexpected error occurred during prediction and attribution."},
            )

