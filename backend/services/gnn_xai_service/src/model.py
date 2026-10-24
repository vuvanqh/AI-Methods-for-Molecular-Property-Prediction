from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string sent by orchestrator")


class GNNXAIInferenceResponse(BaseModel):
    mappedsmiles: str = Field(
        ...,
    )
    prediction: dict[str, bool]
    atomscores: dict[str, list[float]]


class ErrorResponse(BaseModel):
    message: str = Field(..., description="Error message.")
