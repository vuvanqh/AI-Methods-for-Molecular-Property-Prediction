from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Request payload containing the input SMILES string."""

    smiles: str = Field(..., description="SMILES string sent by orchestrator")


class GNNXAIInferenceResponse(BaseModel):
    """Response payload with binary predictions, probabilities, per-atom IG scores, and a V2000 MolBlock."""

    predictions: dict[str, bool]
    probabilities: dict[str, float]
    atomscores: dict[str, dict[int, float | None]] = Field(
        ...,
        description=(
            "Atom-level attribution scores per predicted property: the outer key is the "
            "task/property name, the inner key is the atom index, and the inner value is "
            "the attribution score (float) for heavy atoms or None for explicit hydrogens."
        ),
    )
    molblock: str = Field(
        description="MolBlock representation of the molecule for visualization purposes."
    )


class ErrorResponse(BaseModel):
    """Standard JSON error payload returned on HTTP 422 and 500 responses."""

    message: str = Field(..., description="Error message.")
