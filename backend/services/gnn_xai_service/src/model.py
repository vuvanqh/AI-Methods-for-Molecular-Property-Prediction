from pydantic import BaseModel, Field
from typing import List, Dict, Any

class InferenceRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string sent by orchestrator")



class GNNXAIInferenceResponse(BaseModel):
    mappedsmiles: str = Field(
        ...,
    )
    prediction: Dict[str, bool]
    atomscores: Dict[str, List[float]]

class ErrorResponse(BaseModel):
    message: str = Field(..., description="Error message.")


 