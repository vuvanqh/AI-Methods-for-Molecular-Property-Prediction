from pydantic import BaseModel, Field
from typing import List, Dict, Any

class InferenceRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string sent by orchestrator")

class InferenceResponse(BaseModel):
    smiles: str #TODO INCLUDE OR NOT
    predictions: Dict[str, Any] = Field(..., description="GNN predictions (Tox21, hERG, CYP450)")
    xai_attributions: Dict[str, List[Any]] = Field(..., description="XAI atom attribution scores")

class ErrorResponse(BaseModel):
    message: str = Field(..., description="Error message.")