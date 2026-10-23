from pydantic import BaseModel, Field
from typing import List, Dict, Any

class InferenceRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string sent by orchestrator")

class PropertyDataModel(BaseModel): 
    prediction: bool = Field(
        ..., 
        description="Binary prediction result.",
        example=True
    )
    xai_attributions: List[float] = Field(
        ..., 
        description="Positional attribution array mapped to atom indices.",
        example=[0.1, 0.2, 0.3]
    ) #TODO add what else is needed for XAI


class GNNXAIInferenceResponse(BaseModel):
    metabolism: Dict[str, PropertyDataModel] = Field(
        ..., 
        description="Properties predictions with XAI attributions for metabolism-related properties."
    )
    toxicity: Dict[str, PropertyDataModel] = Field(
        ..., 
        description="Pultitask-GNN properties predictions with XAI attributions for toxicity-related properties."
    )
    
class ErrorResponse(BaseModel):
    message: str = Field(..., description="Error message.")


 