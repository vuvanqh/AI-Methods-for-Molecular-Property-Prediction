from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string sent by orchestrator")
   

class GNNXAIInferenceResponse(BaseModel):
    
    predictions: dict[str, bool]
    probabilities: dict[str, float]
    atomscores: dict[str, dict[int, float]] = Field(..., description="Atom-level scores for each atom in the molecule, where the key of the outer dict is the property, key of the inne dict is the atom index and the value is a dictionary of scores for each property.")
    molblock: str = Field( description="MolBlock representation of the molecule for visualization purposes.")

class ErrorResponse(BaseModel):
    message: str = Field(..., description="Error message.")
