class MoleculeValidationError(ValueError):
    """Raised when the molecule SMILES input is invalid, empty, or unparseable by RDKit (HTTP 422)."""
    pass

class ModelInferenceError(RuntimeError):
    """Raised when internal invariant checks, tensor shapes, or model calculations fail (HTTP 500)."""
    pass

