from .descriptors import describe_molecule
from .gnn_client import gnn_client

async def predict_properties(smiles: str) -> dict:
    molecule = describe_molecule(smiles)  # rejects invalid SMILES before the expensive GNN call
    gnn_data = await gnn_client.get_prediction_and_attribution(smiles)

    return {
        "molecule": molecule,
        "properties_xai_data": gnn_data,
        "similar_compounds": [],
        "llm_summary": "",
    }
