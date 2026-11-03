from .gnn_client import gnn_client

async def predict_properties(smiles: str) -> dict:
    gnn_data = await gnn_client.get_prediction_and_attribution(smiles)

    return {
        "properties_xai_data": gnn_data,
        "similar_compounds": [],
        "llm_summary": "",
    }
