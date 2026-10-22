from .gnn_xai_client import request_gnn_xai_inference

async def predict_properties(smiles: str) -> dict:
    gnn_xai_data = await request_gnn_xai_inference(smiles)
    
    return {
        "smiles": smiles,
        "predictions": gnn_xai_data.get("predictions", {}),
        "xai_attributions": gnn_xai_data.get("xai_attributions", {}),
        "similar_compounds": [], # todo
        "llm_summary": None      # todo
    }
       
    