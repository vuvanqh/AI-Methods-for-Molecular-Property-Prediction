from .descriptors import describe_molecule
from .gnn_client import gnn_client
from .rag.retrieval import retrieve_similar_molecules_with_sources
from .rag.config import RAG_TOP_K
from sqlachemy.orm import Session

async def predict_properties(smiles: str, db: Session) -> dict:
    molecule = describe_molecule(smiles)  # rejects invalid SMILES before the expensive GNN call
    gnn_data = await gnn_client.get_prediction_and_attribution(smiles)

    similar_compouds_results = retrieve_similar_molecules_with_sources(smiles, db=db, top_k=RAG_TOP_K)

    similar_compounds_list = [
        {
            "standardized_smiles": molecule.standardized_smiles,
            "similarity_score": similarity_score,
            "sources": [
                source.source_url for source in molecule.molecule_sources
            ],
        }
        for molecule, similarity_score in similar_compouds_results
    ]

    return {
        "molecule": molecule,
        "properties_xai_data": gnn_data,
        "similar_compounds": similar_compounds_list,
        "llm_summary": "",
    }

