from .descriptors import describe_molecule
from .gnn_client import gnn_client
from .rag.molecule_retrieval import create_retriever
from .rag.retrieval_type import RetrievalType
from .rag.config import RAG_TOP_K, SOURCE_URL_TEMPLATES
from sqlalchemy.orm import Session


async def predict_properties(smiles: str, db: Session) -> dict:
    molecule = describe_molecule(
        smiles
    )  # rejects invalid SMILES before the expensive GNN call
    gnn_data = await gnn_client.get_prediction_and_attribution(smiles)

    retriever = create_retriever(RetrievalType.FINGERPRINT, db)
    results = retriever.retrieve(smiles, RAG_TOP_K)

    similar_compounds_list = [
        {
            "smiles": result.molecule.standardized_smiles,
            "similarity_score": result.score,
            "sources": [
                SOURCE_URL_TEMPLATES[source.provider].format(source.source_identifier)
                for source in result.molecule.molecule_sources
            ],
        }
        for result in results
    ]

    return {
        "molecule": molecule,
        "properties_xai_data": gnn_data,
        "similar_compounds": similar_compounds_list,
        "llm_summary": "",
    }
