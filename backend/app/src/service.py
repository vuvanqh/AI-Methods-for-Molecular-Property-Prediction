import asyncio
import logging

from .descriptors import describe_molecule
from .gnn_client import gnn_client
from .llm_client import llm_client
from .rag.augmentation import ChemblRetrieval
from .rag.augmentation.evidence_collection import gather_evidence
from .rag.augmentation.prompt_builder import build_prompt
from .rag.molecule_retrieval import create_retriever
from .rag.retrieval_type import RetrievalType
from .rag.config import RAG_TOP_K, SOURCE_URL_TEMPLATES
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# each molecule is looked up only at the sources it was ingested from
EVIDENCE_SOURCES = [ChemblRetrieval()]


async def _summarize(prompt: str) -> str:
    try:
        return await llm_client.get_summary(prompt)
    except (ConnectionError, RuntimeError) as exc:
        # an empty summary hides the interpretation card instead of failing the prediction
        logger.warning("LLM summary skipped: %s", exc)
        return ""


async def predict_properties(smiles: str, db: Session) -> dict:
    molecule = describe_molecule(
        smiles
    )  # rejects invalid SMILES before the expensive GNN call

    # TO-DO: CURRENTLY BLOCKING
    retriever = create_retriever(RetrievalType.FINGERPRINT, db)
    results = retriever.retrieve(smiles, RAG_TOP_K)

    # the external evidence lookups take as long as the GNN, so they run side by side
    gnn_data, (evidence, unavailable) = await asyncio.gather(
        gnn_client.get_prediction_and_attribution(smiles),
        gather_evidence(EVIDENCE_SOURCES, molecule["smiles"], results),
    )
    llm_summary = await _summarize(
        build_prompt(smiles, molecule, gnn_data, results, evidence, unavailable)
    )

    similar_compounds_list = [
        {
            "smiles": result.molecule.standardized_smiles,
            "similarity_score": result.score,
            "sources": [
                SOURCE_URL_TEMPLATES[source.provider].format(source.source_identifier)
                for source in result.molecule.molecule_sources
                # a provider ingestion added but that has no link template yet
                if source.provider in SOURCE_URL_TEMPLATES
            ],
        }
        for result in results
    ]

    return {
        "molecule": molecule,
        "properties_xai_data": gnn_data,
        "similar_compounds": similar_compounds_list,
        "llm_summary": llm_summary,
    }
