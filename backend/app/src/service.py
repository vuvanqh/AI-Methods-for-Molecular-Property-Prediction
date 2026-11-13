import asyncio
import logging

from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from .clients import GNNClient, LLMClient
from .descriptors import describe_molecule
from .rag import Retriever
from .rag.augmentation import ClientStrategyBase, build_prompt, gather_evidence
from .rag.config import RAG_TOP_K, SOURCE_URL_TEMPLATES
from .rag.morgan_fingerprint_helper import get_molecular_formula


logger = logging.getLogger(__name__)


async def _summarize(llm_client: LLMClient, prompt: str) -> str:
    try:
        return await llm_client.get_summary(prompt)
    except (ConnectionError, RuntimeError) as exc:
        # an empty summary hides the interpretation card instead of failing the prediction
        logger.warning("LLM summary skipped: %s", exc)
        return ""


async def predict_properties(
    smiles: str,
    retriever: Retriever,
    evidence_sources: list[ClientStrategyBase],
    llm_client: LLMClient,
    gnn_client: GNNClient,
    db: Session,
) -> dict:
    molecule = describe_molecule(
        smiles
    )  # rejects invalid SMILES before the expensive GNN call

    ret = retriever.create_retriever(db)
    results = await run_in_threadpool(ret.retrieve, smiles, RAG_TOP_K)

    # the external evidence lookups take as long as the GNN, so they run side by side
    gnn_data, (evidence, unavailable) = await asyncio.gather(
        gnn_client.get_prediction_and_attribution(smiles),
        gather_evidence(evidence_sources, molecule["smiles"], results),
    )
    llm_summary = await _summarize(
        llm_client,
        build_prompt(smiles, molecule, gnn_data, results, evidence, unavailable),
    )

    similar_compounds_list = [
        {
            "smiles": result.molecule.standardized_smiles,
            "molecular_formula": get_molecular_formula(
                result.molecule.standardized_smiles
            ),
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
