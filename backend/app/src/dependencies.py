from fastapi import Request

from .clients import GNNClient, LLMClient
from .rag import Retriever
from .rag.augmentation import ClientStrategyBase


def get_evidence_sources(request: Request) -> list[ClientStrategyBase]:
    return request.app.state.evidence_sources


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_gnn_client(request: Request) -> GNNClient:
    return request.app.state.gnn_client
