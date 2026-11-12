from contextlib import asynccontextmanager

from fastapi import FastAPI

from .clients import ChEMBLClient, GNNClient, LLMClient
from .controller import app_router
from .rag import RetrievalType, Retriever
from .rag.augmentation import ChemblRetrieval
from .rag.embeddings.unimol_embeddings import UniMolEmbeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    chembl_client = ChEMBLClient()
    embedding_model = UniMolEmbeddings()
    llm_client = LLMClient()
    gnn_client = GNNClient()

    app.state.evidence_sources = [ChemblRetrieval(chembl_client)]
    app.state.retriever = Retriever(RetrievalType.FUSION, embedding_model)
    app.state.llm_client = llm_client
    app.state.gnn_client = gnn_client
    try:
        yield
    finally:
        await chembl_client.aclose()
        await gnn_client.aclose()
        await llm_client.aclose()
        embedding_model.unload()


app = FastAPI(
    title="AI Molecular Property Prediction Service",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.include_router(app_router, prefix="/api")
