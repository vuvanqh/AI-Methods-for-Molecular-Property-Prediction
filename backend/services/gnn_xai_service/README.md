# GNN & XAI Service (`gnn_xai_service`)

FastAPI service for multi-task molecular property prediction (**Chemprop v2**), atom-level explainability (**Captum Integrated Gradients**), and 3D **RDKit** MolBlock generation.

## Quick Start

From `backend/services/gnn_xai_service`:

```bash
# Run the service
uv run uvicorn src.main:app --reload --port 8000

# Run tests
uv run pytest -v
```

- **Endpoint:** `POST /gnn-xai/prediction_and_attribution`
- **Swagger UI:** http://127.0.0.1:8000/gnn-xai/docs
- **ReDoc:** http://127.0.0.1:8000/gnn-xai/redoc

## Structure

- `src/main.py` — FastAPI app setup and dependency wiring (`lifespan`).
- `src/controller.py` — HTTP route, single-inference `asyncio.Lock`, and status code mapping (`200`, `422`, `499`, `500`).
- `src/service.py` — `MolecularInferenceService` orchestrating SMILES validation, prediction, XAI attribution, and MolBlock assembly.
- `src/gnn_models/` — `GNNPredictor` interface (`gnn_interface.py`) and `Chemprop` implementation (`chemprop.py`).
- `src/xai/` — `XAIExplainer` interface (`xai_interface.py`) and `IntegratedGradientsExplainer` implementation (`integrated_gradients.py`).
- `src/mol_block_generation.py` — 3D conformer embedding and V2000 MolBlock generation.
- `src/config.py`, `src/model.py`, `src/exceptions.py` — Tasks & per-task thresholds, Pydantic schemas, and domain exceptions.
- `tests/unit/` & `tests/e2e/` — Unit tests (interface contracts and stubbed collaborators) and HTTP end-to-end tests.
