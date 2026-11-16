# Orchestrator API (`app`)

FastAPI service that coordinates GNN inference, similar-compound retrieval, ChEMBL evidence, and the LLM interpretation.

## Quick Start

From `backend/app`:

```bash
# Run the service
uv run uvicorn src.main:app --reload --port 8001

# Run tests
uv run pytest -v
```

- **Endpoint:** `POST /api/predict-properties`
- **Swagger UI:** http://127.0.0.1:8001/api/docs
- **ReDoc:** http://127.0.0.1:8001/api/redoc
