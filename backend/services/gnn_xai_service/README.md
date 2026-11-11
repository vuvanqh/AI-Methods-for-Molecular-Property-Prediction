# GNN & XAI Service (`gnn_xai_service`)

FastAPI microservice for multi-task molecular property prediction (18 endpoints across Tox21, CYP450, and hERG) using **Chemprop v2 (Directed MPNN)**, atom-level explainability via **Captum Integrated Gradients**, and **RDKit** MolBlock generation for **3Dmol.js** visualization.

---

## 1. Quick Start

### Run the Service
From the `gnn_xai_service` directory:
```bash
uv run uvicorn src.main:app --reload --port 8000
```

### API Documentation
- **Swagger UI:** http://127.0.0.1:8000/gnn-xai/docs
- **ReDoc:** http://127.0.0.1:8000/gnn-xai/redoc
- **Endpoint:** `POST /gnn-xai/prediction_and_attribution`

### Run the Test Suite
All `pytest` commands must be executed from the **`backend/services/gnn_xai_service`** folder:
```bash
cd backend/services/gnn_xai_service

# Run all unit, integration, and E2E tests
uv run pytest -v
```

---

## 2. Request Pipeline Overview

1. **`src/main.py` (`lifespan`)** initializes a single `MolecularInferenceService` instance and an `asyncio.Lock` on `app.state`.
2. **`src/controller.py` (`POST /gnn-xai/prediction_and_attribution`)** acquires `app.state.inference_lock`, aborts with HTTP `499` if the client disconnected while queued, and dispatches `inference_service.predict_and_explain(smiles)` via `run_in_threadpool`.
3. **`src/service.py` (`MolecularInferenceService.predict_and_explain`)**:
   - Validates and parses the input SMILES with RDKit (`Chem.MolFromSmiles`), raising `MoleculeValidationError` (HTTP `422`) on invalid input.
   - Normalizes the molecule via a canonical SMILES round-trip (`Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))`) so atom indices and tetrahedral chiral tags (`@`/`@@`) are independent of input SMILES traversal order.
   - Calls `Chemprop.predict(mol)` (`src/gnn_models/chemprop.py`) to compute 18 probabilities, binary predictions, and message-passing atom embeddings $H \in \mathbb{R}^{K \times d}$.
   - Calls `IntegratedGradientsExplainer.explain(...)` (`src/xai/integrated_gradients.py`) to compute per-atom attribution scores for all 18 tasks relative to a zero-embedding baseline.
   - Calls `mol_block_generation(mol)` (`src/mol_block_generation.py`) to add explicit hydrogens (`Chem.AddHs`), embed coordinates (`ETKDGv3` with seed `0xf00d`, fallback to `useRandomCoords=True`, and built-in 2D coordinate fallback in `Chem.MolToMolBlock`), and return `(molblock, mol_with_h)`.
   - Verifies that heavy atoms remain at indices `0 .. K-1` and appended hydrogens at `K .. N-1`, mapping heavy-atom IG scores to `float` and hydrogen scores to `None`.

---

## 3. Codebase Structure (`src/`)

| File | Purpose & How It Works |
| :--- | :--- |
| **`src/main.py`** | FastAPI application entrypoint. Manages the `lifespan` context manager that creates `app.state.inference_service` and `app.state.inference_lock` once at startup and mounts `gnn_router` under `/gnn-xai`. |
| **`src/controller.py`** | Defines `POST /gnn-xai/prediction_and_attribution`. Serializes model execution with `async with inference_lock`, checks `request.is_disconnected()` (HTTP `499`), runs synchronous CPU/GPU inference in Starlette's threadpool, and maps `MoleculeValidationError` $\rightarrow$ HTTP `422` and `ModelInferenceError` / `RuntimeError` $\rightarrow$ HTTP `500`. |
| **`src/service.py`** | Core domain orchestrator (`MolecularInferenceService`). Validates SMILES, performs canonical SMILES normalization, calls the GNN predictor, IG explainer, and MolBlock generator, enforces intermediate shape/value invariants, and assembles the response dictionary with `None` scores for added visualization hydrogens. |
| **`src/gnn_models/chemprop.py`** | `Chemprop` wrapper around Chemprop v2's `MPNN` (`SimpleMoleculeMolGraphFeaturizer`, `BondMessagePassing`, `NormAggregation`, `BinaryClassificationFFN`). Splits inference into `get_embeddings(batch_graph)` (message passing) and `forward_from_embeddings(atom_embeddings, mol_graph_index)` (aggregation $\rightarrow$ batchnorm $\rightarrow$ FFN) so Captum can interpolate and differentiate with respect to atom embeddings. |
| **`src/xai/integrated_gradients.py`** | `IntegratedGradientsExplainer` using Captum's `IntegratedGradients`. Interpolates between a zero baseline `torch.zeros_like(inputs)` and the molecule's `atom_embeddings` over `IG_STEPS` (`100`) steps for each target task, then sums attributions across the embedding dimension (`dim=-1`) to yield one scalar score per heavy atom. |
| **`src/mol_block_generation.py`** | `mol_block_generation(mol)` prepares a V2000 MolBlock for 3Dmol.js. Calls `Chem.AddHs(mol)` (appending hydrogens after heavy atoms), embeds 3D coordinates via `AllChem.ETKDGv3()` (`randomSeed = 0xf00d`), retries with `params.useRandomCoords = True` if standard embedding returns `-1`, and calls `Chem.MolToMolBlock(mol_with_h)` (which automatically generates 2D coordinates with `z = 0.0000` if 0 conformers exist). |
| **`src/model.py`** | Pydantic v2 request/response schemas: `InferenceRequest`, `GNNXAIInferenceResponse`, and `ErrorResponse`. |
| **`src/config.py`** | Constants defining the 18 endpoints (`TOX_21`, `CYP450`, `HERG`, combined into `TASKS`), `BINARY_PREDICTION_THRESHOLD` (`0.5`), `DEVICE` (`cuda` or `cpu`), and `IG_STEPS` (`100`). |
| **`src/exceptions.py`** | Custom domain exceptions: `MoleculeValidationError(ValueError)` (client input error, HTTP `422`) and `ModelInferenceError(RuntimeError)` (internal invariant/engine failure, HTTP `500`). |

---

## 4. Test Suite Structure (`tests/`)

The `tests/` directory mirrors `src/`:

- **`tests/conftest.py`**: Shared fixtures (`chemprop_predictor`, `ig_explainer`, `inference_service`, representative SMILES fixtures) and deterministic PyTorch seeding.
- **`tests/unit/`**:
  - `test_config.py`, `test_exceptions.py`, `test_model.py`, `test_controller.py`: Unit tests for configuration constants, exception hierarchy, Pydantic validation, and controller status code mapping (`200`, `422`, `499`, `500`).
  - `test_service.py`: SMILES parsing, implicit/explicit/isotopic hydrogen handling, equivalent SMILES & graph isomorphism, canonicalization, and service-level invariant guardrails.
  - `test_mol_block_generation.py`: 3D conformer generation, determinism (`0xf00d`), `useRandomCoords=True` retry, `MolToMolBlock` automatic 2D coordinate fallback, and 0-based RDKit/3Dmol.js vs. 1-based V2000 bond-table indexing.
  - `gnn_models/test_chemprop.py`: Chemprop forward pass, embedding shapes, and prediction invariance across equivalent SMILES.
  - `xai/test_integrated_gradients.py`: Captum IG shapes, finiteness, completeness axiom ($\sum_i \text{IG}_i \approx f(H) - f(0)$), and atom-mapped attribution invariance across equivalent SMILES.
- **`tests/integration/`**:
  - `test_service.py`: Real `MolecularInferenceService` + `Chemprop` + `IntegratedGradientsExplainer` + `mol_block_generation` pipeline integration tests.
- **`tests/e2e/`**:
  - `test_main.py`: Full HTTP end-to-end tests against the FastAPI `app` (lifespan startup/teardown, HTTP `200` and `422` validation, concurrency queuing under `asyncio.Lock`, and OpenAPI schema).

---


