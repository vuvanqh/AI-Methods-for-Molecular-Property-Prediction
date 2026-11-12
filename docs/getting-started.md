# Getting Started

This guide covers configuring, running, and developing the **AI Methods for Molecular Property Prediction** platform.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Configuration (.env)](#configuration-env)
- [Option 1: Docker Compose (Recommended)](#option-1-docker-compose-recommended)
- [Option 2: Local Development](#option-2-local-development)
  - [1. Start the Database](#1-start-the-database)
  - [2. Apply Alembic Migrations](#2-apply-alembic-migrations)
  - [3. Ingest the Reference Corpus](#3-ingest-the-reference-corpus)
  - [4. Start the GNN & XAI Service](#4-start-the-gnn--xai-service)
  - [5. Start the Orchestrator API](#5-start-the-orchestrator-api)
  - [6. Start the Frontend Dev Server](#6-start-the-frontend-dev-server)
- [Service Endpoints](#service-endpoints)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Docker Desktop** (with Docker Compose v2)
- **uv**, the Python package and environment manager: [install uv](https://docs.astral.sh/uv/getting-started/installation/). uv downloads the pinned Python versions itself (`.python-version`: 3.15 for the repository root and `backend/app`, 3.14 for the GNN service, 3.13 for `experiments/gnn`).
- **Node.js 22.12 or newer** (CI uses Node 24) and **npm**
- **Git**

## Configuration (.env)

Create `.env` at the repository root from the template:

```bash
cp .env.example .env
```

Then set your OpenRouter API key:
```env
# Required for the LLM interpretation; Docker Compose refuses to start without it
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional: choose another OpenRouter model
LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

> [!IMPORTANT]
> `.env.example` sets `DATABASE_URL` and `GNN_SERVICE_URL` to the Docker network names (`db:5432`, `gnn-xai:8000`). Docker Compose sets both for the `api` container itself, so the values in `.env` only matter for processes on the host. For local development (Option 2), switch them to the commented variants: `127.0.0.1:5433` for the database and `127.0.0.1:8000` for the GNN service. The orchestrator API and the ingestion script both read the root `.env`.

## Option 1: Docker Compose (Recommended)

Build and start the full stack:

```bash
docker compose up --build
```

Docker Compose starts four containers:
1. **`db`** (`pgvector/pgvector:pg17`): PostgreSQL on host port `5433` (container port `5432`).
2. **`gnn-xai`**: Chemprop v2 and Captum XAI service on port `8000`.
3. **`api`**: FastAPI orchestrator on port `8001`. It waits until `db` and `gnn-xai` are healthy.
4. **`frontend`**: production React build served by Nginx on port `5173`.

To stop the containers:
```bash
docker compose down
```

To also delete the database volume and start fresh:
```bash
docker compose down -v
```

## Option 2: Local Development

Running the services on the host gives hot reloading and faster feedback while you change code.

### 1. Start the Database

Start only the PostgreSQL container:

```bash
docker compose up db -d
```

Check that it is healthy:
```bash
docker compose ps db
```

The database listens on `127.0.0.1:5433` with user `postgres`, password `postgres`, and database `molecules`.

### 2. Apply Alembic Migrations

From the `backend` directory:

```bash
cd backend
uv run alembic upgrade head
cd ..
```

This enables the `vector` extension and creates the tables `molecules`, `molecule_sources`, `molecule_embeddings`, and `analyses`.

### 3. Ingest the Reference Corpus

Fill the similarity corpus with molecules, Morgan fingerprints (`bit(2048)`), and 512-dimensional Uni-Mol embeddings:

```bash
uv run python -m scripts.ingestion.ingest
```

> The script fetches ChEMBL molecules in pages of 1,000, standardizes the SMILES with RDKit, computes both representations, and inserts them into PostgreSQL. Each page is committed separately, so you can stop the script and re-run it at any time. See the **[Corpus Ingestion guide](corpus-ingestion.md)** for details.

### 4. Start the GNN & XAI Service

In a new terminal:

```bash
cd backend/services/gnn_xai_service
uv sync
uv run uvicorn src.main:app --reload --port 8000
```

- Endpoint: `POST http://127.0.0.1:8000/gnn-xai/prediction_and_attribution`
- Swagger UI: `http://127.0.0.1:8000/gnn-xai/docs`

### 5. Start the Orchestrator API

In another terminal:

```bash
cd backend/app
uv sync
uv run uvicorn src.main:app --reload --port 8001
```

- Endpoint: `POST http://127.0.0.1:8001/api/predict-properties`
- Swagger UI: `http://127.0.0.1:8001/api/docs`

### 6. Start the Frontend Dev Server

In a third terminal:

```bash
cd frontend
npm install
npm run dev
```

The app is then available at http://localhost:5173.

## Service Endpoints

| Service | Host URL | Documentation | Purpose |
|---|---|---|---|
| **Frontend** | `http://localhost:5173` | [Frontend styling](frontend-styling.md) | SMILES input and 3D attribution viewer |
| **Orchestrator API** | `http://localhost:8001` | [`/api/docs`](http://127.0.0.1:8001/api/docs), [`/api/redoc`](http://127.0.0.1:8001/api/redoc) | Coordinates GNN inference, retrieval, ChEMBL evidence, and the LLM |
| **GNN & XAI Service** | `http://localhost:8000` | [`/gnn-xai/docs`](http://127.0.0.1:8000/gnn-xai/docs) | 18-task Chemprop model, Integrated Gradients, 3D conformers |
| **PostgreSQL + pgvector** | `127.0.0.1:5433` | [Database design](database-design.md) | Molecules, fingerprints, Uni-Mol embeddings, and analyses |

## Running Tests

### Frontend
From `frontend/`:

```bash
# Vitest unit and integration tests in watch mode
npm test

# Single run with coverage thresholds (80% lines/statements/functions, 75% branches)
npm run test:coverage

# Playwright E2E tests against a mocked backend
npm run test:e2e

# Playwright full-stack smoke test against a running backend
npm run test:e2e:fullstack
```

### Orchestrator API
From `backend/app/`:

```bash
uv run pytest -v
```

### GNN & XAI Service
From `backend/services/gnn_xai_service/`:

```bash
uv run pytest -v
```

## Troubleshooting

### 1. Database Port (5433 vs 5432)
- **Symptom:** host tools connect to the wrong database, or authentication fails, on port `5432`.
- **Cause:** many machines already run a local PostgreSQL on `5432`.
- **Solution:** Docker Compose maps host port `5433` to container port `5432`. Connect host tools to `127.0.0.1:5433`; containers inside the Docker network use `db:5432`.

### 2. Connections to `localhost` Hang on Windows
- **Symptom:** a database or HTTP connection to `localhost` hangs.
- **Cause:** on Windows, `localhost` often resolves to `::1` (IPv6) first, while the Docker ports are bound to `127.0.0.1` only.
- **Solution:** use `127.0.0.1` instead of `localhost` in connection strings and service URLs.

### 3. PyTorch CPU vs CUDA Builds
- **Backend services:** `backend/app` and `backend/services/gnn_xai_service` install CPU-only PyTorch (`pytorch-cpu` index in their `pyproject.toml`), both in Docker and on the host. This keeps the images small, and inference only handles one molecule at a time.
- **Ingestion:** the root `pyproject.toml` installs the CUDA 13.0 build (`pytorch-cu130` via `tool.uv.sources`) so Uni-Mol embeddings can run on an NVIDIA GPU. Change that index if your machine has no compatible GPU.

### 4. Empty LLM Interpretation
- **Symptom:** the prediction succeeds, but the scientific interpretation card is empty.
- **Cause:** a reasoning model used up the token budget (`llm_max_tokens`) on its reasoning, or the OpenRouter API key is missing or over quota.
- **Solution:** check `OPENROUTER_API_KEY` in `.env`, or try another `LLM_MODEL`. A missing interpretation or a failed ChEMBL lookup does not fail the request: predictions, 3D attributions, and similar compounds are still returned.
