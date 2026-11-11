FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /code/backend/services/gnn_xai_service

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libxrender1 libxext6 libsm6 libglib2.0-0 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# dependencies first: torch and chemprop are slow to install, so keep that layer cached
COPY pyproject.toml uv.lock ./
# the cache mount keeps torch downloaded across rebuilds, which otherwise take minutes
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src

ENV PATH="/code/backend/services/gnn_xai_service/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
