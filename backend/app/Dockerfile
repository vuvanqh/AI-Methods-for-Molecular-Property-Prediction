FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# src/core/config.py reads parents[4] as the repo root, so the image mirrors the repo layout
WORKDIR /code/backend/app

# rdkit's drawing module, imported by unimol_tools, links against these X11 libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libxrender1 libxext6 libsm6 libglib2.0-0 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# dependencies first: this layer is rebuilt only when the lock changes, not on every edit
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

ENV PATH="/code/backend/app/.venv/bin:$PATH"
EXPOSE 8001
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]