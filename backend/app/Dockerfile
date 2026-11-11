FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# src/core/config.py reads parents[4] as the repo root, so the image mirrors the repo layout
WORKDIR /code/backend/app

# dependencies first: this layer is rebuilt only when the lock changes, not on every edit
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

ENV PATH="/code/backend/app/.venv/bin:$PATH"
EXPOSE 8001
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
