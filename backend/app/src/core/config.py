from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT_ENV = Path(__file__).resolve().parents[4] / ".env"


class Settings(BaseSettings):
    gnn_service_url: str = "http://localhost:8000/gnn-xai"
    database_url: str
    # inference + integrated gradients can exceed httpx's 5 s default, which surfaced as a false "unavailable" 503
    gnn_timeout_seconds: float = 60.0
    # repo-root .env works from any working directory; a .env in the working directory still overrides it
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT_ENV, ".env"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
