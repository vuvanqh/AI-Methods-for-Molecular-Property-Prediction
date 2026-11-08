from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT_ENV = Path(__file__).resolve().parents[4] / ".env"


class Settings(BaseSettings):
    database_url: str
    gnn_service_url: str = "http://localhost:8000/gnn-xai"
    llm_service_url: str = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY: str = ""
    # inference + integrated gradients can exceed httpx's 5 s default, which surfaced as a false "unavailable" 503
    llm_temp: float = 0.1
    llm_max_tokens: int = 1000
    llm_model: str = "choosemodel"
    llm_timeout_seconds: float = 300

    gnn_timeout_seconds: float = 60.0
    # repo-root .env works from any working directory; a .env in the working directory still overrides it
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT_ENV, ".env"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
