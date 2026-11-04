from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gnn_service_url: str = "http://localhost:8000/gnn-xai"
    database_url: str
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parents[4] / ".env", env_file_encoding="utf-8"
    )


settings = Settings()
