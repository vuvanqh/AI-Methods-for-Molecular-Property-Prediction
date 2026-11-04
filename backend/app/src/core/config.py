from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gnn_service_url: str = "http://localhost:8000/gnn-xai"
    # inference + integrated gradients can exceed httpx's 5 s default, which surfaced as a false "unavailable" 503
    gnn_timeout_seconds: float = 60.0
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()



