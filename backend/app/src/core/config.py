from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gnn_service_url: str = "http://localhost:8000/gnn-xai"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()



