MORGAN_FINGERPRINT_SIZE = 2048
MORGAN_FINGERPRINT_RADIUS = 2
RAG_TOP_K = 5
# molecule_sources stores the provider's identifier; the link the frontend shows is built from it
SOURCE_URL_TEMPLATES = {"ChEMBL": "https://www.ebi.ac.uk/chembl/explore/compound/{}"}

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT_ENV = Path(__file__).resolve().parents[4] / ".env"


class UniMolSettings(BaseSettings):
    model_name: str = "unimolv1"
    batch_size: int = 32
    remove_hydrogens: bool = False
    idle_unload_seconds: float = 600
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT_ENV, ".env"), env_file_encoding="utf-8", extra="ignore"
    )


class RetrievalSettings(BaseSettings):
    fingerprint_weight: float = 0.5
    semantic_weight: float = 0.5
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT_ENV, ".env"), env_file_encoding="utf-8", extra="ignore"
    )


unimolSettings = UniMolSettings()
retrievalSettings = RetrievalSettings()
