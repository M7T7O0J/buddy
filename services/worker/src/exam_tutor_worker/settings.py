from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str
    REDIS_URL: str

    EMBEDDINGS_BASE_URL: str
    EMBEDDINGS_MODEL: str = "BAAI/bge-m3"
    EMBEDDINGS_TIMEOUT_S: float = 60.0

    TOKENIZER_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"
    CHUNK_MIN_TOKENS: int = 500
    CHUNK_MAX_TOKENS: int = 900
    CHUNK_OVERLAP_TOKENS: int = 100
    CHUNK_PARENT_SECTION_LEVEL: int = 2
    CHUNK_FILTER_ENABLED: bool = True
    CHUNK_FILTER_MIN_TOKENS: int = 40
    CHUNK_FILTER_MAX_CHUNKS_PER_DOC: int = 2000
    CHUNK_FILTER_MAX_CHUNKS_PER_PARENT: int = 400

settings = Settings()
