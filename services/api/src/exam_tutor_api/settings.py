from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:3000"

    DATABASE_URL: str
    REDIS_URL: str

    EMBEDDINGS_BASE_URL: str
    EMBEDDINGS_MODEL: str = "BAAI/bge-m3"
    EMBEDDINGS_TIMEOUT_S: float = 60.0

    RETRIEVE_TOP_K: int = 20
    RETRIEVE_TOP_N: int = 8
    RETRIEVE_MIN_SCORE: float = 0.15
    # HNSW ef_search controls recall vs latency (higher = better recall, slower).
    # Set <=0 to disable query-time tuning.
    RETRIEVE_HNSW_EF_SEARCH: int = 0
    # Comma-separated tags to exclude at retrieval time (stored in chunks.metadata.tags).
    RETRIEVE_EXCLUDE_TAGS: str = "front_matter,boilerplate,image_only,duplicate"
    # Drop very low-quality chunks computed during ingestion (stored in chunks.metadata.quality_score).
    # Set <0 to disable.
    RETRIEVE_MIN_QUALITY_SCORE: float = -1.0

    # Optional cross-encoder reranking for better relevance.
    RERANK_ENABLED: bool = False
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_TOP_M: int = 30
    RERANK_BATCH_SIZE: int = 16

    # Cap how much retrieved context is stuffed into the LLM prompt.
    # This is essential for smaller-context local vLLM configs (e.g. 2k tokens).
    PROMPT_SOURCES_MAX_CHUNKS: int = 4
    PROMPT_SOURCES_MAX_TOKENS: int = 1200

    LLM_MODE: str = "mock"   # mock | vllm
    LLM_BASE_URL: str = "http://localhost:8002/v1"
    LLM_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
    LLM_API_KEY: str = "EMPTY"
    LLM_TIMEOUT_S: float = 120.0

    DEFAULT_EXAM: str = "GATE_DA"
    DEFAULT_MODE: str = "doubt"
    DEFAULT_LANGUAGE: str = "en"

    LOCAL_DOC_ROOT: str = "/app/data"

settings = Settings()
