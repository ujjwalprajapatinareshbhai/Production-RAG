from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for the Production-RAG application."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "document_chunks"

    # Retrieval models
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    vector_size: int = 384

    # Ollama answer generation
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:cloud"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str
    postgres_user: str
    postgres_password: str

    # JWT authentication
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "production-rag"
    jwt_audience: str = "production-rag-api"
    access_token_expire_minutes: int = 15


settings = Settings()