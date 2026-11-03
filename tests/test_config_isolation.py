from app.config import settings
from app.vector_store import COLLECTION_NAME


def test_test_environment_is_isolated():
    assert settings.postgres_db == "production_rag_test"
    assert settings.qdrant_collection == "document_chunks_test"
    assert COLLECTION_NAME == "document_chunks_test"
