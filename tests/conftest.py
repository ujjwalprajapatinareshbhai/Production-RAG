import os


def pytest_configure():
    """Use isolated PostgreSQL and Qdrant resources for pytest."""
    os.environ["POSTGRES_DB"] = "production_rag_test"
    os.environ["QDRANT_COLLECTION_NAME"] = "document_chunks_test"