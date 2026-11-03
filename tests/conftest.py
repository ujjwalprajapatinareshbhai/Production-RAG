import os

# Configure isolated test resources before importing application modules.
os.environ["POSTGRES_DB"] = "production_rag_test"
os.environ["QDRANT_COLLECTION"] = "document_chunks_test"