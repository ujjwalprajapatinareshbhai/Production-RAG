import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from uuid import UUID

from qdrant_client import models

from app.config import settings
from app.ingestion import prepare_document
from app.registry import list_documents
from app.vector_store import get_vector_client


PDF_PATH = "data/pdfs/Testing-in-a-Small-Company1.pdf"

TENANT_ID = UUID(
    "feeb3d7d-f2f4-4f10-9ab2-b59c6bb28358"
)


def main():
    # Prepare chunks in memory. This does not write to Qdrant.
    chunks = prepare_document(
        PDF_PATH,
        tenant_id=TENANT_ID,
    )

    document_id = chunks[0].document_id

    print("\n========== PREPARED DOCUMENT ==========")
    print("Tenant ID:", TENANT_ID)
    print("Document ID:", document_id)
    print("Filename:", chunks[0].filename)
    print("Prepared chunks:", len(chunks))

    abstract_chunks = [
        chunk for chunk in chunks
        if chunk.section_type == "abstract"
    ]

    print("Abstract chunks:", len(abstract_chunks))

    # Read the existing PostgreSQL document record.
    print("\n========== POSTGRESQL RECORD ==========")

    documents = list_documents(TENANT_ID)

    matching_documents = [
        document for document in documents
        if document["document_id"] == document_id
    ]

    for document in matching_documents:
        print(document)

    if not matching_documents:
        print("No matching PostgreSQL document record found.")

    # Count only points belonging to this tenant and document.
    print("\n========== QDRANT RECORD ==========")

    client = get_vector_client()

    document_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=str(TENANT_ID)),
            ),
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            ),
        ]
    )

    result = client.count(
        collection_name=settings.qdrant_collection,
        count_filter=document_filter,
        exact=True,
    )

    print("Collection:", settings.qdrant_collection)
    print("Existing Qdrant points:", result.count)

    # Inspect existing section types.
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=document_filter,
        limit=200,
        with_payload=["chunk_index", "section_type", "source_pages"],
        with_vectors=False,
    )

    section_counts = {}

    for point in points:
        section = point.payload.get("section_type", "unknown")
        section_counts[section] = section_counts.get(section, 0) + 1

    print("Existing section counts:", section_counts)


if __name__ == "__main__":
    main()