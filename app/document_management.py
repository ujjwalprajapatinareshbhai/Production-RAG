from uuid import UUID

from app.bm25_retriever import refresh_bm25_index
from app.registry import (
    delete_document_record,
    get_document,
    update_document_status,
)
from app.vector_store import delete_document_chunks


def delete_document(
    tenant_id: UUID,
    document_id: str,
) -> dict:
    """Delete a document from Qdrant, PostgreSQL, and its tenant's BM25 index."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    if not document_id or len(document_id) != 64:
        raise ValueError(
            "document_id must be a 64-character SHA-256 hash."
        )

    # 1. Find the document inside the authenticated tenant.
    document = get_document(tenant_id, document_id)

    if document is None:
        raise ValueError("Document not found.")

    # 2. Mark the document as failed before deleting its vectors.
    updated_document = update_document_status(
        tenant_id,
        document_id,
        status="failed",
        error_message="Document deletion in progress.",
    )

    if updated_document is None:
        raise ValueError("Document not found.")

    # 3. Delete only this tenant's vectors from Qdrant.
    delete_document_chunks(tenant_id, document_id)

    # 4. Delete the PostgreSQL registry record.
    deleted = delete_document_record(tenant_id, document_id)

    if not deleted:
        raise RuntimeError(
            "Qdrant cleanup succeeded, but the registry record "
            "could not be deleted. The document remains marked failed."
        )

    # 5. Refresh only this tenant's BM25 index.
    try:
        refresh_bm25_index(tenant_id)
    except Exception as exc:
        raise RuntimeError(
            "The document was deleted from Qdrant and PostgreSQL, "
            "but the tenant's BM25 index refresh failed."
        ) from exc

    # 6. Return deletion details.
    return {
        "tenant_id": str(tenant_id),
        "document_id": document_id,
        "filename": document["filename"],
        "status": "deleted",
    }