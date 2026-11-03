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

    # Find the document inside the authenticated tenant.
    document = get_document(tenant_id, document_id)

    if document is None:
        raise ValueError("Document not found.")

    # Hide the document from retrieval before deleting its vectors.
    # If this fails, do not continue with destructive operations.
    update_document_status(
        tenant_id,
        document_id,
        status="failed",
        error_message="Document deletion in progress.",
    )

    # Delete only this tenant's vectors.
    # If this fails, the registry record remains available
    # for investigation and a later retry.
    delete_document_chunks(tenant_id, document_id)

    # Remove the registry record only after Qdrant deletion succeeds.
    deleted = delete_document_record(tenant_id, document_id)

    if not deleted:
        raise RuntimeError(
            "Qdrant cleanup succeeded, but the registry record "
            "could not be deleted. The document remains marked failed."
        )

    # Refresh only this tenant's keyword index.
    try:
        refresh_bm25_index(tenant_id)
    except Exception as exc:
        raise RuntimeError(
            "The document was deleted from Qdrant and PostgreSQL, "
            "but the tenant's BM25 index refresh failed."
        ) from exc

    return {
        "tenant_id": str(tenant_id),
        "document_id": document_id,
        "filename": document["filename"],
        "status": "deleted",
    }