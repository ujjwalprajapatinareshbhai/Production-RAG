from app.bm25_retriever import refresh_bm25_index
from app.registry import (
    delete_document_record,
    get_document,
    update_document_status,
)
from app.vector_store import delete_document_chunks


def delete_document(document_id: str) -> dict:
    """Delete a document from Qdrant, PostgreSQL, and the BM25 index."""

    if len(document_id) != 64:
        raise ValueError(
            "document_id must be a 64-character SHA-256 hash."
        )

    document = get_document(document_id)

    if document is None:
        raise ValueError("Document not found.")

    # Hide the document from retrieval before deleting its vectors.
    # If this fails, do not continue with destructive operations.
    update_document_status(
        document_id,
        status="failed",
        error_message="Document deletion in progress.",
    )

    # Delete vectors. If this fails, the registry record remains
    # available for investigation and a later retry.
    delete_document_chunks(document_id)

    # Remove the registry record only after Qdrant deletion succeeds.
    deleted = delete_document_record(document_id)

    if not deleted:
        raise RuntimeError(
            "Qdrant cleanup succeeded, but the registry record "
            "could not be deleted. The document remains marked failed."
        )

    # Rebuild the keyword index without the deleted document.
    try:
        refresh_bm25_index()
    except Exception as exc:
        raise RuntimeError(
            "The document was deleted from Qdrant and PostgreSQL, "
            "but the BM25 index refresh failed."
        ) from exc

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "status": "deleted",
    }