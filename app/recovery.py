from app.bm25_retriever import refresh_bm25_index
from app.registry import (
    claim_pending_cleanup_documents,
    mark_document_cleanup_completed,
    mark_document_cleanup_failed,
    mark_stale_documents_failed,
)
from app.vector_store import delete_document_chunks


def recover_stale_documents(
    stale_after_minutes: int = 30,
    batch_size: int = 10,
    lease_minutes: int = 15,
) -> dict:
    """
    Detect stale ingestion records and clean up their Qdrant vectors.

    Processes at most one batch per invocation. Failed cleanup tasks
    remain pending and can be retried by the next invocation.

    Each claimed task has a UUID fencing token. Only the current
    lease owner can update the cleanup status.
    """

    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be at least 1.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if lease_minutes < 1:
        raise ValueError("lease_minutes must be at least 1.")

    # Step 1: Mark stale ingestion records as failed.
    # Their vector cleanup is added to the pending cleanup queue.
    stale_documents = mark_stale_documents_failed(
        stale_after_minutes=stale_after_minutes,
    )

    # Step 2: Atomically claim a batch of pending cleanup tasks.
    # Each claimed document receives a new UUID fencing token.
    claimed_documents = claim_pending_cleanup_documents(
        limit=batch_size,
        lease_minutes=lease_minutes,
    )

    cleaned_documents = []
    failed_documents = []

    # Step 3: Process each claimed cleanup task.
    for document in claimed_documents:
        document_id = document["document_id"]
        claim_token = document["cleanup_claim_token"]

        # Step 3A: Delete vectors from Qdrant.
        try:
            delete_document_chunks(document_id)

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

            # Record failure only if this worker still owns
            # the cleanup lease.
            updated_document = mark_document_cleanup_failed(
                document_id,
                error_message,
                claim_token,
            )

            if updated_document is None:
                raise RuntimeError(
                    "Cleanup failed, but the registry rejected "
                    "the failure update because the document "
                    "disappeared or this worker no longer owns "
                    f"the cleanup lease: {document_id}"
                ) from exc

            failed_documents.append(
                {
                    "document_id": document_id,
                    "error": error_message,
                }
            )

            # Continue with the next claimed document.
            continue

        # Step 3B: Qdrant deletion succeeded.
        # Mark cleanup completed only if this worker still
        # owns the cleanup lease.
        updated_document = mark_document_cleanup_completed(
            document_id,
            claim_token,
        )

        if updated_document is None:
            raise RuntimeError(
                "Cleanup completion rejected because the "
                "document disappeared or this worker no longer "
                f"owns the cleanup lease: {document_id}"
            )

        cleaned_documents.append(document_id)

    # Step 4: Refresh BM25 if document visibility changed or
    # at least one Qdrant cleanup succeeded.
    bm25_refresh_error = None

    if stale_documents or cleaned_documents:
        try:
            refresh_bm25_index()

        except Exception as exc:
            bm25_refresh_error = f"{type(exc).__name__}: {exc}"

    return {
        "stale_documents_marked": len(stale_documents),
        "cleanup_tasks_claimed": len(claimed_documents),
        "cleanup_completed": len(cleaned_documents),
        "cleanup_failed": len(failed_documents),
        "cleaned_document_ids": cleaned_documents,
        "failed_documents": failed_documents,
        "bm25_refreshed": (
            bm25_refresh_error is None
            and bool(stale_documents or cleaned_documents)
        ),
        "bm25_refresh_error": bm25_refresh_error,
    }