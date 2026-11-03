import logging
import threading
from pathlib import Path
from uuid import UUID

from app.bm25_retriever import refresh_bm25_index
from app.chunker import chunk_pages
from app.pdf_loader import load_pdf
from app.registry import (
    register_document,
    update_document_status,
    update_document_heartbeat,
)
from app.vector_store import delete_document_chunks, upsert_chunks


logger = logging.getLogger(__name__)

# Refresh the heartbeat every 30 seconds during ingestion.
HEARTBEAT_INTERVAL_SECONDS = 30


def prepare_document(
    pdf_path: str | Path,
    tenant_id: UUID,
    chunk_size: int = 400,
    overlap: int = 60,
):
    """Load a PDF and prepare tenant-scoped text chunks without storing vectors."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    pdf_path = Path(pdf_path)

    pages = load_pdf(pdf_path)

    if not pages:
        raise ValueError(f"No pages were extracted from {pdf_path.name}")

    chunks = chunk_pages(
        pages,
        tenant_id=tenant_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError(
            f"No text chunks were produced for {pdf_path.name}"
        )

    return chunks


def _heartbeat_worker(
    tenant_id: UUID,
    document_id: str,
    stop_event: threading.Event,
    heartbeat_error: list[Exception],
) -> None:
    """Refresh the heartbeat periodically until ingestion finishes."""

    while not stop_event.wait(HEARTBEAT_INTERVAL_SECONDS):
        try:
            updated = update_document_heartbeat(
                tenant_id,
                document_id,
            )

            if not updated:
                raise RuntimeError(
                    "Heartbeat update returned False for document "
                    f"{document_id}."
                )

        except Exception as exc:
            heartbeat_error.append(exc)

            logger.exception(
                "Heartbeat failed for tenant %s, document %s",
                tenant_id,
                document_id,
            )

            stop_event.set()
            return


def ingest_document(
    pdf_path: str | Path,
    tenant_id: UUID,
    chunk_size: int = 400,
    overlap: int = 60,
) -> dict:
    """Prepare a PDF, store its tenant-scoped vectors, and track ingestion status."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    pdf_path = Path(pdf_path)

    # Stage 1: Prepare the document with tenant ownership.
    chunks = prepare_document(
        pdf_path,
        tenant_id=tenant_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    document_id = chunks[0].document_id
    filename = chunks[0].filename

    # Stage 2: Register the document under the correct tenant.
    register_document(
        tenant_id,
        document_id,
        filename,
    )

    update_document_status(
        tenant_id,
        document_id,
        "processing",
    )

    stop_event = threading.Event()
    heartbeat_error: list[Exception] = []
    heartbeat_thread = None

    try:
        # Stage 3: Validate that the heartbeat works before storage.
        if not update_document_heartbeat(
            tenant_id,
            document_id,
        ):
            raise RuntimeError(
                "Could not refresh heartbeat for processing document "
                f"{document_id}."
            )

        # Stage 4: Start periodic heartbeat updates.
        heartbeat_thread = threading.Thread(
            target=_heartbeat_worker,
            args=(
                tenant_id,
                document_id,
                stop_event,
                heartbeat_error,
            ),
            name=f"heartbeat-{document_id[:8]}",
            daemon=True,
        )

        heartbeat_thread.start()

        # Stage 5: Store tenant-scoped chunks in Qdrant.
        stored_count = upsert_chunks(chunks)

        if stored_count != len(chunks):
            raise RuntimeError(
                f"Expected to store {len(chunks)} chunks, "
                f"but stored {stored_count}."
            )

        # Stage 6: Stop the heartbeat and wait for its thread to finish.
        stop_event.set()
        heartbeat_thread.join()

        # A background heartbeat failure must not be ignored.
        if heartbeat_error:
            raise RuntimeError(
                f"Periodic heartbeat failed for document {document_id}."
            ) from heartbeat_error[0]

        # Stage 7: Mark ingestion as successful.
        update_document_status(
            tenant_id,
            document_id,
            "completed",
        )

        # Refresh only this tenant's BM25 index.
        # A BM25 refresh failure should not undo successful ingestion.
        try:
            refresh_bm25_index(tenant_id)
        except Exception:
            logger.exception(
                "BM25 index refresh failed for tenant %s",
                tenant_id,
            )

        return {
            "tenant_id": str(tenant_id),
            "document_id": document_id,
            "filename": filename,
            "chunks_prepared": len(chunks),
            "chunks_stored": stored_count,
            "status": "completed",
        }

    except Exception as exc:
        # Stop the background worker before failure handling.
        stop_event.set()

        if heartbeat_thread is not None:
            heartbeat_thread.join()

        # Preserve the original error if the heartbeat also failed.
        if heartbeat_error and heartbeat_error[0] is not exc:
            exc.add_note(
                "Additionally, periodic heartbeat failed: "
                f"{heartbeat_error[0]}"
            )

        # Stage 8: Mark this tenant's document as failed.
        try:
            update_document_status(
                tenant_id,
                document_id,
                "failed",
                error_message=str(exc),
            )

        except Exception as status_exc:
            exc.add_note(
                "Additionally, failed to update the document "
                f"status to 'failed': {status_exc}"
            )

        # Stage 9: Delete only this tenant's partially stored vectors.
        try:
            delete_document_chunks(
                tenant_id,
                document_id,
            )

        except Exception as cleanup_exc:
            exc.add_note(
                "Additionally, failed to clean up partially stored "
                f"Qdrant vectors: {cleanup_exc}"
            )

        raise