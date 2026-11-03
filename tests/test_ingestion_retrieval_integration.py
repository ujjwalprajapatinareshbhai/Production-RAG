from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np
import pymupdf
import pytest
from qdrant_client import models

import app.ingestion as ingestion
import app.vector_store as vector_store

from app.db import get_connection
from app.ingestion import ingest_document
from app.pdf_loader import calculate_file_hash
from app.qdrant_hybrid_retriever import search_qdrant_hybrid
from app.registry import get_document, update_document_status
from app.retriever import search_chunks
from app.vector_store import COLLECTION_NAME, get_vector_client


@pytest.fixture
def test_tenant_id():
    """Create an isolated tenant in the test PostgreSQL database."""

    tenant_id = uuid4()
    tenant_slug = f"integration-{tenant_id.hex}"

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO tenants (tenant_id, name, slug)
            VALUES (%s, %s, %s)
            """,
            (
                tenant_id,
                "Integration Test Tenant",
                tenant_slug,
            ),
        )

    yield tenant_id

    # Each test removes its own documents and Qdrant points first.
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM tenants WHERE tenant_id = %s",
            (tenant_id,),
        )


def _document_filter(tenant_id: UUID, document_id: str):
    """Build a Qdrant filter scoped to one tenant and document."""

    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(
                    value=str(tenant_id),
                ),
            ),
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(
                    value=document_id,
                ),
            ),
        ]
    )


def _cleanup_test_document(
    client,
    tenant_id: UUID,
    document_id: str | None,
):
    """Remove only the current test tenant's document and vectors."""

    if document_id is None:
        return

    if client.collection_exists(COLLECTION_NAME):
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=_document_filter(
                    tenant_id,
                    document_id,
                )
            ),
            wait=True,
        )

    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM documents
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (tenant_id, document_id),
        )


def test_real_ingestion_and_retrieval(
    tmp_path,
    test_tenant_id,
):
    """Test real PDF ingestion, retrieval, status filtering, and cleanup."""

    tenant_id = test_tenant_id
    pdf_path = tmp_path / "integration_quartz_falcon.pdf"
    document_id = None
    client = get_vector_client()

    test_text = """
    Quartz Falcon inventory calibration report.

    The Quartz Falcon team tracks warehouse inventory using calibration
    records. Each inventory item receives a quality inspection before
    shipment. Calibration measurements are recorded alongside the
    inspection date and the responsible technician.

    The inventory report recommends checking calibration records before
    approving warehouse shipments. Missing measurements must be reviewed
    before an inventory item can be marked as ready for dispatch.
    """

    try:
        # 1. Create a small temporary PDF with unique test content.
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_textbox(
                (72, 72, 540, 740),
                test_text,
                fontsize=11,
            )
            pdf.save(str(pdf_path))

        document_id = calculate_file_hash(pdf_path)

        # 2. Run the real tenant-aware ingestion pipeline.
        result = ingest_document(
            pdf_path,
            tenant_id=tenant_id,
            chunk_size=100,
            overlap=10,
        )

        assert result["tenant_id"] == str(tenant_id)
        assert result["document_id"] == document_id
        assert result["status"] == "completed"
        assert result["chunks_stored"] == result["chunks_prepared"]
        assert result["chunks_stored"] > 0

        # 3. Verify the real PostgreSQL registry status.
        document = get_document(
            tenant_id,
            document_id,
        )

        assert document is not None
        assert document["status"] == "completed"
        assert document["tenant_id"] == tenant_id

        # 4. Verify real tenant-scoped dense retrieval.
        query = "How are Quartz Falcon inventory items calibrated?"

        dense_results = search_chunks(
            tenant_id=tenant_id,
            query=query,
            limit=3,
            document_id=document_id,
        )

        assert dense_results
        assert all(
            item["document_id"] == document_id
            for item in dense_results
        )
        assert all(
            item["tenant_id"] == str(tenant_id)
            for item in dense_results
        )

        # 5. Verify real tenant-scoped hybrid retrieval.
        hybrid_results = search_qdrant_hybrid(
            tenant_id=tenant_id,
            query=query,
            limit=3,
            candidate_limit=10,
            document_id=document_id,
        )

        assert hybrid_results
        assert all(
            item["document_id"] == document_id
            for item in hybrid_results
        )
        assert all(
            item["tenant_id"] == str(tenant_id)
            for item in hybrid_results
        )

        # 6. A failed document must no longer be searchable.
        update_document_status(
            tenant_id,
            document_id,
            "failed",
            error_message="Integration test status-filter check",
        )

        assert search_chunks(
            tenant_id=tenant_id,
            query=query,
            limit=3,
            document_id=document_id,
        ) == []

        assert search_qdrant_hybrid(
            tenant_id=tenant_id,
            query=query,
            limit=3,
            candidate_limit=10,
            document_id=document_id,
        ) == []

    finally:
        # 7. Delete only this test tenant's document and vectors.
        _cleanup_test_document(
            client,
            tenant_id,
            document_id,
        )


def test_partial_ingestion_failure_is_not_searchable(
    tmp_path,
    monkeypatch,
    test_tenant_id,
):
    """A partial ingestion failure must be cleaned up and unsearchable."""

    tenant_id = test_tenant_id
    pdf_path = tmp_path / "partial_failure_test.pdf"
    document_id = None
    client = get_vector_client()

    test_text = " ".join(
        [
            (
                "Quartz Falcon warehouse inventory calibration records "
                "require quality inspection before shipment."
            )
            for _ in range(20)
        ]
    )

    try:
        # 1. Create a temporary PDF with enough text for multiple chunks.
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_textbox(
                (72, 72, 540, 740),
                test_text,
                fontsize=11,
            )
            pdf.save(str(pdf_path))

        document_id = calculate_file_hash(pdf_path)

        # 2. Replace embedding calls with deterministic test vectors.
        monkeypatch.setattr(
            vector_store,
            "embed_texts",
            lambda texts: [
                [1.0] + [0.0] * 383
                for _ in texts
            ],
        )

        def fake_sparse_embeddings(texts):
            return iter(
                [
                    SimpleNamespace(
                        indices=np.array([1]),
                        values=np.array([0.5]),
                    )
                    for _ in texts
                ]
            )

        monkeypatch.setattr(
            vector_store._sparse_model,
            "embed",
            fake_sparse_embeddings,
        )

        # 3. Fail the second Qdrant batch after the first batch persists.
        original_upsert = client.upsert
        upsert_calls = 0

        def fail_on_second_upsert(*args, **kwargs):
            nonlocal upsert_calls
            upsert_calls += 1

            if upsert_calls == 2:
                raise RuntimeError(
                    "Simulated partial Qdrant failure"
                )

            return original_upsert(*args, **kwargs)

        monkeypatch.setattr(
            client,
            "upsert",
            fail_on_second_upsert,
        )

        # Force one chunk per batch so the second upsert fails.
        monkeypatch.setattr(
            ingestion,
            "upsert_chunks",
            lambda chunks: vector_store.upsert_chunks(
                chunks,
                batch_size=1,
            ),
        )

        # 4. Ingestion should raise an error and mark the document failed.
        with pytest.raises(
            RuntimeError,
            match="Simulated partial Qdrant failure",
        ):
            ingest_document(
                pdf_path,
                tenant_id=tenant_id,
                chunk_size=20,
                overlap=0,
            )

        document = get_document(
            tenant_id,
            document_id,
        )

        assert document is not None
        assert document["status"] == "failed"
        assert upsert_calls == 2

        # 5. Confirm cleanup removed the partially stored Qdrant points.
        stored_points = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=_document_filter(
                tenant_id,
                document_id,
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )[0]

        assert len(stored_points) == 0

        # 6. Confirm both retrievers return no results for this document.
        query = "Quartz Falcon inventory calibration"

        assert search_chunks(
            tenant_id=tenant_id,
            query=query,
            limit=3,
            document_id=document_id,
        ) == []

        assert search_qdrant_hybrid(
            tenant_id=tenant_id,
            query=query,
            limit=3,
            candidate_limit=10,
            document_id=document_id,
        ) == []

    finally:
        # Delete only this temporary test tenant's vectors and registry row.
        _cleanup_test_document(
            client,
            tenant_id,
            document_id,
        )


def test_ingesting_same_pdf_twice_does_not_duplicate_data(
    tmp_path,
    test_tenant_id,
):
    """Ingesting identical PDF bytes twice should not duplicate records."""

    tenant_id = test_tenant_id
    pdf_path = tmp_path / "duplicate_ingestion_test.pdf"
    document_id = None
    client = get_vector_client()

    test_text = """
    Duplicate ingestion verification report.

    The Quartz Falcon team checks inventory calibration records,
    quality inspections, and warehouse shipment readiness.
    Reprocessing the same document should update existing vector points
    rather than create duplicate points.
    """

    try:
        # 1. Create a temporary PDF.
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_textbox(
                (72, 72, 540, 740),
                test_text,
                fontsize=11,
            )
            pdf.save(str(pdf_path))

        document_id = calculate_file_hash(pdf_path)

        # 2. Ingest the same PDF for the first time.
        first_result = ingest_document(
            pdf_path,
            tenant_id=tenant_id,
            chunk_size=100,
            overlap=10,
        )

        assert first_result["tenant_id"] == str(tenant_id)
        assert first_result["status"] == "completed"
        assert first_result["document_id"] == document_id

        # Count points belonging only to this tenant and document.
        first_count = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=_document_filter(
                tenant_id,
                document_id,
            ),
            exact=True,
        ).count

        assert first_count == first_result["chunks_stored"]

        # 3. Ingest the exact same PDF again.
        second_result = ingest_document(
            pdf_path,
            tenant_id=tenant_id,
            chunk_size=100,
            overlap=10,
        )

        assert second_result["tenant_id"] == str(tenant_id)
        assert second_result["status"] == "completed"
        assert second_result["document_id"] == document_id
        assert second_result["chunks_stored"] == first_result["chunks_stored"]

        # 4. Verify the second ingestion did not increase the point count.
        second_count = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=_document_filter(
                tenant_id,
                document_id,
            ),
            exact=True,
        ).count

        assert second_count == first_count

        # 5. Verify PostgreSQL has only one registry record for this tenant.
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM documents
                WHERE tenant_id = %s
                  AND document_id = %s
                """,
                (tenant_id, document_id),
            ).fetchone()

        assert row[0] == 1

        # The document should remain completed after the retry.
        document = get_document(
            tenant_id,
            document_id,
        )

        assert document["status"] == "completed"

    finally:
        # Remove only this test tenant's vectors and registry row.
        _cleanup_test_document(
            client,
            tenant_id,
            document_id,
        )