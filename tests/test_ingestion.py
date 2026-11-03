import pytest
from uuid import UUID

from app.models import Chunk
import app.ingestion as ingestion


TEST_TENANT_ID = UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture(autouse=True)
def mock_vector_cleanup(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "delete_document_chunks",
        lambda tenant_id, document_id: None,
    )

    # Keep heartbeat successful by default in ingestion unit tests.
    monkeypatch.setattr(
        ingestion,
        "update_document_heartbeat",
        lambda tenant_id, document_id: True,
    )


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chunk-1",
            tenant_id=TEST_TENANT_ID,
            document_id="a" * 64,
            filename="sample.pdf",
            chunk_index=0,
            text="This is the first test chunk.",
            start_page=1,
            end_page=1,
            source_pages=[1],
        ),
        Chunk(
            chunk_id="chunk-2",
            tenant_id=TEST_TENANT_ID,
            document_id="a" * 64,
            filename="sample.pdf",
            chunk_index=1,
            text="This is the second test chunk.",
            start_page=1,
            end_page=2,
            source_pages=[1, 2],
        ),
    ]


def test_ingestion_success(monkeypatch, sample_chunks):
    status_updates = []

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )
    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        lambda tenant_id, doc_id, status, **kwargs: status_updates.append(
            (status, kwargs.get("error_message"))
        ),
    )
    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        lambda chunks: len(chunks),
    )

    result = ingestion.ingest_document(
        "sample.pdf",
        tenant_id=TEST_TENANT_ID,
    )

    assert result["tenant_id"] == str(TEST_TENANT_ID)
    assert result["status"] == "completed"
    assert result["chunks_prepared"] == 2
    assert result["chunks_stored"] == 2

    assert status_updates == [
        ("processing", None),
        ("completed", None),
    ]


def test_ingestion_failure_marks_document_failed(
    monkeypatch, sample_chunks
):
    status_updates = []

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )
    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        lambda tenant_id, doc_id, status, **kwargs: status_updates.append(
            (status, kwargs.get("error_message"))
        ),
    )

    def fail_storage(chunks):
        raise RuntimeError("Qdrant storage failed")

    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        fail_storage,
    )

    with pytest.raises(
        RuntimeError,
        match="Qdrant storage failed",
    ):
        ingestion.ingest_document(
            "sample.pdf",
            tenant_id=TEST_TENANT_ID,
        )

    assert status_updates[0] == ("processing", None)
    assert status_updates[1] == (
        "failed",
        "Qdrant storage failed",
    )


def test_ingestion_detects_chunk_count_mismatch(
    monkeypatch, sample_chunks
):
    status_updates = []

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )
    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        lambda tenant_id, doc_id, status, **kwargs: status_updates.append(
            (status, kwargs.get("error_message"))
        ),
    )
    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        lambda chunks: 1,
    )

    with pytest.raises(
        RuntimeError,
        match="Expected to store 2 chunks",
    ):
        ingestion.ingest_document(
            "sample.pdf",
            tenant_id=TEST_TENANT_ID,
        )

    assert status_updates[-1][0] == "failed"
    assert "Expected to store 2 chunks" in status_updates[-1][1]


def test_ingestion_preserves_original_error_if_failure_update_fails(
    monkeypatch, sample_chunks
):
    def update_status(tenant_id, doc_id, status, **kwargs):
        if status == "completed":
            raise RuntimeError(
                "Could not mark document completed"
            )

        if status == "failed":
            raise RuntimeError(
                "Could not mark document failed"
            )

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )
    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        update_status,
    )
    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        lambda chunks: len(chunks),
    )

    with pytest.raises(
        RuntimeError,
        match="Could not mark document completed",
    ):
        ingestion.ingest_document(
            "sample.pdf",
            tenant_id=TEST_TENANT_ID,
        )


def test_ingestion_stops_if_processing_status_update_fails(
    monkeypatch, sample_chunks
):
    status_updates = []
    storage_calls = []

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )

    def fail_processing_update(
        tenant_id, doc_id, status, **kwargs
    ):
        status_updates.append(status)

        if status == "processing":
            raise RuntimeError("PostgreSQL unavailable")

    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        fail_processing_update,
    )
    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        lambda chunks: storage_calls.append(chunks),
    )

    with pytest.raises(
        RuntimeError,
        match="PostgreSQL unavailable",
    ):
        ingestion.ingest_document(
            "sample.pdf",
            tenant_id=TEST_TENANT_ID,
        )

    assert status_updates == ["processing"]
    assert storage_calls == []


def test_ingestion_recovers_after_retry(monkeypatch, sample_chunks):
    status_updates = []
    storage_attempts = []

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )

    def update_status(tenant_id, doc_id, status, **kwargs):
        status_updates.append(
            (tenant_id, doc_id, status, kwargs.get("error_message"))
        )

    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        update_status,
    )

    def storage_with_first_attempt_failure(chunks):
        storage_attempts.append(chunks)

        if len(storage_attempts) == 1:
            raise RuntimeError("Temporary Qdrant failure")

        return len(chunks)

    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        storage_with_first_attempt_failure,
    )

    # Attempt 1: storage fails.
    with pytest.raises(
        RuntimeError,
        match="Temporary Qdrant failure",
    ):
        ingestion.ingest_document(
            "sample.pdf",
            tenant_id=TEST_TENANT_ID,
        )

    # Attempt 2: retry succeeds.
    result = ingestion.ingest_document(
        "sample.pdf",
        tenant_id=TEST_TENANT_ID,
    )

    expected_document_id = sample_chunks[0].document_id

    assert result["tenant_id"] == str(TEST_TENANT_ID)
    assert result["document_id"] == expected_document_id
    assert result["status"] == "completed"
    assert result["chunks_prepared"] == 2
    assert result["chunks_stored"] == 2

    assert len(storage_attempts) == 2

    assert [status for _, _, status, _ in status_updates] == [
        "processing",
        "failed",
        "processing",
        "completed",
    ]

    assert all(
        tenant_id == TEST_TENANT_ID
        for tenant_id, _, _, _ in status_updates
    )

    assert all(
        doc_id == expected_document_id
        for _, doc_id, _, _ in status_updates
    )

    assert status_updates[1][3] == "Temporary Qdrant failure"
    assert status_updates[3][3] is None


def test_ingestion_refreshes_bm25_index_after_success(
    monkeypatch, sample_chunks
):
    events = []

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: events.append("registered"),
    )

    def update_status(tenant_id, doc_id, status, **kwargs):
        events.append(status)

    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        update_status,
    )
    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        lambda chunks: len(chunks),
    )
    monkeypatch.setattr(
        ingestion,
        "refresh_bm25_index",
        lambda tenant_id: events.append("bm25_refreshed"),
    )

    result = ingestion.ingest_document(
        "sample.pdf",
        tenant_id=TEST_TENANT_ID,
    )

    assert result["status"] == "completed"

    assert events == [
        "registered",
        "processing",
        "completed",
        "bm25_refreshed",
    ]


def test_ingestion_cleans_up_vectors_after_storage_failure(
    monkeypatch, sample_chunks
):
    status_updates = []
    cleanup_calls = []

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )
    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        lambda tenant_id, doc_id, status, **kwargs: status_updates.append(
            status
        ),
    )

    def fail_storage(chunks):
        # Simulate a partial write before Qdrant raises an error.
        raise RuntimeError("Partial Qdrant write failure")

    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        fail_storage,
    )
    monkeypatch.setattr(
        ingestion,
        "delete_document_chunks",
        lambda tenant_id, document_id: cleanup_calls.append(
            (tenant_id, document_id)
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Partial Qdrant write failure",
    ):
        ingestion.ingest_document(
            "sample.pdf",
            tenant_id=TEST_TENANT_ID,
        )

    assert status_updates == ["processing", "failed"]
    assert cleanup_calls == [
        (TEST_TENANT_ID, sample_chunks[0].document_id)
    ]


def test_ingestion_preserves_original_error_if_cleanup_fails(
    monkeypatch, sample_chunks
):
    status_updates = []

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )

    def fake_update_status(tenant_id, doc_id, status, **kwargs):
        status_updates.append(status)

    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        fake_update_status,
    )

    def fail_storage(chunks):
        raise RuntimeError("Original Qdrant storage failure")

    def fail_cleanup(tenant_id, document_id):
        raise RuntimeError("Qdrant cleanup failure")

    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        fail_storage,
    )
    monkeypatch.setattr(
        ingestion,
        "delete_document_chunks",
        fail_cleanup,
    )

    with pytest.raises(
        RuntimeError,
        match="Original Qdrant storage failure",
    ) as exc_info:
        ingestion.ingest_document(
            "sample.pdf",
            tenant_id=TEST_TENANT_ID,
        )

    assert status_updates == ["processing", "failed"]

    assert any(
        "Qdrant cleanup failure" in note
        for note in exc_info.value.__notes__
    )


def test_ingestion_cleans_up_even_if_failed_status_update_fails(
    monkeypatch, sample_chunks
):
    cleanup_calls = []
    status_updates = []

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )

    def fail_failed_status_update(
        tenant_id, doc_id, status, **kwargs
    ):
        status_updates.append(status)

        if status == "failed":
            raise RuntimeError(
                "PostgreSQL status update failed"
            )

    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        fail_failed_status_update,
    )

    def fail_storage(chunks):
        raise RuntimeError("Original ingestion failure")

    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        fail_storage,
    )
    monkeypatch.setattr(
        ingestion,
        "delete_document_chunks",
        lambda tenant_id, document_id: cleanup_calls.append(
            (tenant_id, document_id)
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Original ingestion failure",
    ) as exc_info:
        ingestion.ingest_document(
            "sample.pdf",
            tenant_id=TEST_TENANT_ID,
        )

    assert status_updates == ["processing", "failed"]
    assert cleanup_calls == [
        (TEST_TENANT_ID, sample_chunks[0].document_id)
    ]

    assert any(
        "PostgreSQL status update failed" in note
        for note in exc_info.value.__notes__
    )


def test_ingestion_fails_if_heartbeat_update_fails(
    monkeypatch, sample_chunks
):
    status_updates = []
    cleanup_calls = []

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )
    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        lambda tenant_id, doc_id, status, **kwargs: status_updates.append(
            (status, kwargs.get("error_message"))
        ),
    )
    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        lambda chunks: len(chunks),
    )
    monkeypatch.setattr(
        ingestion,
        "update_document_heartbeat",
        lambda tenant_id, document_id: False,
    )
    monkeypatch.setattr(
        ingestion,
        "delete_document_chunks",
        lambda tenant_id, document_id: cleanup_calls.append(
            (tenant_id, document_id)
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Could not refresh heartbeat",
    ):
        ingestion.ingest_document(
            "sample.pdf",
            tenant_id=TEST_TENANT_ID,
        )

    assert [status for status, _ in status_updates] == [
        "processing",
        "failed",
    ]
    assert "Could not refresh heartbeat" in status_updates[1][1]
    assert cleanup_calls == [
        (TEST_TENANT_ID, sample_chunks[0].document_id)
    ]


def test_ingestion_refreshes_heartbeat_periodically(
    monkeypatch, sample_chunks
):
    import threading

    heartbeat_calls = []
    storage_started = threading.Event()

    monkeypatch.setattr(
        ingestion,
        "HEARTBEAT_INTERVAL_SECONDS",
        0.05,
    )

    monkeypatch.setattr(
        ingestion,
        "prepare_document",
        lambda *args, **kwargs: sample_chunks,
    )
    monkeypatch.setattr(
        ingestion,
        "register_document",
        lambda *args: None,
    )
    monkeypatch.setattr(
        ingestion,
        "update_document_status",
        lambda *args, **kwargs: None,
    )

    def fake_heartbeat(tenant_id, document_id):
        assert tenant_id == TEST_TENANT_ID
        heartbeat_calls.append(document_id)
        return True

    monkeypatch.setattr(
        ingestion,
        "update_document_heartbeat",
        fake_heartbeat,
    )

    def slow_storage(chunks):
        storage_started.set()

        # Keep storage running long enough for multiple heartbeats.
        assert storage_started.wait(timeout=2)
        threading.Event().wait(0.25)

        return len(chunks)

    monkeypatch.setattr(
        ingestion,
        "upsert_chunks",
        slow_storage,
    )

    monkeypatch.setattr(
        ingestion,
        "refresh_bm25_index",
        lambda tenant_id: None,
    )

    result = ingestion.ingest_document(
        "sample.pdf",
        tenant_id=TEST_TENANT_ID,
    )

    assert result["status"] == "completed"

    # One initial heartbeat plus multiple periodic heartbeats.
    assert len(heartbeat_calls) >= 3

    assert all(
        document_id == sample_chunks[0].document_id
        for document_id in heartbeat_calls
    )