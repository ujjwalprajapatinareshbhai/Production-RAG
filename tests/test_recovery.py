import pytest
from uuid import uuid4

from app import recovery


def test_recover_stale_documents_successfully(monkeypatch):
    tenant_id = uuid4()
    document_id = "a" * 64
    claim_token = uuid4()

    stale_documents = [
        {
            "tenant_id": tenant_id,
            "document_id": document_id,
        },
    ]

    claimed_documents = [
        {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "cleanup_claim_token": claim_token,
        },
    ]

    deleted_ids = []
    completed_updates = []
    refreshed = []

    monkeypatch.setattr(
        recovery,
        "mark_stale_documents_failed",
        lambda stale_after_minutes: stale_documents,
    )

    monkeypatch.setattr(
        recovery,
        "claim_pending_cleanup_documents",
        lambda limit, lease_minutes: claimed_documents,
    )

    def mock_delete(received_tenant_id, received_document_id):
        deleted_ids.append(
            (received_tenant_id, received_document_id)
        )

    monkeypatch.setattr(
        recovery,
        "delete_document_chunks",
        mock_delete,
    )

    def mock_completed(
        received_tenant_id,
        received_document_id,
        received_claim_token,
    ):
        completed_updates.append(
            (
                received_tenant_id,
                received_document_id,
                received_claim_token,
            )
        )
        return {"document_id": received_document_id}

    monkeypatch.setattr(
        recovery,
        "mark_document_cleanup_completed",
        mock_completed,
    )

    monkeypatch.setattr(
        recovery,
        "refresh_all_bm25_indexes",
        lambda: refreshed.append(True),
    )

    result = recovery.recover_stale_documents()

    assert result["stale_documents_marked"] == 1
    assert result["cleanup_tasks_claimed"] == 1
    assert result["cleanup_completed"] == 1
    assert result["cleanup_failed"] == 0

    assert deleted_ids == [
        (tenant_id, document_id)
    ]

    assert completed_updates == [
        (tenant_id, document_id, claim_token)
    ]

    assert refreshed == [True]

    assert result["cleaned_documents"] == [
        {
            "tenant_id": str(tenant_id),
            "document_id": document_id,
        }
    ]

    assert result["bm25_refreshed"] is True
    assert result["bm25_refresh_error"] is None


def test_recover_stale_documents_handles_qdrant_failure(
    monkeypatch,
):
    tenant_id = uuid4()
    document_id = "b" * 64
    claim_token = uuid4()

    monkeypatch.setattr(
        recovery,
        "mark_stale_documents_failed",
        lambda stale_after_minutes: [],
    )

    monkeypatch.setattr(
        recovery,
        "claim_pending_cleanup_documents",
        lambda limit, lease_minutes: [
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "cleanup_claim_token": claim_token,
            }
        ],
    )

    def mock_delete(received_tenant_id, received_document_id):
        assert received_tenant_id == tenant_id
        assert received_document_id == document_id
        raise ConnectionError("Qdrant connection failed.")

    monkeypatch.setattr(
        recovery,
        "delete_document_chunks",
        mock_delete,
    )

    failed_updates = []

    def mock_failed(
        received_tenant_id,
        received_document_id,
        error_message,
        received_claim_token,
    ):
        failed_updates.append(
            (
                received_tenant_id,
                received_document_id,
                error_message,
                received_claim_token,
            )
        )

        return {
            "document_id": received_document_id,
            "cleanup_pending": True,
            "cleanup_error": error_message,
        }

    monkeypatch.setattr(
        recovery,
        "mark_document_cleanup_failed",
        mock_failed,
    )

    refreshed = []

    monkeypatch.setattr(
        recovery,
        "refresh_all_bm25_indexes",
        lambda: refreshed.append(True),
    )

    result = recovery.recover_stale_documents()

    assert result["cleanup_tasks_claimed"] == 1
    assert result["cleanup_completed"] == 0
    assert result["cleanup_failed"] == 1

    assert len(failed_updates) == 1

    (
        failed_tenant_id,
        failed_document_id,
        error_message,
        received_token,
    ) = failed_updates[0]

    assert failed_tenant_id == tenant_id
    assert failed_document_id == document_id
    assert "Qdrant connection failed" in error_message
    assert received_token == claim_token

    assert result["failed_documents"][0]["tenant_id"] == str(tenant_id)
    assert result["failed_documents"][0]["document_id"] == document_id
    assert "Qdrant connection failed" in result["failed_documents"][0]["error"]

    # No stale documents or successful deletions means no refresh.
    assert refreshed == []
    assert result["bm25_refreshed"] is False
    assert result["bm25_refresh_error"] is None


def test_recover_stale_documents_handles_bm25_refresh_failure(
    monkeypatch,
):
    tenant_id = uuid4()
    document_id = "c" * 64
    claim_token = uuid4()

    monkeypatch.setattr(
        recovery,
        "mark_stale_documents_failed",
        lambda stale_after_minutes: [],
    )

    monkeypatch.setattr(
        recovery,
        "claim_pending_cleanup_documents",
        lambda limit, lease_minutes: [
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "cleanup_claim_token": claim_token,
            }
        ],
    )

    monkeypatch.setattr(
        recovery,
        "delete_document_chunks",
        lambda received_tenant_id, received_document_id: None,
    )

    completed_updates = []

    def mock_completed(
        received_tenant_id,
        received_document_id,
        received_claim_token,
    ):
        completed_updates.append(
            (
                received_tenant_id,
                received_document_id,
                received_claim_token,
            )
        )
        return {"document_id": received_document_id}

    monkeypatch.setattr(
        recovery,
        "mark_document_cleanup_completed",
        mock_completed,
    )

    def mock_refresh():
        raise RuntimeError("BM25 refresh failed.")

    monkeypatch.setattr(
        recovery,
        "refresh_all_bm25_indexes",
        mock_refresh,
    )

    result = recovery.recover_stale_documents()

    # Qdrant cleanup remains successful.
    assert completed_updates == [
        (tenant_id, document_id, claim_token)
    ]

    assert result["cleaned_documents"] == [
        {
            "tenant_id": str(tenant_id),
            "document_id": document_id,
        }
    ]

    assert result["cleanup_completed"] == 1
    assert result["cleanup_failed"] == 0

    # BM25 failure is reported separately.
    assert result["bm25_refreshed"] is False
    assert "BM25 refresh failed" in result["bm25_refresh_error"]


def test_recover_stale_documents_refreshes_bm25_when_stale_documents_found(
    monkeypatch,
):
    tenant_id = uuid4()

    monkeypatch.setattr(
        recovery,
        "mark_stale_documents_failed",
        lambda stale_after_minutes: [
            {
                "tenant_id": tenant_id,
                "document_id": "d" * 64,
            }
        ],
    )

    monkeypatch.setattr(
        recovery,
        "claim_pending_cleanup_documents",
        lambda limit, lease_minutes: [],
    )

    refreshed = []

    monkeypatch.setattr(
        recovery,
        "refresh_all_bm25_indexes",
        lambda: refreshed.append(True),
    )

    result = recovery.recover_stale_documents()

    assert result["stale_documents_marked"] == 1
    assert result["cleanup_tasks_claimed"] == 0
    assert refreshed == [True]
    assert result["bm25_refreshed"] is True


def test_recover_stale_documents_does_not_refresh_when_nothing_changed(
    monkeypatch,
):
    monkeypatch.setattr(
        recovery,
        "mark_stale_documents_failed",
        lambda stale_after_minutes: [],
    )

    monkeypatch.setattr(
        recovery,
        "claim_pending_cleanup_documents",
        lambda limit, lease_minutes: [],
    )

    refreshed = []

    monkeypatch.setattr(
        recovery,
        "refresh_all_bm25_indexes",
        lambda: refreshed.append(True),
    )

    result = recovery.recover_stale_documents()

    assert result["stale_documents_marked"] == 0
    assert result["cleanup_tasks_claimed"] == 0
    assert result["cleanup_completed"] == 0
    assert result["cleanup_failed"] == 0

    assert refreshed == []
    assert result["bm25_refreshed"] is False


def test_recover_stale_documents_passes_configuration(
    monkeypatch,
):
    received = {}

    def mock_mark_stale(stale_after_minutes):
        received["stale_after_minutes"] = stale_after_minutes
        return []

    def mock_claim(limit, lease_minutes):
        received["limit"] = limit
        received["lease_minutes"] = lease_minutes
        return []

    monkeypatch.setattr(
        recovery,
        "mark_stale_documents_failed",
        mock_mark_stale,
    )

    monkeypatch.setattr(
        recovery,
        "claim_pending_cleanup_documents",
        mock_claim,
    )

    monkeypatch.setattr(
        recovery,
        "refresh_all_bm25_indexes",
        lambda: None,
    )

    recovery.recover_stale_documents(
        stale_after_minutes=60,
        batch_size=5,
        lease_minutes=20,
    )

    assert received == {
        "stale_after_minutes": 60,
        "limit": 5,
        "lease_minutes": 20,
    }


@pytest.mark.parametrize(
    ("kwargs", "error_message"),
    [
        (
            {"stale_after_minutes": 0},
            "stale_after_minutes must be at least 1",
        ),
        (
            {"batch_size": 0},
            "batch_size must be at least 1",
        ),
        (
            {"lease_minutes": 0},
            "lease_minutes must be at least 1",
        ),
    ],
)
def test_recover_stale_documents_rejects_invalid_arguments(
    kwargs,
    error_message,
):
    with pytest.raises(ValueError, match=error_message):
        recovery.recover_stale_documents(**kwargs)


def test_recovery_rejects_completion_when_lease_is_lost(
    monkeypatch,
):
    tenant_id = uuid4()
    document_id = "e" * 64
    claim_token = uuid4()

    deleted_ids = []
    failed_updates = []

    monkeypatch.setattr(
        recovery,
        "mark_stale_documents_failed",
        lambda stale_after_minutes: [],
    )

    monkeypatch.setattr(
        recovery,
        "claim_pending_cleanup_documents",
        lambda limit, lease_minutes: [
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "cleanup_claim_token": claim_token,
            }
        ],
    )

    monkeypatch.setattr(
        recovery,
        "delete_document_chunks",
        lambda received_tenant_id, received_id: deleted_ids.append(
            (received_tenant_id, received_id)
        ),
    )

    # Simulate another worker reclaiming the task.
    # The old worker's completion update is rejected.
    monkeypatch.setattr(
        recovery,
        "mark_document_cleanup_completed",
        lambda received_tenant_id, received_id, received_token: None,
    )

    def mock_failed(
        received_tenant_id,
        received_id,
        error_message,
        received_token,
    ):
        failed_updates.append(received_id)
        return {"document_id": received_id}

    monkeypatch.setattr(
        recovery,
        "mark_document_cleanup_failed",
        mock_failed,
    )

    with pytest.raises(
        RuntimeError,
        match="no longer owns the cleanup lease",
    ):
        recovery.recover_stale_documents()

    assert deleted_ids == [
        (tenant_id, document_id)
    ]

    # The code must not try to mark the task failed after
    # the completion update has already been rejected.
    assert failed_updates == []


def test_recovery_rejects_failure_update_when_lease_is_lost(
    monkeypatch,
):
    tenant_id = uuid4()
    document_id = "f" * 64
    claim_token = uuid4()

    completed_updates = []

    monkeypatch.setattr(
        recovery,
        "mark_stale_documents_failed",
        lambda stale_after_minutes: [],
    )

    monkeypatch.setattr(
        recovery,
        "claim_pending_cleanup_documents",
        lambda limit, lease_minutes: [
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "cleanup_claim_token": claim_token,
            }
        ],
    )

    def mock_delete(received_tenant_id, received_id):
        raise ConnectionError("Qdrant connection failed.")

    monkeypatch.setattr(
        recovery,
        "delete_document_chunks",
        mock_delete,
    )

    # Simulate another worker reclaiming the task.
    # The old worker's failure update is rejected.
    monkeypatch.setattr(
        recovery,
        "mark_document_cleanup_failed",
        lambda received_tenant_id, received_id, error_message, received_token: None,
    )

    def mock_completed(
        received_tenant_id,
        received_id,
        received_token,
    ):
        completed_updates.append(received_id)
        return {"document_id": received_id}

    monkeypatch.setattr(
        recovery,
        "mark_document_cleanup_completed",
        mock_completed,
    )

    with pytest.raises(
        RuntimeError,
        match="no longer owns the cleanup lease",
    ):
        recovery.recover_stale_documents()

    # Completion must not be attempted after Qdrant fails.
    assert completed_updates == []