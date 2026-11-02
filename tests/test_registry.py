import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.db import get_connection

from app.registry import (
    claim_pending_cleanup_documents,
    get_document,
    initialize_registry,
    list_completed_document_ids,
    list_pending_cleanup_documents,
    list_stale_documents,
    mark_document_cleanup_completed,
    mark_document_cleanup_failed,
    mark_stale_documents_failed,
    register_document,
    update_document_status,
    update_document_heartbeat,
)


@pytest.fixture(scope="module", autouse=True)
def setup_registry():
    initialize_registry()


@pytest.fixture
def document():
    document_id = uuid.uuid4().hex + uuid.uuid4().hex

    data = {
        "document_id": document_id,
        "filename": "test_document.pdf",
    }

    yield data

    # Clean up the test record after each test.
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM documents WHERE document_id = %s",
            (document_id,),
        )


def test_register_document(document):
    result = register_document(
        document["document_id"],
        document["filename"],
    )

    assert result["document_id"] == document["document_id"]
    assert result["filename"] == document["filename"]
    assert result["status"] == "pending"


def test_get_document(document):
    register_document(
        document["document_id"],
        document["filename"],
    )

    result = get_document(document["document_id"])

    assert result is not None
    assert result["filename"] == document["filename"]


def test_duplicate_registration_preserves_status(document):
    document_id = document["document_id"]

    register_document(document_id, "original.pdf")
    update_document_status(document_id, "completed")

    result = register_document(document_id, "renamed.pdf")

    assert result["filename"] == "renamed.pdf"
    assert result["status"] == "completed"


def test_update_document_status(document):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    result = update_document_status(
        document_id,
        "failed",
        "Test processing error",
    )

    assert result["status"] == "failed"
    assert result["error_message"] == "Test processing error"


def test_invalid_status_is_rejected(document):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    with pytest.raises(ValueError):
        update_document_status(document_id, "unknown")


def test_missing_document_returns_none():
    result = get_document("0" * 64)

    assert result is None


def test_list_completed_document_ids(document):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])
    update_document_status(document_id, "completed")

    result = list_completed_document_ids()

    assert document_id in result


def test_list_stale_documents_returns_stale_pending_document(document):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE document_id = %s
            """,
            (document_id,),
        )

    result = list_stale_documents(stale_after_minutes=30)
    result_ids = [row["document_id"] for row in result]

    assert document_id in result_ids


def test_list_stale_documents_returns_stale_processing_document(document):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])
    update_document_status(document_id, "processing")

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE document_id = %s
            """,
            (document_id,),
        )

    result = list_stale_documents(stale_after_minutes=30)
    result_ids = [row["document_id"] for row in result]

    assert document_id in result_ids


def test_list_stale_documents_excludes_recent_document(document):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    result = list_stale_documents(stale_after_minutes=30)
    result_ids = [row["document_id"] for row in result]

    assert document_id not in result_ids


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_list_stale_documents_excludes_completed_and_failed(
    document,
    status,
):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])
    update_document_status(document_id, status)

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE document_id = %s
            """,
            (document_id,),
        )

    result = list_stale_documents(stale_after_minutes=30)
    result_ids = [row["document_id"] for row in result]

    assert document_id not in result_ids


@pytest.mark.parametrize("minutes", [0, -1, -30])
def test_list_stale_documents_rejects_invalid_cutoff(minutes):
    with pytest.raises(ValueError, match="at least 1"):
        list_stale_documents(stale_after_minutes=minutes)


def test_heartbeat_updates_processing_document(document):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])
    update_document_status(document_id, "processing")

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE document_id = %s
            """,
            (document_id,),
        )

    result = update_document_heartbeat(document_id)
    updated_document = get_document(document_id)

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)

    assert result is True
    assert updated_document["updated_at"] > cutoff


@pytest.mark.parametrize("status", ["pending", "completed", "failed"])
def test_heartbeat_does_not_update_non_processing_document(
    document,
    status,
):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    if status != "pending":
        update_document_status(document_id, status)

    result = update_document_heartbeat(document_id)

    assert result is False


def test_heartbeat_returns_false_for_missing_document():
    result = update_document_heartbeat("0" * 64)

    assert result is False


# ---------------------------------------------------------
# Stale document recovery tests
# ---------------------------------------------------------


@pytest.mark.parametrize("status", ["pending", "processing"])
def test_mark_stale_documents_failed_updates_stale_documents(
    document,
    status,
):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    if status == "processing":
        update_document_status(document_id, "processing")

    # Simulate an ingestion that has been inactive for two hours.
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE document_id = %s
            """,
            (document_id,),
        )

    result = mark_stale_documents_failed(stale_after_minutes=30)

    updated_document = get_document(document_id)
    result_ids = [row["document_id"] for row in result]

    assert document_id in result_ids
    assert updated_document["status"] == "failed"
    assert (
        updated_document["error_message"]
        == "Document ingestion became stale."
    )

    assert updated_document["cleanup_pending"] is True
    assert updated_document["cleanup_error"] is None
    assert updated_document["cleanup_claimed_at"] is None
    assert updated_document["cleanup_claim_token"] is None


def test_mark_stale_documents_failed_preserves_recent_document(document):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    result = mark_stale_documents_failed(stale_after_minutes=30)

    updated_document = get_document(document_id)
    result_ids = [row["document_id"] for row in result]

    assert document_id not in result_ids
    assert updated_document["status"] == "pending"
    assert updated_document["cleanup_pending"] is False


def test_mark_stale_documents_failed_preserves_completed_document(document):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])
    update_document_status(document_id, "completed")

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE document_id = %s
            """,
            (document_id,),
        )

    result = mark_stale_documents_failed(stale_after_minutes=30)

    updated_document = get_document(document_id)
    result_ids = [row["document_id"] for row in result]

    assert document_id not in result_ids
    assert updated_document["status"] == "completed"
    assert updated_document["cleanup_pending"] is False


@pytest.mark.parametrize("minutes", [0, -1, -30])
def test_mark_stale_documents_failed_rejects_invalid_cutoff(minutes):
    with pytest.raises(ValueError, match="at least 1"):
        mark_stale_documents_failed(stale_after_minutes=minutes)


# ---------------------------------------------------------
# Pending cleanup tests
# ---------------------------------------------------------


def test_list_pending_cleanup_documents_returns_pending_document(
    document,
):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE document_id = %s
            """,
            (document_id,),
        )

    result = mark_stale_documents_failed(stale_after_minutes=30)

    assert document_id in [
        row["document_id"] for row in result
    ]

    cleanup_documents = list_pending_cleanup_documents()
    cleanup_ids = [
        row["document_id"] for row in cleanup_documents
    ]

    assert document_id in cleanup_ids


def test_list_pending_cleanup_documents_excludes_completed_cleanup(
    document,
):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    cleanup_documents = list_pending_cleanup_documents()
    cleanup_ids = [
        row["document_id"] for row in cleanup_documents
    ]

    assert document_id not in cleanup_ids


# ---------------------------------------------------------
# Cleanup completion tests
# ---------------------------------------------------------


def test_mark_document_cleanup_completed(document):
    document_id = document["document_id"]
    claim_token = uuid.uuid4()

    register_document(document_id, document["filename"])

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_error = 'Previous Qdrant error',
                cleanup_claimed_at = NOW(),
                cleanup_claim_token = %s
            WHERE document_id = %s
            """,
            (claim_token, document_id),
        )

    result = mark_document_cleanup_completed(
        document_id,
        claim_token,
    )

    assert result is not None
    assert result["cleanup_pending"] is False
    assert result["cleanup_error"] is None
    assert result["cleanup_claimed_at"] is None
    assert result["cleanup_claim_token"] is None


def test_mark_document_cleanup_completed_rejects_old_token(document):
    document_id = document["document_id"]

    old_token = uuid.uuid4()
    current_token = uuid.uuid4()

    register_document(document_id, document["filename"])

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_claimed_at = NOW(),
                cleanup_claim_token = %s
            WHERE document_id = %s
            """,
            (current_token, document_id),
        )

    result = mark_document_cleanup_completed(
        document_id,
        old_token,
    )

    updated_document = get_document(document_id)

    assert result is None
    assert updated_document["cleanup_pending"] is True
    assert updated_document["cleanup_claim_token"] == current_token


def test_mark_document_cleanup_completed_returns_none_for_missing_document():
    result = mark_document_cleanup_completed(
        "0" * 64,
        uuid.uuid4(),
    )

    assert result is None


# ---------------------------------------------------------
# Cleanup failure tests
# ---------------------------------------------------------


def test_mark_document_cleanup_failed(document):
    document_id = document["document_id"]
    claim_token = uuid.uuid4()

    register_document(document_id, document["filename"])

    error_message = "Qdrant connection failed."

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_claimed_at = NOW(),
                cleanup_claim_token = %s
            WHERE document_id = %s
            """,
            (claim_token, document_id),
        )

    result = mark_document_cleanup_failed(
        document_id,
        error_message,
        claim_token,
    )

    assert result is not None
    assert result["cleanup_pending"] is True
    assert result["cleanup_error"] == error_message
    assert result["cleanup_claimed_at"] is None
    assert result["cleanup_claim_token"] is None


def test_mark_document_cleanup_failed_rejects_old_token(document):
    document_id = document["document_id"]

    old_token = uuid.uuid4()
    current_token = uuid.uuid4()

    register_document(document_id, document["filename"])

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_claimed_at = NOW(),
                cleanup_claim_token = %s
            WHERE document_id = %s
            """,
            (current_token, document_id),
        )

    result = mark_document_cleanup_failed(
        document_id,
        "Old worker error",
        old_token,
    )

    updated_document = get_document(document_id)

    assert result is None
    assert updated_document["cleanup_pending"] is True
    assert updated_document["cleanup_claim_token"] == current_token
    assert updated_document["cleanup_error"] is None


def test_mark_document_cleanup_failed_rejects_empty_error(document):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    with pytest.raises(ValueError, match="cannot be empty"):
        mark_document_cleanup_failed(
            document_id,
            "",
            uuid.uuid4(),
        )


def test_mark_document_cleanup_failed_returns_none_for_missing_document():
    result = mark_document_cleanup_failed(
        "0" * 64,
        "Qdrant connection failed.",
        uuid.uuid4(),
    )

    assert result is None


# ---------------------------------------------------------
# Cleanup claim tests
# ---------------------------------------------------------


def test_claim_pending_cleanup_documents_claims_unclaimed_document(
    document,
):
    document_id = document["document_id"]

    register_document(document_id, document["filename"])

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET cleanup_pending = TRUE
            WHERE document_id = %s
            """,
            (document_id,),
        )

    result = claim_pending_cleanup_documents()

    claimed_document = next(
        (
            row
            for row in result
            if row["document_id"] == document_id
        ),
        None,
    )

    assert claimed_document is not None
    assert claimed_document["cleanup_claimed_at"] is not None
    assert isinstance(claimed_document["cleanup_claim_token"], uuid.UUID)

    stored_document = get_document(document_id)

    assert stored_document["cleanup_claim_token"] == (
        claimed_document["cleanup_claim_token"]
    )


def test_claim_pending_cleanup_documents_skips_active_lease(
    document,
):
    document_id = document["document_id"]
    existing_token = uuid.uuid4()

    register_document(document_id, document["filename"])

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_claimed_at = NOW(),
                cleanup_claim_token = %s
            WHERE document_id = %s
            """,
            (existing_token, document_id),
        )

    result = claim_pending_cleanup_documents()

    claimed_ids = [row["document_id"] for row in result]

    assert document_id not in claimed_ids

    stored_document = get_document(document_id)

    assert stored_document["cleanup_claim_token"] == existing_token


def test_claim_pending_cleanup_documents_reclaims_expired_lease(
    document,
):
    document_id = document["document_id"]

    old_token = uuid.uuid4()

    register_document(document_id, document["filename"])

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_claimed_at = NOW() - INTERVAL '2 hours',
                cleanup_claim_token = %s
            WHERE document_id = %s
            """,
            (old_token, document_id),
        )

    result = claim_pending_cleanup_documents(
        lease_minutes=15,
    )

    claimed_document = next(
        (
            row
            for row in result
            if row["document_id"] == document_id
        ),
        None,
    )

    assert claimed_document is not None

    new_token = claimed_document["cleanup_claim_token"]

    assert new_token is not None
    assert new_token != old_token
    assert isinstance(new_token, uuid.UUID)

    stored_document = get_document(document_id)

    assert stored_document["cleanup_claim_token"] == new_token
    assert stored_document["cleanup_claimed_at"] is not None


def test_claim_pending_cleanup_documents_respects_limit(
    document,
):
    first_id = document["document_id"]
    second_id = uuid.uuid4().hex + uuid.uuid4().hex

    register_document(first_id, "first.pdf")
    register_document(second_id, "second.pdf")

    try:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET cleanup_pending = TRUE
                WHERE document_id IN (%s, %s)
                """,
                (first_id, second_id),
            )

        result = claim_pending_cleanup_documents(limit=1)

        assert len(result) == 1

        claimed_id = result[0]["document_id"]

        assert claimed_id in {first_id, second_id}
        assert result[0]["cleanup_claim_token"] is not None

    finally:
        with get_connection() as conn:
            conn.execute(
                """
                DELETE FROM documents
                WHERE document_id = %s
                """,
                (second_id,),
            )


@pytest.mark.parametrize("limit", [0, -1])
def test_claim_pending_cleanup_documents_rejects_invalid_limit(
    limit,
):
    with pytest.raises(ValueError, match="limit must be at least 1"):
        claim_pending_cleanup_documents(limit=limit)


@pytest.mark.parametrize("lease_minutes", [0, -1])
def test_claim_pending_cleanup_documents_rejects_invalid_lease(
    lease_minutes,
):
    with pytest.raises(
        ValueError,
        match="lease_minutes must be at least 1",
    ):
        claim_pending_cleanup_documents(
            lease_minutes=lease_minutes,
        )