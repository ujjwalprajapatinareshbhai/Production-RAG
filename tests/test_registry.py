import uuid
from datetime import datetime, timedelta, timezone
from psycopg.errors import UniqueViolation
import pytest

from app.db import get_connection
from app.auth import generate_invitation_token, hash_invitation_token
from app.registry import (
    claim_pending_cleanup_documents,
    create_tenant,
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
    create_invitation,
    create_membership,
    create_user,
    get_active_membership,
    get_user_by_id,
    accept_invitation,
)


# =========================================================
# TEST DATABASE INITIALIZATION
# =========================================================

@pytest.fixture(scope="module", autouse=True)
def setup_registry():
    initialize_registry()


# =========================================================
# TEST TENANT FIXTURE
# =========================================================

@pytest.fixture
def tenant():
    """Create an isolated tenant for a registry test."""

    tenant_data = create_tenant(
        name="Registry Test Tenant",
        slug=f"registry-test-{uuid.uuid4().hex}",
    )

    yield tenant_data

    # Delete the tenant after its test documents are removed.
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM tenants
            WHERE tenant_id = %s
            """,
            (tenant_data["tenant_id"],),
        )


# =========================================================
# TEST DOCUMENT FIXTURE
# =========================================================

@pytest.fixture
def document(tenant):
    """Create test document data belonging to one tenant."""

    document_id = uuid.uuid4().hex + uuid.uuid4().hex

    data = {
        "tenant_id": tenant["tenant_id"],
        "document_id": document_id,
        "filename": "test_document.pdf",
    }

    yield data

    # Delete only this tenant's document.
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM documents
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (
                data["tenant_id"],
                document_id,
            ),
        )


# =========================================================
# DOCUMENT REGISTRATION TESTS
# =========================================================

def test_register_document(document):
    result = register_document(
        document["tenant_id"],
        document["document_id"],
        document["filename"],
    )

    assert result["tenant_id"] == document["tenant_id"]
    assert result["document_id"] == document["document_id"]
    assert result["filename"] == document["filename"]
    assert result["status"] == "pending"


def test_get_document(document):
    register_document(
        document["tenant_id"],
        document["document_id"],
        document["filename"],
    )

    result = get_document(
        document["tenant_id"],
        document["document_id"],
    )

    assert result is not None
    assert result["filename"] == document["filename"]


def test_duplicate_registration_preserves_status(document):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        "original.pdf",
    )

    update_document_status(
        tenant_id,
        document_id,
        "completed",
    )

    result = register_document(
        tenant_id,
        document_id,
        "renamed.pdf",
    )

    assert result["filename"] == "renamed.pdf"
    assert result["status"] == "completed"


def test_update_document_status(document):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    result = update_document_status(
        tenant_id,
        document_id,
        "failed",
        "Test processing error",
    )

    assert result["status"] == "failed"
    assert result["error_message"] == "Test processing error"


def test_invalid_status_is_rejected(document):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    with pytest.raises(ValueError):
        update_document_status(
            tenant_id,
            document_id,
            "unknown",
        )


def test_missing_document_returns_none(tenant):
    result = get_document(
        tenant["tenant_id"],
        "0" * 64,
    )

    assert result is None


# =========================================================
# TENANT ISOLATION TESTS
# =========================================================

def test_same_document_id_can_exist_in_two_tenants(tenant):
    second_tenant = create_tenant(
        name="Second Registry Test Tenant",
        slug=f"registry-test-{uuid.uuid4().hex}",
    )

    document_id = uuid.uuid4().hex + uuid.uuid4().hex

    try:
        first_result = register_document(
            tenant["tenant_id"],
            document_id,
            "first_tenant.pdf",
        )

        second_result = register_document(
            second_tenant["tenant_id"],
            document_id,
            "second_tenant.pdf",
        )

        assert first_result["document_id"] == document_id
        assert second_result["document_id"] == document_id

        assert first_result["tenant_id"] != second_result["tenant_id"]

        assert first_result["filename"] == "first_tenant.pdf"
        assert second_result["filename"] == "second_tenant.pdf"

    finally:
        # Delete both documents before deleting either tenant.
        with get_connection() as conn:
            conn.execute(
                """
                DELETE FROM documents
                WHERE document_id = %s
                  AND tenant_id IN (%s, %s)
                """,
                (
                    document_id,
                    tenant["tenant_id"],
                    second_tenant["tenant_id"],
                ),
            )

            conn.execute(
                """
                DELETE FROM tenants
                WHERE tenant_id = %s
                """,
                (second_tenant["tenant_id"],),
            )


def test_tenant_cannot_get_another_tenants_document(document):
    second_tenant = create_tenant(
        name="Another Registry Tenant",
        slug=f"registry-test-{uuid.uuid4().hex}",
    )

    try:
        register_document(
            document["tenant_id"],
            document["document_id"],
            document["filename"],
        )

        result = get_document(
            second_tenant["tenant_id"],
            document["document_id"],
        )

        assert result is None

    finally:
        with get_connection() as conn:
            conn.execute(
                """
                DELETE FROM tenants
                WHERE tenant_id = %s
                """,
                (second_tenant["tenant_id"],),
            )


def test_tenant_cannot_update_another_tenants_document(document):
    second_tenant = create_tenant(
        name="Another Registry Tenant",
        slug=f"registry-test-{uuid.uuid4().hex}",
    )

    try:
        register_document(
            document["tenant_id"],
            document["document_id"],
            document["filename"],
        )

        result = update_document_status(
            second_tenant["tenant_id"],
            document["document_id"],
            "completed",
        )

        original = get_document(
            document["tenant_id"],
            document["document_id"],
        )

        assert result is None
        assert original["status"] == "pending"

    finally:
        with get_connection() as conn:
            conn.execute(
                """
                DELETE FROM tenants
                WHERE tenant_id = %s
                """,
                (second_tenant["tenant_id"],),
            )


# =========================================================
# COMPLETED DOCUMENT TESTS
# =========================================================

def test_list_completed_document_ids(document):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    update_document_status(
        tenant_id,
        document_id,
        "completed",
    )

    result = list_completed_document_ids(tenant_id)

    assert document_id in result


def test_completed_document_ids_are_tenant_scoped(document):
    second_tenant = create_tenant(
        name="Another Registry Tenant",
        slug=f"registry-test-{uuid.uuid4().hex}",
    )

    try:
        register_document(
            document["tenant_id"],
            document["document_id"],
            document["filename"],
        )

        update_document_status(
            document["tenant_id"],
            document["document_id"],
            "completed",
        )

        result = list_completed_document_ids(
            second_tenant["tenant_id"],
        )

        assert document["document_id"] not in result

    finally:
        with get_connection() as conn:
            conn.execute(
                """
                DELETE FROM tenants
                WHERE tenant_id = %s
                """,
                (second_tenant["tenant_id"],),
            )


# =========================================================
# STALE DOCUMENT TESTS
# System-wide recovery operations
# =========================================================

def test_list_stale_documents_returns_stale_pending_document(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (tenant_id, document_id),
        )

    result = list_stale_documents(stale_after_minutes=30)
    result_ids = [row["document_id"] for row in result]

    assert document_id in result_ids


def test_list_stale_documents_returns_stale_processing_document(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    update_document_status(
        tenant_id,
        document_id,
        "processing",
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (tenant_id, document_id),
        )

    result = list_stale_documents(stale_after_minutes=30)
    result_ids = [row["document_id"] for row in result]

    assert document_id in result_ids


def test_list_stale_documents_excludes_recent_document(document):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    result = list_stale_documents(stale_after_minutes=30)
    result_ids = [row["document_id"] for row in result]

    assert document_id not in result_ids


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_list_stale_documents_excludes_completed_and_failed(
    document,
    status,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    update_document_status(
        tenant_id,
        document_id,
        status,
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (tenant_id, document_id),
        )

    result = list_stale_documents(stale_after_minutes=30)
    result_ids = [row["document_id"] for row in result]

    assert document_id not in result_ids


@pytest.mark.parametrize("minutes", [0, -1, -30])
def test_list_stale_documents_rejects_invalid_cutoff(minutes):
    with pytest.raises(ValueError, match="at least 1"):
        list_stale_documents(stale_after_minutes=minutes)


# =========================================================
# HEARTBEAT TESTS
# =========================================================

def test_heartbeat_updates_processing_document(document):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    update_document_status(
        tenant_id,
        document_id,
        "processing",
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (tenant_id, document_id),
        )

    result = update_document_heartbeat(
        tenant_id,
        document_id,
    )

    updated_document = get_document(
        tenant_id,
        document_id,
    )

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)

    assert result is True
    assert updated_document["updated_at"] > cutoff


@pytest.mark.parametrize(
    "status",
    ["pending", "completed", "failed"],
)
def test_heartbeat_does_not_update_non_processing_document(
    document,
    status,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    if status != "pending":
        update_document_status(
            tenant_id,
            document_id,
            status,
        )

    result = update_document_heartbeat(
        tenant_id,
        document_id,
    )

    assert result is False


def test_heartbeat_returns_false_for_missing_document(tenant):
    result = update_document_heartbeat(
        tenant["tenant_id"],
        "0" * 64,
    )

    assert result is False


# =========================================================
# STALE DOCUMENT RECOVERY TESTS
# =========================================================

@pytest.mark.parametrize("status", ["pending", "processing"])
def test_mark_stale_documents_failed_updates_stale_documents(
    document,
    status,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    if status == "processing":
        update_document_status(
            tenant_id,
            document_id,
            "processing",
        )

    # Simulate ingestion inactive for two hours.
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (tenant_id, document_id),
        )

    result = mark_stale_documents_failed(
        stale_after_minutes=30,
    )

    updated_document = get_document(
        tenant_id,
        document_id,
    )

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


def test_mark_stale_documents_failed_preserves_recent_document(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    result = mark_stale_documents_failed(
        stale_after_minutes=30,
    )

    updated_document = get_document(
        tenant_id,
        document_id,
    )

    result_ids = [row["document_id"] for row in result]

    assert document_id not in result_ids
    assert updated_document["status"] == "pending"
    assert updated_document["cleanup_pending"] is False


def test_mark_stale_documents_failed_preserves_completed_document(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    update_document_status(
        tenant_id,
        document_id,
        "completed",
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (tenant_id, document_id),
        )

    result = mark_stale_documents_failed(
        stale_after_minutes=30,
    )

    updated_document = get_document(
        tenant_id,
        document_id,
    )

    result_ids = [row["document_id"] for row in result]

    assert document_id not in result_ids
    assert updated_document["status"] == "completed"
    assert updated_document["cleanup_pending"] is False


@pytest.mark.parametrize("minutes", [0, -1, -30])
def test_mark_stale_documents_failed_rejects_invalid_cutoff(
    minutes,
):
    with pytest.raises(ValueError, match="at least 1"):
        mark_stale_documents_failed(
            stale_after_minutes=minutes,
        )


# =========================================================
# PENDING CLEANUP TESTS
# =========================================================

def test_list_pending_cleanup_documents_returns_pending_document(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET updated_at = NOW() - INTERVAL '2 hours'
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (tenant_id, document_id),
        )

    result = mark_stale_documents_failed(
        stale_after_minutes=30,
    )

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
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    cleanup_documents = list_pending_cleanup_documents()

    cleanup_ids = [
        row["document_id"] for row in cleanup_documents
    ]

    assert document_id not in cleanup_ids


# =========================================================
# CLEANUP COMPLETION TESTS
# =========================================================

def test_mark_document_cleanup_completed(document):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]
    claim_token = uuid.uuid4()

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_error = 'Previous Qdrant error',
                cleanup_claimed_at = NOW(),
                cleanup_claim_token = %s
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (claim_token, tenant_id, document_id),
        )

    result = mark_document_cleanup_completed(
        tenant_id,
        document_id,
        claim_token,
    )

    assert result is not None
    assert result["cleanup_pending"] is False
    assert result["cleanup_error"] is None
    assert result["cleanup_claimed_at"] is None
    assert result["cleanup_claim_token"] is None


def test_mark_document_cleanup_completed_rejects_old_token(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    old_token = uuid.uuid4()
    current_token = uuid.uuid4()

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_claimed_at = NOW(),
                cleanup_claim_token = %s
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (current_token, tenant_id, document_id),
        )

    result = mark_document_cleanup_completed(
        tenant_id,
        document_id,
        old_token,
    )

    updated_document = get_document(
        tenant_id,
        document_id,
    )

    assert result is None
    assert updated_document["cleanup_pending"] is True
    assert updated_document["cleanup_claim_token"] == current_token


def test_mark_document_cleanup_completed_returns_none_for_missing_document(
    tenant,
):
    result = mark_document_cleanup_completed(
        tenant["tenant_id"],
        "0" * 64,
        uuid.uuid4(),
    )

    assert result is None


# =========================================================
# CLEANUP FAILURE TESTS
# =========================================================

def test_mark_document_cleanup_failed(document):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]
    claim_token = uuid.uuid4()

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    error_message = "Qdrant connection failed."

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_claimed_at = NOW(),
                cleanup_claim_token = %s
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (claim_token, tenant_id, document_id),
        )

    result = mark_document_cleanup_failed(
        tenant_id,
        document_id,
        error_message,
        claim_token,
    )

    assert result is not None
    assert result["cleanup_pending"] is True
    assert result["cleanup_error"] == error_message
    assert result["cleanup_claimed_at"] is None
    assert result["cleanup_claim_token"] is None


def test_mark_document_cleanup_failed_rejects_old_token(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    old_token = uuid.uuid4()
    current_token = uuid.uuid4()

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_claimed_at = NOW(),
                cleanup_claim_token = %s
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (current_token, tenant_id, document_id),
        )

    result = mark_document_cleanup_failed(
        tenant_id,
        document_id,
        "Old worker error",
        old_token,
    )

    updated_document = get_document(
        tenant_id,
        document_id,
    )

    assert result is None
    assert updated_document["cleanup_pending"] is True
    assert updated_document["cleanup_claim_token"] == current_token
    assert updated_document["cleanup_error"] is None


def test_mark_document_cleanup_failed_rejects_empty_error(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        mark_document_cleanup_failed(
            tenant_id,
            document_id,
            "",
            uuid.uuid4(),
        )


def test_mark_document_cleanup_failed_returns_none_for_missing_document(
    tenant,
):
    result = mark_document_cleanup_failed(
        tenant["tenant_id"],
        "0" * 64,
        "Qdrant connection failed.",
        uuid.uuid4(),
    )

    assert result is None


# =========================================================
# CLEANUP CLAIM TESTS
# =========================================================

def test_claim_pending_cleanup_documents_claims_unclaimed_document(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET cleanup_pending = TRUE
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (tenant_id, document_id),
        )

    result = claim_pending_cleanup_documents()

    claimed_document = next(
        (
            row
            for row in result
            if row["tenant_id"] == tenant_id
            and row["document_id"] == document_id
        ),
        None,
    )

    assert claimed_document is not None
    assert claimed_document["cleanup_claimed_at"] is not None

    assert isinstance(
        claimed_document["cleanup_claim_token"],
        uuid.UUID,
    )

    stored_document = get_document(
        tenant_id,
        document_id,
    )

    assert stored_document["cleanup_claim_token"] == (
        claimed_document["cleanup_claim_token"]
    )


def test_claim_pending_cleanup_documents_skips_active_lease(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]
    existing_token = uuid.uuid4()

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_claimed_at = NOW(),
                cleanup_claim_token = %s
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (existing_token, tenant_id, document_id),
        )

    result = claim_pending_cleanup_documents()

    claimed_ids = [
        (row["tenant_id"], row["document_id"])
        for row in result
    ]

    assert (tenant_id, document_id) not in claimed_ids

    stored_document = get_document(
        tenant_id,
        document_id,
    )

    assert stored_document["cleanup_claim_token"] == existing_token


def test_claim_pending_cleanup_documents_reclaims_expired_lease(
    document,
):
    tenant_id = document["tenant_id"]
    document_id = document["document_id"]

    old_token = uuid.uuid4()

    register_document(
        tenant_id,
        document_id,
        document["filename"],
    )

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET
                cleanup_pending = TRUE,
                cleanup_claimed_at = NOW() - INTERVAL '2 hours',
                cleanup_claim_token = %s
            WHERE tenant_id = %s
              AND document_id = %s
            """,
            (old_token, tenant_id, document_id),
        )

    result = claim_pending_cleanup_documents(
        lease_minutes=15,
    )

    claimed_document = next(
        (
            row
            for row in result
            if row["tenant_id"] == tenant_id
            and row["document_id"] == document_id
        ),
        None,
    )

    assert claimed_document is not None

    new_token = claimed_document["cleanup_claim_token"]

    assert new_token is not None
    assert new_token != old_token
    assert isinstance(new_token, uuid.UUID)

    stored_document = get_document(
        tenant_id,
        document_id,
    )

    assert stored_document["cleanup_claim_token"] == new_token
    assert stored_document["cleanup_claimed_at"] is not None


def test_claim_pending_cleanup_documents_respects_limit(
    document,
):
    tenant_id = document["tenant_id"]
    first_id = document["document_id"]
    second_id = uuid.uuid4().hex + uuid.uuid4().hex

    register_document(
        tenant_id,
        first_id,
        "first.pdf",
    )

    register_document(
        tenant_id,
        second_id,
        "second.pdf",
    )

    try:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET
                    cleanup_pending = TRUE,
                    cleanup_claimed_at = NULL,
                    updated_at = NOW() - INTERVAL '1 day'
                WHERE tenant_id = %s
                   AND document_id IN (%s, %s)
                """,
                (tenant_id, first_id, second_id),
            )

        result = claim_pending_cleanup_documents(limit=1)

        assert len(result) == 1

        claimed_id = result[0]["document_id"]

        assert claimed_id in {first_id, second_id}
        assert result[0]["tenant_id"] == tenant_id
        assert result[0]["cleanup_claim_token"] is not None

    finally:
        with get_connection() as conn:
            conn.execute(
                """
                DELETE FROM documents
                WHERE tenant_id = %s
                  AND document_id = %s
                """,
                (tenant_id, second_id),
            )


@pytest.mark.parametrize("limit", [0, -1])
def test_claim_pending_cleanup_documents_rejects_invalid_limit(
    limit,
):
    with pytest.raises(
        ValueError,
        match="limit must be at least 1",
    ):
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

@pytest.fixture
def invitation_users(tenant):
    """Create and clean up users for invitation tests."""

    owner = create_user(
        f"owner-{uuid.uuid4().hex}@example.com",
        "test-password-hash",
    )

    admin = create_user(
        f"admin-{uuid.uuid4().hex}@example.com",
        "test-password-hash",
    )

    member = create_user(
        f"member-{uuid.uuid4().hex}@example.com",
        "test-password-hash",
    )

    users = {
        "owner": owner,
        "admin": admin,
        "member": member,
    }

    for role, user in users.items():
        create_membership(
            tenant["tenant_id"],
            user["user_id"],
            role,
        )

    yield users

    # Clean up invitations and memberships first.
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM tenant_invitations
            WHERE tenant_id = %s;
            """,
            (tenant["tenant_id"],),
        )

        conn.execute(
            """
            DELETE FROM tenant_memberships
            WHERE tenant_id = %s;
            """,
            (tenant["tenant_id"],),
        )

        # Users are independent accounts, so delete them explicitly.
        for user in users.values():
            conn.execute(
                """
                DELETE FROM users
                WHERE user_id = %s;
                """,
                (user["user_id"],),
            )

@pytest.fixture
def pending_invitation(invitation_users, tenant):
    """Create a pending invitation for a new member."""

    email = f"invitee-{uuid.uuid4().hex}@example.com"
    token = generate_invitation_token()

    invitation = create_invitation(
        tenant_id=tenant["tenant_id"],
        email=email,
        role="member",
        invited_by=invitation_users["owner"]["user_id"],
        token_hash=hash_invitation_token(token),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    return {
        "email": email,
        "token": token,
        "token_hash": hash_invitation_token(token),
        "invitation": invitation,
    }


@pytest.fixture
def invited_user(pending_invitation):
    """Create the user who was invited."""

    user = create_user(
        pending_invitation["email"],
        "test-password-hash",
    )

    yield user

    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM tenant_memberships
            WHERE user_id = %s;
            """,
            (user["user_id"],),
        )

        conn.execute(
            """
            DELETE FROM users
            WHERE user_id = %s;
            """,
            (user["user_id"],),
        )

# =========================================================
# TENANT INVITATION PERMISSION TESTS
# =========================================================

def test_owner_can_invite_admin(invitation_users, tenant):
    token = generate_invitation_token()

    result = create_invitation(
        tenant_id=tenant["tenant_id"],
        email="new-admin@example.com",
        role="admin",
        invited_by=invitation_users["owner"]["user_id"],
        token_hash=hash_invitation_token(token),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    assert result["email"] == "new-admin@example.com"
    assert result["role"] == "admin"
    assert result["tenant_id"] == tenant["tenant_id"]
    assert "token_hash" not in result


def test_admin_can_invite_member(invitation_users, tenant):
    token = generate_invitation_token()

    result = create_invitation(
        tenant_id=tenant["tenant_id"],
        email="new-member@example.com",
        role="member",
        invited_by=invitation_users["admin"]["user_id"],
        token_hash=hash_invitation_token(token),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    assert result["email"] == "new-member@example.com"
    assert result["role"] == "member"


def test_admin_cannot_invite_admin(invitation_users, tenant):
    token = generate_invitation_token()

    with pytest.raises(
        PermissionError,
        match="Only owners can invite administrators",
    ):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email="another-admin@example.com",
            role="admin",
            invited_by=invitation_users["admin"]["user_id"],
            token_hash=hash_invitation_token(token),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )


def test_member_cannot_invite_users(invitation_users, tenant):
    token = generate_invitation_token()

    with pytest.raises(
        PermissionError,
        match="Only owners and admins can invite users",
    ):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email="unauthorized@example.com",
            role="member",
            invited_by=invitation_users["member"]["user_id"],
            token_hash=hash_invitation_token(token),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )

def test_duplicate_active_invitation_is_rejected(
    invitation_users,
    tenant,
):
    email = "duplicate@example.com"

    create_invitation(
        tenant_id=tenant["tenant_id"],
        email=email,
        role="member",
        invited_by=invitation_users["owner"]["user_id"],
        token_hash=hash_invitation_token(
            generate_invitation_token()
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    with pytest.raises(UniqueViolation):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email=email,
            role="member",
            invited_by=invitation_users["owner"]["user_id"],
            token_hash=hash_invitation_token(
                generate_invitation_token()
            ),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )

def test_expired_invitation_can_be_replaced(
    invitation_users,
    tenant,
):
    email = "expired-invitation@example.com"

    first_invitation = create_invitation(
        tenant_id=tenant["tenant_id"],
        email=email,
        role="member",
        invited_by=invitation_users["owner"]["user_id"],
        token_hash=hash_invitation_token(
            generate_invitation_token()
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    # Simulate expiration directly in the test database.
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE tenant_invitations
            SET expires_at = NOW() - INTERVAL '1 hour'
            WHERE invitation_id = %s;
            """,
            (first_invitation["invitation_id"],),
        )

    replacement = create_invitation(
        tenant_id=tenant["tenant_id"],
        email=email,
        role="member",
        invited_by=invitation_users["owner"]["user_id"],
        token_hash=hash_invitation_token(
            generate_invitation_token()
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    assert replacement["invitation_id"] != first_invitation["invitation_id"]

    # Confirm the expired invitation was revoked.
    with get_connection() as conn:
        old_invitation = conn.execute(
            """
            SELECT revoked_at
            FROM tenant_invitations
            WHERE invitation_id = %s;
            """,
            (first_invitation["invitation_id"],),
        ).fetchone()

    assert old_invitation[0] is not None

def test_invitation_rejects_invalid_role(invitation_users, tenant):
    with pytest.raises(
        ValueError,
        match="Invalid invitation role",
    ):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email="invalid-role@example.com",
            role="owner",
            invited_by=invitation_users["owner"]["user_id"],
            token_hash=hash_invitation_token(
                generate_invitation_token()
            ),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
     )
          
def test_invitation_rejects_invalid_token_hash(invitation_users, tenant):
    with pytest.raises(
        ValueError,
        match="Invalid invitation token hash",
    ):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email="invalid-token@example.com",
            role="member",
            invited_by=invitation_users["owner"]["user_id"],
            token_hash="not-a-valid-hash",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )

def test_invitation_rejects_empty_email(invitation_users, tenant):
    with pytest.raises(
        ValueError,
        match="Email cannot be empty",
    ):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email="   ",
            role="member",
            invited_by=invitation_users["owner"]["user_id"],
            token_hash=hash_invitation_token(
                generate_invitation_token()
            ),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )

def test_invitation_rejects_past_expiry(invitation_users, tenant):
    with pytest.raises(
        ValueError,
        match="Invitation expiry must be in the future",
    ):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email="past-expiry@example.com",
            role="member",
            invited_by=invitation_users["owner"]["user_id"],
            token_hash=hash_invitation_token(
                generate_invitation_token()
            ),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

def test_invitation_rejects_naive_expiry(invitation_users, tenant):
    with pytest.raises(
        ValueError,
        match="Invitation expiry must be timezone-aware",
    ):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email="naive-expiry@example.com",
            role="member",
            invited_by=invitation_users["owner"]["user_id"],
            token_hash=hash_invitation_token(
                generate_invitation_token()
            ),
            expires_at=datetime.now() + timedelta(hours=24),
        )

def test_invitation_rejects_short_token_hash(invitation_users, tenant):
    with pytest.raises(
        ValueError,
        match="Invalid invitation token hash",
    ):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email="short-hash@example.com",
            role="member",
            invited_by=invitation_users["owner"]["user_id"],
            token_hash="a" * 32,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )

def test_invitation_rejects_non_hex_token_hash(invitation_users, tenant):
    with pytest.raises(
        ValueError,
        match="Invalid invitation token hash",
    ):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email="non-hex-hash@example.com",
            role="member",
            invited_by=invitation_users["owner"]["user_id"],
            token_hash="g" * 64,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )

def test_invitation_normalizes_email(invitation_users, tenant):
    result = create_invitation(
        tenant_id=tenant["tenant_id"],
        email="  NEW.USER@EXAMPLE.COM  ",
        role="member",
        invited_by=invitation_users["owner"]["user_id"],
        token_hash=hash_invitation_token(
            generate_invitation_token()
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    assert result["email"] == "new.user@example.com"

def test_invitation_result_does_not_expose_token_hash(
    invitation_users,
    tenant,
):
    result = create_invitation(
        tenant_id=tenant["tenant_id"],
        email="secure-invitation@example.com",
        role="member",
        invited_by=invitation_users["owner"]["user_id"],
        token_hash=hash_invitation_token(
            generate_invitation_token()
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    assert "token_hash" not in result
    assert "token" not in result

def test_invitation_stores_token_hash(
    invitation_users,
    tenant,
):
    token = generate_invitation_token()
    token_hash = hash_invitation_token(token)

    result = create_invitation(
        tenant_id=tenant["tenant_id"],
        email="stored-hash@example.com",
        role="member",
        invited_by=invitation_users["owner"]["user_id"],
        token_hash=token_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    with get_connection() as conn:
        stored_hash = conn.execute(
            """
            SELECT token_hash
            FROM tenant_invitations
            WHERE invitation_id = %s;
            """,
            (result["invitation_id"],),
        ).fetchone()

    assert stored_hash[0] == token_hash
    assert stored_hash[0] != token

def test_new_invitation_starts_pending(
    invitation_users,
    tenant,
):
    result = create_invitation(
        tenant_id=tenant["tenant_id"],
        email="pending-invitation@example.com",
        role="member",
        invited_by=invitation_users["owner"]["user_id"],
        token_hash=hash_invitation_token(
            generate_invitation_token()
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    with get_connection() as conn:
        invitation = conn.execute(
            """
            SELECT accepted_at, revoked_at, expires_at
            FROM tenant_invitations
            WHERE invitation_id = %s;
            """,
            (result["invitation_id"],),
        ).fetchone()

    assert invitation[0] is None
    assert invitation[1] is None
    assert invitation[2] > datetime.now(timezone.utc)

def test_invitation_records_inviter(
    invitation_users,
    tenant,
):
    inviter_id = invitation_users["owner"]["user_id"]

    result = create_invitation(
        tenant_id=tenant["tenant_id"],
        email="audited-invitation@example.com",
        role="member",
        invited_by=inviter_id,
        token_hash=hash_invitation_token(
            generate_invitation_token()
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    with get_connection() as conn:
        recorded_inviter = conn.execute(
            """
            SELECT invited_by
            FROM tenant_invitations
            WHERE invitation_id = %s;
            """,
            (result["invitation_id"],),
        ).fetchone()

    assert recorded_inviter[0] == inviter_id

def test_invitation_records_correct_tenant(
    invitation_users,
    tenant,
):
    result = create_invitation(
        tenant_id=tenant["tenant_id"],
        email="tenant-check@example.com",
        role="member",
        invited_by=invitation_users["owner"]["user_id"],
        token_hash=hash_invitation_token(
            generate_invitation_token()
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    with get_connection() as conn:
        recorded_tenant = conn.execute(
            """
            SELECT tenant_id
            FROM tenant_invitations
            WHERE invitation_id = %s;
            """,
            (result["invitation_id"],),
        ).fetchone()

    assert recorded_tenant[0] == tenant["tenant_id"]
    assert result["tenant_id"] == tenant["tenant_id"]

def test_existing_member_cannot_be_invited_again(
    invitation_users,
    tenant,
):
    existing_member_email = invitation_users["member"]["email"]

    with pytest.raises(
        ValueError,
        match="already an active tenant member",
    ):
        create_invitation(
            tenant_id=tenant["tenant_id"],
            email=existing_member_email,
            role="member",
            invited_by=invitation_users["owner"]["user_id"],
            token_hash=hash_invitation_token(
                generate_invitation_token()
            ),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )

def test_existing_member_can_be_invited_to_another_tenant(
    invitation_users,
    tenant,
):
    # Create a second tenant.
    other_tenant = create_tenant(
        name="Another Invitation Tenant",
        slug=f"another-invitation-{uuid.uuid4().hex}",
    )

    # The inviter must be an active owner of the second tenant.
    create_membership(
        other_tenant["tenant_id"],
        invitation_users["owner"]["user_id"],
        "owner",
    )

    try:
        # Invite a user who already belongs to the original tenant.
        result = create_invitation(
            tenant_id=other_tenant["tenant_id"],
            email=invitation_users["member"]["email"],
            role="member",
            invited_by=invitation_users["owner"]["user_id"],
            token_hash=hash_invitation_token(
                generate_invitation_token()
            ),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )

        # Verify the invitation belongs to the second tenant.
        assert result["tenant_id"] == other_tenant["tenant_id"]
        assert result["email"] == invitation_users["member"]["email"]

    finally:
        # Clean up all records belonging to the second tenant.
        with get_connection() as conn:
            conn.execute(
                """
                DELETE FROM tenant_invitations
                WHERE tenant_id = %s;
                """,
                (other_tenant["tenant_id"],),
            )

            conn.execute(
                """
                DELETE FROM tenant_memberships
                WHERE tenant_id = %s;
                """,
                (other_tenant["tenant_id"],),
            )

            conn.execute(
                """
                DELETE FROM tenants
                WHERE tenant_id = %s;
                """,
                (other_tenant["tenant_id"],),
            )

# =========================================================
# TENANT INVITATION ACCEPTANCE TESTS
# =========================================================

def test_user_can_accept_valid_invitation(
    pending_invitation,
    invited_user,
    tenant,
):
    result = accept_invitation(
        token_hash=pending_invitation["token_hash"],
        user_id=invited_user["user_id"],
    )

    # The invitation should be marked as accepted.
    assert result["invitation_id"] == (
        pending_invitation["invitation"]["invitation_id"]
    )
    assert result["tenant_id"] == tenant["tenant_id"]
    assert result["email"] == pending_invitation["email"]
    assert result["role"] == "member"
    assert result["accepted_at"] is not None

    # The user should now have an active membership.
    membership = get_active_membership(
        tenant["tenant_id"],
        invited_user["user_id"],
    )

    assert membership is not None
    assert membership["is_active"] is True
    assert membership["role"] == "member"

    # Sensitive token hashes must not be returned.
    assert "token_hash" not in result

def test_user_cannot_accept_invitation_for_different_email(
    pending_invitation,
    tenant,
):
    different_user = create_user(
        f"wrong-user-{uuid.uuid4().hex}@example.com",
        "test-password-hash",
    )

    try:
        with pytest.raises(
            PermissionError,
            match="Invitation email does not match user email",
        ):
            accept_invitation(
                token_hash=pending_invitation["token_hash"],
                user_id=different_user["user_id"],
            )

        # No membership should have been created.
        membership = get_active_membership(
            tenant["tenant_id"],
            different_user["user_id"],
        )

        assert membership is None

        # The invitation must remain pending.
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT accepted_at
                FROM tenant_invitations
                WHERE invitation_id = %s;
                """,
                (pending_invitation["invitation"]["invitation_id"],),
            ).fetchone()

        assert row[0] is None

    finally:
        with get_connection() as conn:
            conn.execute(
                """
                DELETE FROM tenant_memberships
                WHERE user_id = %s;
                """,
                (different_user["user_id"],),
            )
            conn.execute(
                """
                DELETE FROM users
                WHERE user_id = %s;
                """,
                (different_user["user_id"],),
            )

def test_accept_invitation_rejects_unknown_token(invited_user):
    unknown_token_hash = hash_invitation_token(
        generate_invitation_token()
    )

    with pytest.raises(
        ValueError,
        match="Invitation not found",
    ):
        accept_invitation(
            token_hash=unknown_token_hash,
            user_id=invited_user["user_id"],
        )

def test_accept_invitation_rejects_expired_invitation(
    pending_invitation,
    invited_user,
):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE tenant_invitations
            SET expires_at = NOW() - INTERVAL '1 hour'
            WHERE invitation_id = %s;
            """,
            (pending_invitation["invitation"]["invitation_id"],),
        )

    with pytest.raises(
        ValueError,
        match="Invitation has expired",
    ):
        accept_invitation(
            token_hash=pending_invitation["token_hash"],
            user_id=invited_user["user_id"],
        )

def test_accept_invitation_rejects_revoked_invitation(
    pending_invitation,
    invited_user,
):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE tenant_invitations
            SET revoked_at = NOW()
            WHERE invitation_id = %s;
            """,
            (pending_invitation["invitation"]["invitation_id"],),
        )

    with pytest.raises(
        ValueError,
        match="Invitation has been revoked",
    ):
        accept_invitation(
            token_hash=pending_invitation["token_hash"],
            user_id=invited_user["user_id"],
        )

def test_accept_invitation_cannot_be_accepted_twice(
    pending_invitation,
    invited_user,
):
    # First acceptance should succeed.
    accept_invitation(
        token_hash=pending_invitation["token_hash"],
        user_id=invited_user["user_id"],
    )

    # Reusing the same invitation must fail.
    with pytest.raises(
        ValueError,
        match="Invitation has already been accepted",
    ):
        accept_invitation(
            token_hash=pending_invitation["token_hash"],
            user_id=invited_user["user_id"],
        )

def test_accept_invitation_reactivates_inactive_membership(
    pending_invitation,
    invited_user,
    tenant,
):
    # Create an inactive membership for the invited user.
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO tenant_memberships (
                tenant_id,
                user_id,
                role,
                is_active
            )
            VALUES (%s, %s, 'admin', FALSE);
            """,
            (tenant["tenant_id"], invited_user["user_id"]),
        )

    result = accept_invitation(
        token_hash=pending_invitation["token_hash"],
        user_id=invited_user["user_id"],
    )

    membership = get_active_membership(
        tenant["tenant_id"],
        invited_user["user_id"],
    )

    assert membership is not None
    assert membership["is_active"] is True
    assert membership["role"] == "member"
    assert result["role"] == "member"
    assert result["accepted_at"] is not None

def test_accept_invitation_rejects_existing_active_membership(
    pending_invitation,
    invited_user,
    tenant,
):
    # Simulate the user becoming an active member after the invitation
    # was created.
    create_membership(
        tenant["tenant_id"],
        invited_user["user_id"],
        role="admin",
    )

    with pytest.raises(
        ValueError,
        match="User is already an active member",
    ):
        accept_invitation(
            token_hash=pending_invitation["token_hash"],
            user_id=invited_user["user_id"],
        )

    # The existing membership must remain unchanged.
    membership = get_active_membership(
        tenant["tenant_id"],
        invited_user["user_id"],
    )

    assert membership is not None
    assert membership["role"] == "admin"

    # The invitation must remain pending.
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT accepted_at
            FROM tenant_invitations
            WHERE invitation_id = %s;
            """,
            (pending_invitation["invitation"]["invitation_id"],),
        ).fetchone()

    assert row[0] is None

def test_inactive_user_cannot_accept_invitation(
    pending_invitation,
    invited_user,
    tenant,
):
    # Deactivate the invited user's account.
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET is_active = FALSE
            WHERE user_id = %s;
            """,
            (invited_user["user_id"],),
        )

    with pytest.raises(
        PermissionError,
        match="User account is inactive",
    ):
        accept_invitation(
            token_hash=pending_invitation["token_hash"],
            user_id=invited_user["user_id"],
        )

    # No membership should have been created.
    membership = get_active_membership(
        tenant["tenant_id"],
        invited_user["user_id"],
    )

    assert membership is None

    # The invitation must remain pending.
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT accepted_at
            FROM tenant_invitations
            WHERE invitation_id = %s;
            """,
            (pending_invitation["invitation"]["invitation_id"],),
        ).fetchone()

    assert row[0] is None

def test_accept_invitation_rejects_nonexistent_user(
    pending_invitation,
):
    nonexistent_user_id = uuid.uuid4()

    with pytest.raises(
        ValueError,
        match="User not found",
    ):
        accept_invitation(
            token_hash=pending_invitation["token_hash"],
            user_id=nonexistent_user_id,
        )