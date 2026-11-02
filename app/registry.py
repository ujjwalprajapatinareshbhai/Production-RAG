from uuid import UUID

from psycopg.rows import dict_row

from app.db import get_connection


VALID_STATUSES = {
    "pending",
    "processing",
    "completed",
    "failed",
}


def initialize_registry() -> None:
    """Create the documents table and apply schema migrations."""

    create_table_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY
                CHECK (length(document_id) = 64),

            filename TEXT NOT NULL,

            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (
                    status IN (
                        'pending',
                        'processing',
                        'completed',
                        'failed'
                    )
                ),

            error_message TEXT,

            cleanup_pending BOOLEAN NOT NULL DEFAULT FALSE,
            cleanup_error TEXT,
            cleanup_claimed_at TIMESTAMPTZ,
            cleanup_claim_token UUID,

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """

    # Add columns to databases where the table already exists.
    # IF NOT EXISTS makes these migrations safe to run repeatedly.
    alter_table_sql = """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS cleanup_pending
            BOOLEAN NOT NULL DEFAULT FALSE;

        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS cleanup_error TEXT;

        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS cleanup_claimed_at TIMESTAMPTZ;

        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS cleanup_claim_token UUID;
    """

    with get_connection() as conn:
        conn.execute(create_table_sql)
        conn.execute(alter_table_sql)


def register_document(
    document_id: str,
    filename: str,
) -> dict:
    """Register a PDF, without resetting an existing document's status."""

    if len(document_id) != 64:
        raise ValueError(
            "document_id must be a 64-character SHA-256 hash."
        )

    if not filename.strip():
        raise ValueError("filename cannot be empty.")

    sql = """
        INSERT INTO documents (document_id, filename)
        VALUES (%s, %s)

        ON CONFLICT (document_id)
        DO UPDATE SET
            filename = EXCLUDED.filename,
            updated_at = NOW()

        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (document_id, filename))
            return cursor.fetchone()


def get_document(document_id: str) -> dict | None:
    """Retrieve a registered document by its SHA-256 hash."""

    sql = """
        SELECT *
        FROM documents
        WHERE document_id = %s;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (document_id,))
            return cursor.fetchone()


def update_document_status(
    document_id: str,
    status: str,
    error_message: str | None = None,
) -> dict | None:
    """Update processing status and optionally record an error."""

    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status: {status}. "
            f"Allowed statuses: {sorted(VALID_STATUSES)}"
        )

    sql = """
        UPDATE documents
        SET
            status = %s,
            error_message = %s,
            updated_at = NOW()
        WHERE document_id = %s
        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql,
                (status, error_message, document_id),
            )
            return cursor.fetchone()


def list_completed_document_ids() -> list[str]:
    """Return document IDs whose ingestion status is completed."""

    sql = """
        SELECT document_id
        FROM documents
        WHERE status = 'completed'
        ORDER BY document_id;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()

    return [row[0] for row in rows]


def list_documents() -> list[dict]:
    """Return all registered documents, newest first."""

    sql = """
        SELECT
            document_id,
            filename,
            status,
            error_message,
            cleanup_pending,
            cleanup_error,
            cleanup_claimed_at,
            cleanup_claim_token,
            created_at,
            updated_at
        FROM documents
        ORDER BY created_at DESC, document_id;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql)
            return cursor.fetchall()


def delete_document_record(document_id: str) -> bool:
    """Delete a document registry record."""

    sql = """
        DELETE FROM documents
        WHERE document_id = %s
        RETURNING document_id;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (document_id,))
            return cursor.fetchone() is not None


def list_stale_documents(
    stale_after_minutes: int = 30,
) -> list[dict]:
    """Return pending or processing documents older than a cutoff."""

    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be at least 1.")

    sql = """
        SELECT
            document_id,
            filename,
            status,
            error_message,
            cleanup_pending,
            cleanup_error,
            cleanup_claimed_at,
            cleanup_claim_token,
            created_at,
            updated_at
        FROM documents
        WHERE status IN ('pending', 'processing')
          AND updated_at < NOW() - (%s * INTERVAL '1 minute')
        ORDER BY updated_at ASC;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (stale_after_minutes,))
            return cursor.fetchall()


def update_document_heartbeat(document_id: str) -> bool:
    """Refresh the heartbeat for a document currently being processed."""

    sql = """
        UPDATE documents
        SET updated_at = NOW()
        WHERE document_id = %s
          AND status = 'processing'
        RETURNING document_id;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (document_id,))
            return cursor.fetchone() is not None


def mark_stale_documents_failed(
    stale_after_minutes: int = 30,
) -> list[dict]:
    """
    Mark stale pending and processing documents as failed.

    Also marks their vector cleanup as pending so a later recovery
    process can retry cleanup if Qdrant is unavailable.
    """

    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be at least 1.")

    sql = """
        UPDATE documents
        SET
            status = 'failed',
            error_message = 'Document ingestion became stale.',
            cleanup_pending = TRUE,
            cleanup_error = NULL,
            cleanup_claimed_at = NULL,
            cleanup_claim_token = NULL,
            updated_at = NOW()
        WHERE status IN ('pending', 'processing')
          AND updated_at < NOW() - (%s * INTERVAL '1 minute')
        RETURNING
            document_id,
            filename,
            status,
            error_message,
            cleanup_pending,
            cleanup_error,
            cleanup_claimed_at,
            cleanup_claim_token,
            created_at,
            updated_at;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (stale_after_minutes,))
            return cursor.fetchall()


def list_pending_cleanup_documents() -> list[dict]:
    """Return documents that still require vector cleanup."""

    sql = """
        SELECT
            document_id,
            filename,
            status,
            cleanup_pending,
            cleanup_error,
            cleanup_claimed_at,
            cleanup_claim_token,
            created_at,
            updated_at
        FROM documents
        WHERE cleanup_pending = TRUE
        ORDER BY updated_at ASC;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql)
            return cursor.fetchall()


def mark_document_cleanup_completed(
    document_id: str,
    claim_token: UUID,
) -> dict | None:
    """
    Mark cleanup as completed only if the supplied token owns the lease.

    A worker with an expired or replaced token cannot complete the task.
    """

    sql = """
        UPDATE documents
        SET
            cleanup_pending = FALSE,
            cleanup_error = NULL,
            cleanup_claimed_at = NULL,
            cleanup_claim_token = NULL,
            updated_at = NOW()
        WHERE document_id = %s
          AND cleanup_pending = TRUE
          AND cleanup_claim_token = %s
        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (document_id, claim_token))
            return cursor.fetchone()


def mark_document_cleanup_failed(
    document_id: str,
    error_message: str,
    claim_token: UUID,
) -> dict | None:
    """
    Record cleanup failure only if the supplied token owns the lease.

    Keeps cleanup pending so a future recovery run can retry it.
    """

    if not error_message.strip():
        raise ValueError("error_message cannot be empty.")

    sql = """
        UPDATE documents
        SET
            cleanup_pending = TRUE,
            cleanup_error = %s,
            cleanup_claimed_at = NULL,
            cleanup_claim_token = NULL,
            updated_at = NOW()
        WHERE document_id = %s
          AND cleanup_pending = TRUE
          AND cleanup_claim_token = %s
        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql,
                (error_message, document_id, claim_token),
            )
            return cursor.fetchone()


def claim_pending_cleanup_documents(
    limit: int = 10,
    lease_minutes: int = 15,
) -> list[dict]:
    """
    Atomically claim pending cleanup tasks.

    A task can be claimed when it has no active lease or
    its previous lease has expired.

    Every claim receives a new UUID fencing token.
    """

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    if lease_minutes < 1:
        raise ValueError("lease_minutes must be at least 1.")

    sql = """
        WITH candidates AS (
            SELECT document_id
            FROM documents
            WHERE cleanup_pending = TRUE
              AND (
                  cleanup_claimed_at IS NULL
                  OR cleanup_claimed_at
                     < NOW() - (%s * INTERVAL '1 minute')
              )
            ORDER BY updated_at ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        )
        UPDATE documents AS d
        SET
            cleanup_claimed_at = NOW(),
            cleanup_claim_token = gen_random_uuid()
        FROM candidates
        WHERE d.document_id = candidates.document_id
        RETURNING d.*;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (lease_minutes, limit))
            return cursor.fetchall()