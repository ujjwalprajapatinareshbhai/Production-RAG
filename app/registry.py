from datetime import datetime, timezone
from uuid import UUID, uuid4
from psycopg.rows import dict_row

from app.db import get_connection


VALID_STATUSES = {
    "pending",
    "processing",
    "completed",
    "failed",
}

VALID_ROLES = {
    "owner",
    "admin",
    "member",
}

INVITATION_ROLES = {
    "admin",
    "member",
}


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def initialize_registry() -> None:
    """
    Initialize the multi-tenant PostgreSQL schema.

    Safe to run repeatedly against an existing database.
    Existing documents are assigned to Legacy Workspace
    if they do not already have a tenant.
    """

    with get_connection() as conn:

        # -------------------------------------------------
        # 1. Tenants
        # -------------------------------------------------

        conn.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL,
                slug TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                CONSTRAINT tenants_name_not_empty
                    CHECK (length(trim(name)) > 0),

                CONSTRAINT tenants_slug_not_empty
                    CHECK (length(trim(slug)) > 0)
            );
        """)

        # -------------------------------------------------
        # 2. Users
        # -------------------------------------------------

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                CONSTRAINT users_email_not_empty
                    CHECK (length(trim(email)) > 0),

                CONSTRAINT users_password_hash_not_empty
                    CHECK (length(trim(password_hash)) > 0)
            );
        """)

        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_unique
            ON users (lower(email));
        """)

        # -------------------------------------------------
        # 3. Tenant memberships
        # -------------------------------------------------

        conn.execute("""
            CREATE TABLE IF NOT EXISTS tenant_memberships (
                tenant_id UUID NOT NULL,
                user_id UUID NOT NULL,

                role TEXT NOT NULL DEFAULT 'member',
                is_active BOOLEAN NOT NULL DEFAULT TRUE,

                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                PRIMARY KEY (tenant_id, user_id),

                CONSTRAINT tenant_memberships_tenant_fk
                    FOREIGN KEY (tenant_id)
                    REFERENCES tenants (tenant_id)
                    ON DELETE CASCADE,

                CONSTRAINT tenant_memberships_user_fk
                    FOREIGN KEY (user_id)
                    REFERENCES users (user_id)
                    ON DELETE CASCADE,

                CONSTRAINT tenant_memberships_role_check
                    CHECK (role IN ('owner', 'admin', 'member'))
            );
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS tenant_memberships_user_idx
            ON tenant_memberships (user_id);
        """)

        # -------------------------------------------------
        # 4. Documents table
        # -------------------------------------------------

        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT NOT NULL
                    CHECK (length(document_id) = 64),

                tenant_id UUID NOT NULL,

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
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                PRIMARY KEY (tenant_id, document_id),

                CONSTRAINT documents_tenant_fk
                    FOREIGN KEY (tenant_id)
                    REFERENCES tenants (tenant_id)
                    ON DELETE RESTRICT
            );
        """)

        # -------------------------------------------------
        # 5. Add missing columns to existing databases
        # -------------------------------------------------

        conn.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS tenant_id UUID;
        """)

        conn.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS cleanup_pending
                BOOLEAN NOT NULL DEFAULT FALSE;

            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS cleanup_error TEXT;

            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS cleanup_claimed_at TIMESTAMPTZ;

            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS cleanup_claim_token UUID;
        """)

        # -------------------------------------------------
        # 6. Create the legacy workspace
        # -------------------------------------------------

        conn.execute("""
            INSERT INTO tenants (name, slug)
            VALUES ('Legacy Workspace', 'legacy-workspace')
            ON CONFLICT (slug) DO NOTHING;
        """)

        # -------------------------------------------------
        # 7. Assign existing documents to Legacy Workspace
        # -------------------------------------------------

        conn.execute("""
            UPDATE documents
            SET tenant_id = (
                SELECT tenant_id
                FROM tenants
                WHERE slug = 'legacy-workspace'
            )
            WHERE tenant_id IS NULL;
        """)

        # -------------------------------------------------
        # 8. Ensure tenant foreign key exists
        # -------------------------------------------------

        conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'documents_tenant_fk'
                      AND conrelid = 'documents'::regclass
                ) THEN
                    ALTER TABLE documents
                    ADD CONSTRAINT documents_tenant_fk
                    FOREIGN KEY (tenant_id)
                    REFERENCES tenants (tenant_id)
                    ON DELETE RESTRICT;
                END IF;
            END $$;
        """)

        # -------------------------------------------------
        # 9. Ensure tenant_id is mandatory
        # -------------------------------------------------

        conn.execute("""
            ALTER TABLE documents
            ALTER COLUMN tenant_id SET NOT NULL;
        """)

        # -------------------------------------------------
        # 10. Ensure composite primary key
        # -------------------------------------------------

        conn.execute("""
            DO $$
            DECLARE
                current_pk_columns TEXT[];
            BEGIN
                SELECT array_agg(a.attname ORDER BY k.ordinality)
                INTO current_pk_columns
                FROM pg_constraint c
                CROSS JOIN LATERAL
                    unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality)
                JOIN pg_attribute a
                    ON a.attrelid = c.conrelid
                   AND a.attnum = k.attnum
                WHERE c.conrelid = 'documents'::regclass
                  AND c.contype = 'p';

                IF current_pk_columns IS DISTINCT FROM
                   ARRAY['tenant_id', 'document_id']::TEXT[] THEN

                    IF EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE contype = 'f'
                          AND confrelid = 'documents'::regclass
                    ) THEN
                        RAISE EXCEPTION
                            'Cannot replace documents primary key: '
                            'referencing foreign keys exist.';
                    END IF;

                    IF current_pk_columns IS NOT NULL THEN
                        ALTER TABLE documents
                        DROP CONSTRAINT documents_pkey;
                    END IF;

                    ALTER TABLE documents
                    ADD CONSTRAINT documents_pkey
                    PRIMARY KEY (tenant_id, document_id);
                END IF;
            END $$;
        """)

        # -------------------------------------------------
        # 11. Tenant-aware document index
        # -------------------------------------------------

        conn.execute("""
            CREATE INDEX IF NOT EXISTS documents_tenant_created_idx
            ON documents (tenant_id, created_at DESC);
        """)


# =========================================================
# TENANT OPERATIONS
# =========================================================

def create_tenant(
    name: str,
    slug: str,
) -> dict:

    if not name or not name.strip():
        raise ValueError("Tenant name cannot be empty.")

    if not slug or not slug.strip():
        raise ValueError("Tenant slug cannot be empty.")

    sql = """
        INSERT INTO tenants (name, slug)
        VALUES (%s, %s)
        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (name.strip(), slug.strip()))
            return cursor.fetchone()


def get_tenant(tenant_id: UUID) -> dict | None:

    sql = """
        SELECT *
        FROM tenants
        WHERE tenant_id = %s;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (tenant_id,))
            return cursor.fetchone()


# =========================================================
# USER OPERATIONS
# =========================================================

def create_user(
    email: str,
    password_hash: str,
) -> dict:

    if not email or not email.strip():
        raise ValueError("Email cannot be empty.")

    if not password_hash:
        raise ValueError("Password hash cannot be empty.")

    sql = """
        INSERT INTO users (email, password_hash)
        VALUES (%s, %s)
        RETURNING user_id, email, is_active, created_at;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql,
                (email.strip().lower(), password_hash),
            )
            return cursor.fetchone()


def get_user_by_email(email: str) -> dict | None:

    sql = """
        SELECT *
        FROM users
        WHERE lower(email) = lower(%s);
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (email,))
            return cursor.fetchone()


def get_user_by_id(user_id: UUID) -> dict | None:

    sql = """
        SELECT *
        FROM users
        WHERE user_id = %s;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (user_id,))
            return cursor.fetchone()


# =========================================================
# MEMBERSHIP OPERATIONS
# =========================================================

def create_membership(
    tenant_id: UUID,
    user_id: UUID,
    role: str = "member",
) -> dict:

    if role not in VALID_ROLES:
        raise ValueError(
            f"Invalid role: {role}. "
            f"Allowed roles: {sorted(VALID_ROLES)}"
        )

    sql = """
        INSERT INTO tenant_memberships (
            tenant_id,
            user_id,
            role
        )
        VALUES (%s, %s, %s)
        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql,
                (tenant_id, user_id, role),
            )
            return cursor.fetchone()

def bootstrap_first_owner(
    tenant_slug: str,
    email: str,
    password_hash: str,
) -> dict:
    """
    Create the first owner of a tenant.

    User creation and owner membership happen in one
    PostgreSQL transaction.
    """

    if not tenant_slug or not tenant_slug.strip():
        raise ValueError("Tenant slug cannot be empty.")

    if not email or not email.strip():
        raise ValueError("Email cannot be empty.")

    if not password_hash:
        raise ValueError("Password hash cannot be empty.")

    normalized_email = email.strip().lower()

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:

            # 1. Lock the tenant row to prevent concurrent bootstrap.
            cursor.execute(
                """
                SELECT tenant_id, name, slug
                FROM tenants
                WHERE slug = %s
                FOR UPDATE;
                """,
                (tenant_slug.strip(),),
            )

            tenant = cursor.fetchone()

            if tenant is None:
                raise ValueError("Tenant does not exist.")

            # 2. Refuse bootstrap if an owner already exists.
            cursor.execute(
                """
                SELECT user_id
                FROM tenant_memberships
                WHERE tenant_id = %s
                  AND role = 'owner'
                LIMIT 1;
                """,
                (tenant["tenant_id"],),
            )

            if cursor.fetchone() is not None:
                raise ValueError(
                    "This tenant already has an owner. "
                    "Bootstrap is not allowed."
                )

            # 3. Create the user.
            cursor.execute(
                """
                INSERT INTO users (email, password_hash)
                VALUES (%s, %s)
                RETURNING user_id, email, is_active, created_at;
                """,
                (normalized_email, password_hash),
            )

            user = cursor.fetchone()

            # 4. Create the owner's tenant membership.
            cursor.execute(
                """
                INSERT INTO tenant_memberships (
                    tenant_id,
                    user_id,
                    role
                )
                VALUES (%s, %s, 'owner')
                RETURNING tenant_id, user_id, role, is_active;
                """,
                (tenant["tenant_id"], user["user_id"]),
            )

            membership = cursor.fetchone()

            return {
                "tenant": tenant,
                "user": user,
                "membership": membership,
            }


def get_active_membership(
    tenant_id: UUID,
    user_id: UUID,
) -> dict | None:

    sql = """
        SELECT
            tenant_id,
            user_id,
            role,
            is_active,
            created_at
        FROM tenant_memberships
        WHERE tenant_id = %s
          AND user_id = %s
          AND is_active = TRUE;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (tenant_id, user_id))
            return cursor.fetchone()

def create_invitation(
    tenant_id: UUID,
    email: str,
    role: str,
    invited_by: UUID,
    token_hash: str,
    expires_at: datetime,
) -> dict:
    """
    Create a tenant invitation.

    Owners can invite admins and members.
    Admins can invite members only.

    Only the SHA-256 token hash is stored.
    """

    # 1. Validate email.
    if not email or not email.strip():
        raise ValueError("Email cannot be empty.")

    normalized_email = email.strip().lower()

    # 2. Validate invitation role.
    if role not in INVITATION_ROLES:
        raise ValueError(f"Invalid invitation role: {role}.")

    # 3. Validate token hash.
    if (
        not token_hash
        or len(token_hash) != 64
        or not all(c in "0123456789abcdef" for c in token_hash.lower())
    ):
        raise ValueError("Invalid invitation token hash.")

    # 4. Validate expiry timestamp.
    if (
        expires_at.tzinfo is None
        or expires_at.utcoffset() is None
    ):
        raise ValueError("Invitation expiry must be timezone-aware.")

    if expires_at <= datetime.now(timezone.utc):
        raise ValueError("Invitation expiry must be in the future.")

    # 5. Start database transaction.
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:

            # 6. Lock and verify inviter membership.
            cursor.execute(
                """
                SELECT role
                FROM tenant_memberships
                WHERE tenant_id = %s
                  AND user_id = %s
                  AND is_active = TRUE
                FOR UPDATE;
                """,
                (tenant_id, invited_by),
            )

            inviter = cursor.fetchone()

            if inviter is None:
                raise PermissionError(
                    "Active tenant membership required."
                )

            inviter_role = inviter["role"]

            # 7. Enforce invitation permissions.
            if inviter_role not in {"owner", "admin"}:
                raise PermissionError(
                    "Only owners and admins can invite users."
                )

            if role == "admin" and inviter_role != "owner":
                raise PermissionError(
                    "Only owners can invite administrators."
                )

            # 8. Revoke expired outstanding invitations.
            # This prevents the unique pending-email index
            # from blocking a new invitation.
            cursor.execute(
                """
                UPDATE tenant_invitations
                SET revoked_at = NOW()
                WHERE tenant_id = %s
                  AND lower(email) = %s
                  AND accepted_at IS NULL
                  AND revoked_at IS NULL
                  AND expires_at <= NOW();
                """,
                (tenant_id, normalized_email),
            )

            # 9. Prevent invitations to existing active members.
            cursor.execute(
                """
                SELECT 1
                FROM users AS u
                JOIN tenant_memberships AS tm
                  ON tm.user_id = u.user_id
                WHERE tm.tenant_id = %s
                  AND lower(u.email) = %s
                  AND tm.is_active = TRUE
                LIMIT 1;
                """,
                (tenant_id, normalized_email),
            )

            if cursor.fetchone() is not None:
                raise ValueError(
                    "This user is already an active tenant member."
                )

            # 10. Insert the invitation.
            cursor.execute(
                """
                INSERT INTO tenant_invitations (
                    tenant_id,
                    email,
                    role,
                    token_hash,
                    invited_by,
                    expires_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING
                    invitation_id,
                    tenant_id,
                    email,
                    role,
                    invited_by,
                    expires_at,
                    created_at;
                """,
                (
                    tenant_id,
                    normalized_email,
                    role,
                    token_hash,
                    invited_by,
                    expires_at,
                ),
            )

            return cursor.fetchone()

def accept_invitation(
    token_hash: str,
    user_id: UUID,
) -> dict:
    """
    Accept a tenant invitation.

    Validates the invitation and authenticated user's email,
    creates or reactivates membership, and marks the invitation
    accepted in one database transaction.
    """

    if not token_hash:
        raise ValueError("Invitation token hash cannot be empty.")

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:

            # 1. Find and lock the invitation.
            cursor.execute(
                """
                SELECT
                    invitation_id,
                    tenant_id,
                    email,
                    role,
                    expires_at,
                    accepted_at,
                    revoked_at
                FROM tenant_invitations
                WHERE token_hash = %s
                FOR UPDATE;
                """,
                (token_hash,),
            )

            invitation = cursor.fetchone()

            if invitation is None:
                raise ValueError("Invitation not found.")

            # 2. Check invitation state.
            if invitation["accepted_at"] is not None:
                raise ValueError(
                    "Invitation has already been accepted."
                )

            if invitation["revoked_at"] is not None:
                raise ValueError(
                    "Invitation has been revoked."
                )

            if invitation["expires_at"] <= datetime.now(timezone.utc):
                raise ValueError(
                    "Invitation has expired."
                )

            # 3. Find and lock the authenticated user.
            cursor.execute(
                """
                SELECT email, is_active
                FROM users
                WHERE user_id = %s
                FOR UPDATE;
                """,
                (user_id,),
            )

            user = cursor.fetchone()

            if user is None:
                raise ValueError("User not found.")

            if not user["is_active"]:
                raise PermissionError(
                    "User account is inactive."
                )

            # 4. Verify the actual database email.
            if (
                user["email"].strip().lower()
                != invitation["email"].strip().lower()
            ):
                raise PermissionError(
                    "Invitation email does not match user email."
                )

            # 5. Check for an existing membership.
            cursor.execute(
                """
                SELECT role, is_active
                FROM tenant_memberships
                WHERE tenant_id = %s
                  AND user_id = %s
                FOR UPDATE;
                """,
                (
                    invitation["tenant_id"],
                    user_id,
                ),
            )

            membership = cursor.fetchone()

            if membership is not None and membership["is_active"]:
                raise ValueError(
                    "User is already an active member."
                )

            # 6. Create or reactivate the membership.
            if membership is not None:
                cursor.execute(
                    """
                    UPDATE tenant_memberships
                    SET
                        role = %s,
                        is_active = TRUE
                    WHERE tenant_id = %s
                      AND user_id = %s;
                    """,
                    (
                        invitation["role"],
                        invitation["tenant_id"],
                        user_id,
                    ),
                )

            else:
                cursor.execute(
                    """
                    INSERT INTO tenant_memberships (
                        tenant_id,
                        user_id,
                        role,
                        is_active
                    )
                    VALUES (%s, %s, %s, TRUE);
                    """,
                    (
                        invitation["tenant_id"],
                        user_id,
                        invitation["role"],
                    ),
                )

            # 7. Mark the invitation as accepted.
            cursor.execute(
                """
                UPDATE tenant_invitations
                SET accepted_at = NOW()
                WHERE invitation_id = %s
                RETURNING accepted_at;
                """,
                (invitation["invitation_id"],),
            )

            accepted = cursor.fetchone()

            # 8. Return safe invitation metadata.
            return {
                "invitation_id": invitation["invitation_id"],
                "tenant_id": invitation["tenant_id"],
                "email": invitation["email"],
                "role": invitation["role"],
                "accepted_at": accepted["accepted_at"],
            }
# =========================================================
# DOCUMENT REGISTRY
# =========================================================

def register_document(
    tenant_id: UUID,
    document_id: str,
    filename: str,
) -> dict:

    if len(document_id) != 64:
        raise ValueError(
            "document_id must be a 64-character SHA-256 hash."
        )

    if not filename or not filename.strip():
        raise ValueError("filename cannot be empty.")

    sql = """
        INSERT INTO documents (
            tenant_id,
            document_id,
            filename
        )
        VALUES (%s, %s, %s)

        ON CONFLICT (tenant_id, document_id)
        DO UPDATE SET
            filename = EXCLUDED.filename,
            updated_at = NOW()

        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql,
                (tenant_id, document_id, filename),
            )
            return cursor.fetchone()


def get_document(
    tenant_id: UUID,
    document_id: str,
) -> dict | None:

    sql = """
        SELECT *
        FROM documents
        WHERE tenant_id = %s
          AND document_id = %s;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (tenant_id, document_id))
            return cursor.fetchone()


def update_document_status(
    tenant_id: UUID,
    document_id: str,
    status: str,
    error_message: str | None = None,
) -> dict | None:

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
        WHERE tenant_id = %s
          AND document_id = %s
        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql,
                (
                    status,
                    error_message,
                    tenant_id,
                    document_id,
                ),
            )
            return cursor.fetchone()


def list_completed_document_ids(
    tenant_id: UUID,
) -> list[str]:

    sql = """
        SELECT document_id
        FROM documents
        WHERE tenant_id = %s
          AND status = 'completed'
        ORDER BY document_id;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (tenant_id,))
            rows = cursor.fetchall()

    return [row[0] for row in rows]


def list_documents(
    tenant_id: UUID,
) -> list[dict]:

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
        WHERE tenant_id = %s
        ORDER BY created_at DESC, document_id;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (tenant_id,))
            return cursor.fetchall()


def delete_document_record(
    tenant_id: UUID,
    document_id: str,
) -> bool:

    sql = """
        DELETE FROM documents
        WHERE tenant_id = %s
          AND document_id = %s
        RETURNING document_id;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (tenant_id, document_id))
            return cursor.fetchone() is not None


# =========================================================
# HEARTBEAT
# =========================================================

def update_document_heartbeat(
    tenant_id: UUID,
    document_id: str,
) -> bool:

    sql = """
        UPDATE documents
        SET updated_at = NOW()
        WHERE tenant_id = %s
          AND document_id = %s
          AND status = 'processing'
        RETURNING document_id;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (tenant_id, document_id))
            return cursor.fetchone() is not None


# =========================================================
# STALE DOCUMENT RECOVERY
# System-wide trusted background operations
# =========================================================

def list_stale_documents(
    stale_after_minutes: int = 30,
) -> list[dict]:

    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be at least 1.")

    sql = """
        SELECT *
        FROM documents
        WHERE status IN ('pending', 'processing')
          AND updated_at < NOW() - (%s * INTERVAL '1 minute')
        ORDER BY updated_at ASC;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (stale_after_minutes,))
            return cursor.fetchall()


def mark_stale_documents_failed(
    stale_after_minutes: int = 30,
) -> list[dict]:

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
        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (stale_after_minutes,))
            return cursor.fetchall()


# =========================================================
# PENDING CLEANUP
# =========================================================

def list_pending_cleanup_documents() -> list[dict]:

    sql = """
        SELECT *
        FROM documents
        WHERE cleanup_pending = TRUE
        ORDER BY updated_at ASC;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql)
            return cursor.fetchall()


def mark_document_cleanup_completed(
    tenant_id: UUID,
    document_id: str,
    claim_token: UUID,
) -> dict | None:

    sql = """
        UPDATE documents
        SET
            cleanup_pending = FALSE,
            cleanup_error = NULL,
            cleanup_claimed_at = NULL,
            cleanup_claim_token = NULL,
            updated_at = NOW()
        WHERE tenant_id = %s
          AND document_id = %s
          AND cleanup_pending = TRUE
          AND cleanup_claim_token = %s
        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql,
                (tenant_id, document_id, claim_token),
            )
            return cursor.fetchone()


def mark_document_cleanup_failed(
    tenant_id: UUID,
    document_id: str,
    error_message: str,
    claim_token: UUID,
) -> dict | None:

    if not error_message or not error_message.strip():
        raise ValueError("error_message cannot be empty.")

    sql = """
        UPDATE documents
        SET
            cleanup_pending = TRUE,
            cleanup_error = %s,
            cleanup_claimed_at = NULL,
            cleanup_claim_token = NULL,
            updated_at = NOW()
        WHERE tenant_id = %s
          AND document_id = %s
          AND cleanup_pending = TRUE
          AND cleanup_claim_token = %s
        RETURNING *;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql,
                (
                    error_message,
                    tenant_id,
                    document_id,
                    claim_token,
                ),
            )
            return cursor.fetchone()


def claim_pending_cleanup_documents(
    limit: int = 10,
    lease_minutes: int = 15,
) -> list[dict]:

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    if lease_minutes < 1:
        raise ValueError("lease_minutes must be at least 1.")

    sql = """
        WITH candidates AS (
            SELECT tenant_id, document_id
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
        WHERE d.tenant_id = candidates.tenant_id
          AND d.document_id = candidates.document_id
        RETURNING d.*;
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (lease_minutes, limit))
            return cursor.fetchall()