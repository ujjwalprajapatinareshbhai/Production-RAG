-- Migration 003: Tenant Invitations
--
-- Stores secure, one-time invitations for users to join a tenant.
-- Invitation tokens are stored as SHA-256 hashes, never as plaintext.

CREATE TABLE IF NOT EXISTS tenant_invitations (
    invitation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    tenant_id UUID NOT NULL
        REFERENCES tenants(tenant_id)
        ON DELETE CASCADE,

    email TEXT NOT NULL,

    role TEXT NOT NULL
        CHECK (role IN ('admin', 'member')),

    token_hash TEXT NOT NULL UNIQUE,

    invited_by UUID NOT NULL
        REFERENCES users(user_id)
        ON DELETE RESTRICT,

    expires_at TIMESTAMPTZ NOT NULL,

    accepted_at TIMESTAMPTZ,

    revoked_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (
        accepted_at IS NULL
        OR revoked_at IS NULL
    )
);

-- Prevent multiple outstanding invitations for the same
-- tenant and email address.
CREATE UNIQUE INDEX IF NOT EXISTS
    tenant_invitations_pending_email_unique
ON tenant_invitations (
    tenant_id,
    lower(email)
)
WHERE accepted_at IS NULL
  AND revoked_at IS NULL;

-- Supports tenant-scoped invitation lookups.
CREATE INDEX IF NOT EXISTS
    tenant_invitations_tenant_created_idx
ON tenant_invitations (
    tenant_id,
    created_at DESC
);