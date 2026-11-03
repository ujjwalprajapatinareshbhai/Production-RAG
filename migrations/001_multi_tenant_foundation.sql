BEGIN;

-- =====================================================
-- 1. Create tenants
-- =====================================================

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


-- =====================================================
-- 2. Create users
-- =====================================================

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

CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_unique
    ON users (lower(email));


-- =====================================================
-- 3. Create tenant memberships
-- =====================================================

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

CREATE INDEX IF NOT EXISTS tenant_memberships_user_idx
    ON tenant_memberships (user_id);


-- =====================================================
-- 4. Create the legacy workspace
-- =====================================================

INSERT INTO tenants (name, slug)
VALUES ('Legacy Workspace', 'legacy-workspace')
ON CONFLICT (slug) DO NOTHING;


-- =====================================================
-- 5. Add tenant_id to existing documents
-- =====================================================

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS tenant_id UUID;

UPDATE documents
SET tenant_id = (
    SELECT tenant_id
    FROM tenants
    WHERE slug = 'legacy-workspace'
)
WHERE tenant_id IS NULL;


-- =====================================================
-- 6. Add tenant foreign key
-- =====================================================

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


-- =====================================================
-- 7. Make tenant_id mandatory
-- =====================================================

ALTER TABLE documents
ALTER COLUMN tenant_id SET NOT NULL;


-- =====================================================
-- 8. Add tenant-scoped document index
-- =====================================================

CREATE INDEX IF NOT EXISTS documents_tenant_created_idx
    ON documents (tenant_id, created_at DESC);


COMMIT;