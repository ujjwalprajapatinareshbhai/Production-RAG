BEGIN;

-- Safety check: every existing document must have a tenant.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM documents
        WHERE tenant_id IS NULL
    ) THEN
        RAISE EXCEPTION
            'Migration stopped: documents with NULL tenant_id exist.';
    END IF;
END $$;

-- Replace the old single-column primary key.
ALTER TABLE documents
    DROP CONSTRAINT documents_pkey;

-- Documents are now uniquely identified within each tenant.
ALTER TABLE documents
    ADD CONSTRAINT documents_pkey
    PRIMARY KEY (tenant_id, document_id);

COMMIT;