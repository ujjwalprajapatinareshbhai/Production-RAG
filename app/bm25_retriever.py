import re
from threading import RLock
from uuid import UUID

from rank_bm25 import BM25Okapi
from qdrant_client import models

from app.registry import list_completed_document_ids
from app.vector_store import COLLECTION_NAME, get_vector_client


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks_from_qdrant(tenant_id: UUID) -> list[dict]:
    """Load completed-document chunks belonging to one tenant."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    completed_ids = set(
        list_completed_document_ids(tenant_id)
    )

    if not completed_ids:
        return []

    client = get_vector_client()
    chunks = []
    offset = None

    # Restrict Qdrant scrolling to the requested tenant.
    tenant_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(
                    value=str(tenant_id)
                ),
            )
        ]
    )

    while True:
        response, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=tenant_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in response:
            payload = point.payload or {}

            document_id = payload.get("document_id")
            text = payload.get("text", "")

            # Only include completed documents for this tenant.
            if document_id not in completed_ids:
                continue

            if not text.strip():
                continue

            chunks.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "tenant_id": payload.get("tenant_id"),
                    "document_id": document_id,
                    "filename": payload.get("filename"),
                    "text": text,
                    "start_page": payload.get("start_page"),
                    "end_page": payload.get("end_page"),
                    "source_pages": payload.get("source_pages", []),
                }
            )

        if next_offset is None:
            break

        offset = next_offset

    return chunks


class TenantBM25Cache:
    """Hold an independent BM25 index for one tenant."""

    def __init__(self):
        self.document_ids: tuple[str, ...] | None = None
        self.chunks: list[dict] = []
        self.tokenized_chunks: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.dirty = True


# Each tenant receives an independent cache.
_bm25_lock = RLock()
_bm25_caches: dict[UUID, TenantBM25Cache] = {}


def _get_tenant_cache(tenant_id: UUID) -> TenantBM25Cache:
    """Get or create the BM25 cache for a tenant."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    with _bm25_lock:
        if tenant_id not in _bm25_caches:
            _bm25_caches[tenant_id] = TenantBM25Cache()

        return _bm25_caches[tenant_id]


def _build_index(
    cache: TenantBM25Cache,
    chunks: list[dict],
) -> None:
    """Build one tenant's BM25 index."""

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    if tokenized_chunks and any(tokenized_chunks):
        bm25 = BM25Okapi(tokenized_chunks)
    else:
        bm25 = None

    # Replace cache contents only after the new index is built.
    cache.chunks = chunks
    cache.tokenized_chunks = tokenized_chunks
    cache.bm25 = bm25


def refresh_bm25_index(tenant_id: UUID) -> None:
    """Reload completed-document chunks and rebuild one tenant's index."""

    cache = _get_tenant_cache(tenant_id)

    with _bm25_lock:
        # Assume the cache is unsafe until refresh succeeds.
        cache.dirty = True

        completed_ids = tuple(
            sorted(list_completed_document_ids(tenant_id))
        )

        chunks = load_chunks_from_qdrant(tenant_id)

        _build_index(cache, chunks)
        cache.document_ids = completed_ids

        # Mark clean only after all refresh steps succeed.
        cache.dirty = False


def refresh_all_bm25_indexes() -> None:
    """Mark every existing tenant cache dirty.

    System-wide recovery can call this after deleting stale
    document chunks. Each cache will rebuild on its next search.
    """

    with _bm25_lock:
        for cache in _bm25_caches.values():
            cache.dirty = True


def invalidate_bm25_index(tenant_id: UUID) -> None:
    """Mark one tenant's BM25 index dirty."""

    cache = _get_tenant_cache(tenant_id)

    with _bm25_lock:
        cache.dirty = True


def search_bm25(
    tenant_id: UUID,
    query: str,
    limit: int = 3,
) -> list[dict]:
    """Search completed-document chunks using one tenant's BM25 index."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    cache = _get_tenant_cache(tenant_id)

    with _bm25_lock:
        completed_ids = tuple(
            sorted(list_completed_document_ids(tenant_id))
        )

        # Rebuild if the cache is dirty or the document set changed.
        if (
            cache.dirty
            or completed_ids != cache.document_ids
        ):
            cache.dirty = True

            chunks = load_chunks_from_qdrant(tenant_id)

            _build_index(cache, chunks)
            cache.document_ids = completed_ids

            # Mark clean only after a successful rebuild.
            cache.dirty = False

        if not cache.chunks or cache.bm25 is None:
            return []

        scores = cache.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(cache.chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []
        query_token_set = set(query_tokens)

        for index in ranked_indices:
            document_tokens = set(
                cache.tokenized_chunks[index]
            )

            # Only return documents that actually match
            # at least one query token.
            if not query_token_set.intersection(document_tokens):
                continue

            result = cache.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

            if len(results) >= limit:
                break

        return results