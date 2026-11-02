import re
from threading import RLock

from rank_bm25 import BM25Okapi

from app.registry import list_completed_document_ids
from app.vector_store import COLLECTION_NAME, get_vector_client


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks_from_qdrant() -> list[dict]:
    """Load chunks belonging only to completed documents."""
    completed_ids = set(list_completed_document_ids())

    if not completed_ids:
        return []

    client = get_vector_client()
    chunks = []
    offset = None

    while True:
        response, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in response:
            payload = point.payload or {}
            document_id = payload.get("document_id")
            text = payload.get("text", "")

            if document_id not in completed_ids:
                continue

            if not text.strip():
                continue

            chunks.append(
                {
                    "chunk_id": payload.get("chunk_id"),
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


# In-memory BM25 cache. It lasts for the lifetime of this Python process.
_bm25_lock = RLock()
_cached_document_ids: tuple[str, ...] | None = None
_cached_chunks: list[dict] = []
_cached_tokenized_chunks: list[list[str]] = []
_cached_bm25: BM25Okapi | None = None
_bm25_cache_dirty = True


def _build_index(chunks: list[dict]) -> None:
    """Build the BM25 index from a list of document chunks."""
    global _cached_chunks
    global _cached_tokenized_chunks
    global _cached_bm25

    _cached_chunks = chunks
    _cached_tokenized_chunks = [
        tokenize(chunk["text"]) for chunk in chunks
    ]

    if _cached_tokenized_chunks and any(_cached_tokenized_chunks):
        _cached_bm25 = BM25Okapi(_cached_tokenized_chunks)
    else:
        _cached_bm25 = None


def refresh_bm25_index() -> None:
    """Reload completed-document chunks and rebuild the BM25 index."""
    global _cached_document_ids
    global _bm25_cache_dirty

    with _bm25_lock:
        # Assume the cache is unsafe until refresh completes successfully.
        _bm25_cache_dirty = True

        completed_ids = tuple(sorted(list_completed_document_ids()))
        chunks = load_chunks_from_qdrant()

        _build_index(chunks)
        _cached_document_ids = completed_ids

        # Mark the cache clean only after all refresh steps succeed.
        _bm25_cache_dirty = False


def search_bm25(query: str, limit: int = 3) -> list[dict]:
    """Search completed-document chunks using a cached BM25 index."""
    global _cached_document_ids
    global _bm25_cache_dirty

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    with _bm25_lock:
        completed_ids = tuple(sorted(list_completed_document_ids()))

        # Rebuild if the cache is dirty or the document set has changed.
        if (
            _bm25_cache_dirty
            or completed_ids != _cached_document_ids
        ):
            _bm25_cache_dirty = True

            chunks = load_chunks_from_qdrant()
            _build_index(chunks)
            _cached_document_ids = completed_ids

            # Mark clean only after a successful rebuild.
            _bm25_cache_dirty = False

        if not _cached_chunks or _cached_bm25 is None:
            return []

        scores = _cached_bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(_cached_chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices:
            if scores[index] <= 0:
                break

            result = _cached_chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

            if len(results) >= limit:
                break

        return results