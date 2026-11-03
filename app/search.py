from uuid import UUID

from app.context_selector import (
    classify_query_intent,
    select_context,
)
from app.qdrant_hybrid_retriever import search_qdrant_hybrid
from app.reranker import rerank_chunks


DEFAULT_CANDIDATE_LIMIT = 10
OVERVIEW_CANDIDATE_LIMIT = 60


def search_documents(
    tenant_id: UUID,
    query: str,
    limit: int = 3,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    document_id: str | None = None,
    collection_name: str | None = None,
) -> list[dict]:
    """Retrieve, rerank, and select context within one tenant."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if limit < 1 or candidate_limit < 1:
        raise ValueError("Search limits must be at least 1.")

    # Classify once so retrieval and context selection share the intent.
    intent = classify_query_intent(query)

    # Overview questions need a deeper candidate pool to find the abstract.
    retrieval_limit = candidate_limit

    if intent == "overview":
        retrieval_limit = max(
            candidate_limit,
            OVERVIEW_CANDIDATE_LIMIT,
        )

    retrieval_kwargs = {
        "tenant_id": tenant_id,
        "query": query,
        "limit": retrieval_limit,
        "candidate_limit": retrieval_limit,
        "document_id": document_id,
    }

    # Preserve the default collection for existing callers.
    if collection_name is not None:
        retrieval_kwargs["collection_name"] = collection_name

    candidates = search_qdrant_hybrid(**retrieval_kwargs)

    ranked_chunks = rerank_chunks(
        query=query,
        candidates=candidates,
        limit=retrieval_limit,
    )

    return select_context(
        query=query,
        ranked_chunks=ranked_chunks,
        limit=limit,
        intent=intent,
    )