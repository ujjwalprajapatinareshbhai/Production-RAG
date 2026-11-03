from uuid import UUID

from app.bm25_retriever import search_bm25
from app.retriever import search_chunks


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    limit: int = 3,
    k: int = 60,
) -> list[dict]:
    """Combine ranked search results using Reciprocal Rank Fusion."""

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    fused = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = result.get("chunk_id")

            if not chunk_id:
                continue

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                }

            fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    results = sorted(
        fused.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )

    return results[:limit]


def search_hybrid(
    tenant_id: UUID,
    query: str,
    limit: int = 3,
    candidate_limit: int = 10,
    document_id: str | None = None,
) -> list[dict]:
    """Combine tenant-scoped dense and BM25 retrieval using RRF."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    if candidate_limit < 1:
        raise ValueError("candidate_limit must be at least 1.")

    # Stage 1: Semantic retrieval, restricted to this tenant.
    dense_results = search_chunks(
        tenant_id=tenant_id,
        query=query,
        limit=candidate_limit,
        document_id=document_id,
    )

    # Stage 2: Keyword retrieval, restricted to this tenant.
    bm25_results = search_bm25(
        tenant_id=tenant_id,
        query=query,
        limit=candidate_limit,
    )

    # Stage 3: Apply optional document-level filtering.
    if document_id is not None:
        bm25_results = [
            result
            for result in bm25_results
            if result.get("document_id") == document_id
        ]

    # Stage 4: Combine both rankings.
    return reciprocal_rank_fusion(
        [dense_results, bm25_results],
        limit=limit,
    )