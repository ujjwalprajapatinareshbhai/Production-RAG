from uuid import UUID

from app.context_selector import (
    _query_requests_abstract,
    _query_requests_report_summary,
    classify_query_intent,
    select_context,
)
from app.qdrant_hybrid_retriever import (
    search_document_abstract,
    search_qdrant_hybrid,
)
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

    # STEP 1: Handle explicit abstract requests directly.
    is_abstract_request = _query_requests_abstract(query)

    if is_abstract_request and document_id is not None:
        print("\n========== DIRECT ABSTRACT RETRIEVAL ==========")

        abstract_kwargs = {
            "tenant_id": tenant_id,
            "document_id": document_id,
        }

        # Preserve support for custom Qdrant collections.
        if collection_name is not None:
            abstract_kwargs["collection_name"] = collection_name

        abstract_chunks = search_document_abstract(
            **abstract_kwargs
        )

        print("Query:", query)
        print("Abstract chunks found:", len(abstract_chunks))

        for chunk in abstract_chunks:
            print("Section:", chunk.get("section_type"))
            print("Pages:", chunk.get("source_pages"))

        print("===============================================\n")

        return abstract_chunks[:limit]

    # STEP 2: Classify the query for normal hybrid retrieval.
    intent = classify_query_intent(query)

    # Explicit report-summary questions need deeper retrieval.
    is_report_summary = _query_requests_report_summary(query)

    retrieval_limit = candidate_limit

    if intent == "overview" or is_report_summary:
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

    # STEP 3: Retrieve candidates using dense + BM25 + RRF.
    candidates = search_qdrant_hybrid(**retrieval_kwargs)

    # DEBUG 1: Inspect all retrieved candidates.
    print("\n========== RETRIEVAL DEBUG ==========")
    print("Query:", query)
    print("Retrieval limit:", retrieval_limit)
    print("Total candidates:", len(candidates))

    for index, chunk in enumerate(candidates, start=1):
        print(
            f"Candidate {index} | "
            f"Pages: {chunk.get('source_pages')} | "
            f"Section: {chunk.get('section_type')} | "
            f"Text: {(chunk.get('text') or '')[:150].replace(chr(10), ' ')}"
        )

    print("=====================================\n")

    # DEBUG 2: Inspect the old mixed abstract candidate if present.
    for chunk in candidates:
        if chunk.get("source_pages") == [4, 5, 6]:
            print("\n========== OLD ABSTRACT CANDIDATE FULL TEXT ==========")
            print("Section:", chunk.get("section_type"))
            print("Pages:", chunk.get("source_pages"))
            print("Full text:")
            print(chunk.get("text"))
            print("=====================================================\n")

    # STEP 4: Rerank candidates using the CrossEncoder.
    ranked_chunks = rerank_chunks(
        query=query,
        candidates=candidates,
        limit=retrieval_limit,
    )

    # DEBUG 3: Inspect the top 15 reranked candidates.
    print("\n========== RERANKING DEBUG ==========")
    print("Total ranked chunks:", len(ranked_chunks))

    for index, chunk in enumerate(ranked_chunks[:15], start=1):
        print(
            f"Rank {index} | "
            f"Pages: {chunk.get('source_pages')} | "
            f"Section: {chunk.get('section_type')} | "
            f"Score: {chunk.get('reranker_score')}"
        )

    print("=====================================\n")

    # STEP 5: Select the final context.
    return select_context(
        query=query,
        ranked_chunks=ranked_chunks,
        limit=limit,
        intent=intent,
    )