from uuid import uuid4

import app.search as search_module


TEST_TENANT_ID = uuid4()


def test_search_documents_passes_candidates_to_reranker(monkeypatch):
    """Verify tenant propagation and candidate flow to the reranker."""

    calls = {}

    candidates = [
        {
            "chunk_id": "chunk-1",
            "text": "A passage about defect tracking.",
            "source_pages": [37],
            "score": 0.8,
        }
    ]

    ranked_chunks = [
        {
            **candidates[0],
            "reranker_score": 6.1,
        }
    ]

    selected_chunks = [
        {
            **ranked_chunks[0],
            "selected": True,
        }
    ]

    def fake_classify_query_intent(query):
        calls["classifier"] = {"query": query}
        return "specific_detail"

    def fake_hybrid_search(**kwargs):
        calls["hybrid"] = kwargs
        return candidates

    def fake_rerank(query, candidates, limit):
        calls["rerank"] = {
            "query": query,
            "candidates": candidates,
            "limit": limit,
        }
        return ranked_chunks

    def fake_select_context(query, ranked_chunks, limit, intent=None):
        calls["selector"] = {
            "query": query,
            "ranked_chunks": ranked_chunks,
            "limit": limit,
            "intent": intent,
        }
        return selected_chunks[:limit]

    monkeypatch.setattr(
        search_module,
        "classify_query_intent",
        fake_classify_query_intent,
    )
    monkeypatch.setattr(
        search_module,
        "search_qdrant_hybrid",
        fake_hybrid_search,
    )
    monkeypatch.setattr(
        search_module,
        "rerank_chunks",
        fake_rerank,
    )
    monkeypatch.setattr(
        search_module,
        "select_context",
        fake_select_context,
    )

    results = search_module.search_documents(
        tenant_id=TEST_TENANT_ID,
        query="How can defects be tracked?",
        limit=1,
        candidate_limit=10,
        document_id="test-document",
    )

    # Verify query intent classification.
    assert calls["classifier"]["query"] == "How can defects be tracked?"

    # Verify tenant-aware hybrid retrieval.
    assert calls["hybrid"]["tenant_id"] == TEST_TENANT_ID
    assert calls["hybrid"]["query"] == "How can defects be tracked?"
    assert calls["hybrid"]["limit"] == 10
    assert calls["hybrid"]["candidate_limit"] == 10
    assert calls["hybrid"]["document_id"] == "test-document"

    # Verify candidates are passed to the reranker.
    assert calls["rerank"]["query"] == "How can defects be tracked?"
    assert calls["rerank"]["candidates"] == candidates
    assert calls["rerank"]["limit"] == 10

    # Verify reranked chunks are passed to context selection.
    assert calls["selector"]["query"] == "How can defects be tracked?"
    assert calls["selector"]["ranked_chunks"] == ranked_chunks
    assert calls["selector"]["limit"] == 1
    assert calls["selector"]["intent"] == "specific_detail"

    assert results == selected_chunks


def test_overview_query_uses_adaptive_retrieval_depth(monkeypatch):
    """Overview queries should retrieve up to 60 candidates by default."""

    calls = {}

    def fake_classify_query_intent(query):
        return "overview"

    def fake_hybrid_search(**kwargs):
        calls["hybrid"] = kwargs
        return []

    def fake_rerank(query, candidates, limit):
        calls["rerank"] = {
            "query": query,
            "limit": limit,
        }
        return []

    def fake_select_context(query, ranked_chunks, limit, intent=None):
        calls["selector"] = {
            "limit": limit,
            "intent": intent,
        }
        return []

    monkeypatch.setattr(
        search_module,
        "classify_query_intent",
        fake_classify_query_intent,
    )
    monkeypatch.setattr(
        search_module,
        "search_qdrant_hybrid",
        fake_hybrid_search,
    )
    monkeypatch.setattr(
        search_module,
        "rerank_chunks",
        fake_rerank,
    )
    monkeypatch.setattr(
        search_module,
        "select_context",
        fake_select_context,
    )

    search_module.search_documents(
        tenant_id=TEST_TENANT_ID,
        query="What is the main purpose of this document?",
    )

    # Tenant must reach hybrid retrieval.
    assert calls["hybrid"]["tenant_id"] == TEST_TENANT_ID

    # Overview questions should retrieve 60 candidates by default.
    assert calls["hybrid"]["limit"] == 60
    assert calls["hybrid"]["candidate_limit"] == 60

    # The reranker should receive the same retrieval depth.
    assert calls["rerank"]["limit"] == 60

    # Final context selection still uses the requested output limit.
    assert calls["selector"]["limit"] == 3
    assert calls["selector"]["intent"] == "overview"


def test_specific_detail_query_uses_default_retrieval_depth(monkeypatch):
    """Specific-detail queries should use the normal default of 10 candidates."""

    calls = {}

    def fake_classify_query_intent(query):
        return "specific_detail"

    def fake_hybrid_search(**kwargs):
        calls["hybrid"] = kwargs
        return []

    def fake_rerank(query, candidates, limit):
        calls["rerank"] = {
            "query": query,
            "limit": limit,
        }
        return []

    def fake_select_context(query, ranked_chunks, limit, intent=None):
        calls["selector"] = {
            "limit": limit,
            "intent": intent,
        }
        return []

    monkeypatch.setattr(
        search_module,
        "classify_query_intent",
        fake_classify_query_intent,
    )
    monkeypatch.setattr(
        search_module,
        "search_qdrant_hybrid",
        fake_hybrid_search,
    )
    monkeypatch.setattr(
        search_module,
        "rerank_chunks",
        fake_rerank,
    )
    monkeypatch.setattr(
        search_module,
        "select_context",
        fake_select_context,
    )

    search_module.search_documents(
        tenant_id=TEST_TENANT_ID,
        query="How are defects tracked?",
    )

    # Tenant must reach hybrid retrieval.
    assert calls["hybrid"]["tenant_id"] == TEST_TENANT_ID

    # Specific-detail questions use the normal retrieval depth.
    assert calls["hybrid"]["limit"] == 10
    assert calls["hybrid"]["candidate_limit"] == 10

    # Reranking uses the same candidate depth.
    assert calls["rerank"]["limit"] == 10

    # Context selection retains its default output limit.
    assert calls["selector"]["limit"] == 3
    assert calls["selector"]["intent"] == "specific_detail"

def test_abstract_request_uses_direct_retrieval(monkeypatch):
    """Explicit abstract requests should retrieve the abstract directly."""

    calls = {}

    abstract_chunks = [
        {
            "chunk_id": "abstract-1",
            "text": "This is the document abstract.",
            "source_pages": [5],
            "section_type": "abstract",
        }
    ]

    def fake_search_document_abstract(**kwargs):
        calls["abstract"] = kwargs
        return abstract_chunks

    def fake_classify_query_intent(query):
        calls["classifier"] = query
        raise AssertionError(
            "Abstract requests should skip query classification."
        )

    def fake_hybrid_search(**kwargs):
        raise AssertionError(
            "Abstract requests should skip hybrid retrieval."
        )

    def fake_rerank(query, candidates, limit):
        raise AssertionError(
            "Abstract requests should skip reranking."
        )

    monkeypatch.setattr(
        search_module,
        "search_document_abstract",
        fake_search_document_abstract,
    )
    monkeypatch.setattr(
        search_module,
        "classify_query_intent",
        fake_classify_query_intent,
    )
    monkeypatch.setattr(
        search_module,
        "search_qdrant_hybrid",
        fake_hybrid_search,
    )
    monkeypatch.setattr(
        search_module,
        "rerank_chunks",
        fake_rerank,
    )

    results = search_module.search_documents(
        tenant_id=TEST_TENANT_ID,
        query="What is the abstract of this document?",
        limit=3,
        document_id="test-document",
    )

    assert calls["abstract"]["tenant_id"] == TEST_TENANT_ID
    assert calls["abstract"]["document_id"] == "test-document"

    assert "classifier" not in calls
    assert results == abstract_chunks


def test_abstract_request_returns_empty_when_abstract_missing(monkeypatch):
    """Return an empty result when no abstract chunk exists."""

    calls = {}

    def fake_search_document_abstract(**kwargs):
        calls["abstract"] = kwargs
        return []

    monkeypatch.setattr(
        search_module,
        "search_document_abstract",
        fake_search_document_abstract,
    )

    results = search_module.search_documents(
        tenant_id=TEST_TENANT_ID,
        query="Show me the abstract.",
        document_id="test-document",
    )

    assert calls["abstract"]["tenant_id"] == TEST_TENANT_ID
    assert calls["abstract"]["document_id"] == "test-document"
    assert results == []


def test_abstract_request_preserves_custom_collection(monkeypatch):
    """Direct abstract retrieval should support custom collections."""

    calls = {}

    def fake_search_document_abstract(**kwargs):
        calls["abstract"] = kwargs
        return []

    monkeypatch.setattr(
        search_module,
        "search_document_abstract",
        fake_search_document_abstract,
    )

    search_module.search_documents(
        tenant_id=TEST_TENANT_ID,
        query="What is the abstract?",
        document_id="test-document",
        collection_name="document_chunks_test",
    )

    assert calls["abstract"]["collection_name"] == "document_chunks_test"