import app.search as search_module


def test_search_documents_passes_candidates_to_reranker(monkeypatch):
    """Verify hybrid candidates are passed to the reranker."""

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

    # Mock query intent classification to avoid a real Ollama API call.
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

    # Replace external dependencies with test doubles.
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
        query="How can defects be tracked?",
        limit=1,
        candidate_limit=10,
        document_id="test-document",
    )

    # Verify query intent classification.
    assert calls["classifier"]["query"] == "How can defects be tracked?"

    # Verify hybrid retrieval.
    assert calls["hybrid"]["query"] == "How can defects be tracked?"
    assert calls["hybrid"]["limit"] == 10
    assert calls["hybrid"]["candidate_limit"] == 10
    assert calls["hybrid"]["document_id"] == "test-document"

    # Verify


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
        query="What is the main purpose of this document?",
    )

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
        query="How are defects tracked?",
    )

    # Specific-detail questions use the normal retrieval depth.
    assert calls["hybrid"]["limit"] == 10
    assert calls["hybrid"]["candidate_limit"] == 10

    # Reranking uses the same candidate depth.
    assert calls["rerank"]["limit"] == 10

    # Context selection still returns at most 