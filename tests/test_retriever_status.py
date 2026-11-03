from types import SimpleNamespace
from uuid import uuid4

import app.qdrant_hybrid_retriever as hybrid_module
import app.retriever as dense_module


TEST_TENANT_ID = uuid4()


class FakeList(list):
    """Provide the tolist() method expected by the sparse-vector API."""

    def tolist(self):
        return list(self)


class FakeClient:
    def __init__(self):
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=[])


def test_dense_retriever_filters_to_completed_documents(monkeypatch):
    client = FakeClient()

    monkeypatch.setattr(
        dense_module,
        "list_completed_document_ids",
        lambda tenant_id: ["completed-document"],
    )
    monkeypatch.setattr(
        dense_module,
        "embed_text",
        lambda query: [0.1, 0.2],
    )
    monkeypatch.setattr(
        dense_module,
        "get_vector_client",
        lambda: client,
    )

    results = dense_module.search_chunks(
        tenant_id=TEST_TENANT_ID,
        query="defect tracking",
    )

    assert results == []
    assert len(client.calls) == 1

    query_filter = client.calls[0]["query_filter"]

    # First condition must restrict results to the requested tenant.
    assert query_filter.must[0].key == "tenant_id"
    assert query_filter.must[0].match.value == str(TEST_TENANT_ID)

    # Second condition must restrict results to completed documents.
    assert query_filter.must[1].key == "document_id"
    assert query_filter.must[1].match.any == ["completed-document"]


def test_dense_retriever_skips_search_when_no_documents_completed(
    monkeypatch,
):
    def unexpected_call(*args, **kwargs):
        raise AssertionError(
            "Search should not execute when no documents are completed."
        )

    monkeypatch.setattr(
        dense_module,
        "list_completed_document_ids",
        lambda tenant_id: [],
    )
    monkeypatch.setattr(
        dense_module,
        "embed_text",
        unexpected_call,
    )
    monkeypatch.setattr(
        dense_module,
        "get_vector_client",
        unexpected_call,
    )

    results = dense_module.search_chunks(
        tenant_id=TEST_TENANT_ID,
        query="defect tracking",
    )

    assert results == []


def test_hybrid_retriever_filters_to_completed_documents(monkeypatch):
    client = FakeClient()

    class FakeSparseModel:
        def query_embed(self, queries):
            return iter([
                SimpleNamespace(
                    indices=FakeList([1, 2]),
                    values=FakeList([0.5, 0.8]),
                )
            ])

    monkeypatch.setattr(
        hybrid_module,
        "list_completed_document_ids",
        lambda tenant_id: ["completed-document"],
    )
    monkeypatch.setattr(
        hybrid_module,
        "embed_text",
        lambda query: [0.1, 0.2],
    )
    monkeypatch.setattr(
        hybrid_module,
        "_sparse_model",
        FakeSparseModel(),
    )
    monkeypatch.setattr(
        hybrid_module,
        "get_vector_client",
        lambda: client,
    )

    results = hybrid_module.search_qdrant_hybrid(
        tenant_id=TEST_TENANT_ID,
        query="defect tracking",
    )

    assert results == []
    assert len(client.calls) == 1

    call = client.calls[0]
    query_filter = call["query_filter"]

    # Final RRF query must enforce tenant isolation.
    assert query_filter.must[0].key == "tenant_id"
    assert query_filter.must[0].match.value == str(TEST_TENANT_ID)

    # Final RRF query must include completed-document filtering.
    assert query_filter.must[1].key == "document_id"
    assert query_filter.must[1].match.any == ["completed-document"]

    # Both dense and sparse candidate searches must use the same filter.
    assert len(call["prefetch"]) == 2

    for prefetch in call["prefetch"]:
        assert prefetch.filter.must[0].key == "tenant_id"
        assert prefetch.filter.must[0].match.value == str(TEST_TENANT_ID)

        assert prefetch.filter.must[1].key == "document_id"
        assert prefetch.filter.must[1].match.any == [
            "completed-document"
        ]


def test_hybrid_retriever_skips_search_when_no_documents_completed(
    monkeypatch,
):
    def unexpected_call(*args, **kwargs):
        raise AssertionError(
            "Search should not execute when no documents are completed."
        )

    monkeypatch.setattr(
        hybrid_module,
        "list_completed_document_ids",
        lambda tenant_id: [],
    )
    monkeypatch.setattr(
        hybrid_module,
        "embed_text",
        unexpected_call,
    )
    monkeypatch.setattr(
        hybrid_module,
        "get_vector_client",
        unexpected_call,
    )

    results = hybrid_module.search_qdrant_hybrid(
        tenant_id=TEST_TENANT_ID,
        query="defect tracking",
    )

    assert results == []