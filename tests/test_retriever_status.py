
from types import SimpleNamespace

import app.qdrant_hybrid_retriever as hybrid_module
import app.retriever as dense_module


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
        lambda: ["completed-document"],
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

    results = dense_module.search_chunks("defect tracking")

    assert results == []
    assert len(client.calls) == 1

    query_filter = client.calls[0]["query_filter"]
    condition = query_filter.must[0]

    assert condition.key == "document_id"
    assert condition.match.any == ["completed-document"]


def test_dense_retriever_skips_search_when_no_documents_completed(
    monkeypatch,
):
    def unexpected_call():
        raise AssertionError("Completed-document lookup should return empty.")

    monkeypatch.setattr(
        dense_module,
        "list_completed_document_ids",
        lambda: [],
    )
    monkeypatch.setattr(
        dense_module,
        "embed_text",
        lambda query: unexpected_call(),
    )
    monkeypatch.setattr(
        dense_module,
        "get_vector_client",
        lambda: unexpected_call(),
    )

    results = dense_module.search_chunks("defect tracking")

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
        lambda: ["completed-document"],
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

    results = hybrid_module.search_qdrant_hybrid("defect tracking")

    assert results == []
    assert len(client.calls) == 1

    call = client.calls[0]
    query_filter = call["query_filter"]

    assert query_filter.must[0].match.any == ["completed-document"]

    # Both dense and sparse candidate searches must use the same filter.
    assert len(call["prefetch"]) == 2

    for prefetch in call["prefetch"]:
        assert prefetch.filter.must[0].match.any == [
            "completed-document"
        ]


def test_hybrid_retriever_skips_search_when_no_documents_completed(
    monkeypatch,
):
    def unexpected_call():
        raise AssertionError("Completed-document lookup should return empty.")

    monkeypatch.setattr(
        hybrid_module,
        "list_completed_document_ids",
        lambda: [],
    )
    monkeypatch.setattr(
        hybrid_module,
        "embed_text",
        lambda query: unexpected_call(),
    )
    monkeypatch.setattr(
        hybrid_module,
        "get_vector_client",
        lambda: unexpected_call(),
    )

    results = hybrid_module.search_qdrant_hybrid("defect tracking")

    assert results == []
