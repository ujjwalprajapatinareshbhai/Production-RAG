from types import SimpleNamespace

import app.bm25_retriever as bm25_module


class FakeClient:
    def __init__(self, points):
        self.points = points
        self.calls = 0

    def scroll(self, **kwargs):
        self.calls += 1
        return self.points, None


def make_point(document_id, text):
    return SimpleNamespace(
        payload={
            "document_id": document_id,
            "chunk_id": f"chunk-{document_id}",
            "filename": "example.pdf",
            "text": text,
            "start_page": 1,
            "end_page": 1,
            "source_pages": [1],
        }
    )


def test_load_chunks_includes_only_completed_documents(monkeypatch):
    points = [
        make_point("completed-document", "Defect tracking process"),
        make_point("failed-document", "Defect tracking failure"),
        make_point("pending-document", "Defect tracking pending"),
    ]

    client = FakeClient(points)

    monkeypatch.setattr(
        bm25_module,
        "list_completed_document_ids",
        lambda: ["completed-document"],
    )
    monkeypatch.setattr(
        bm25_module,
        "get_vector_client",
        lambda: client,
    )

    chunks = bm25_module.load_chunks_from_qdrant()

    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "completed-document"
    assert chunks[0]["text"] == "Defect tracking process"


def test_load_chunks_skips_qdrant_when_no_documents_completed(
    monkeypatch,
):
    def unexpected_client_call():
        raise AssertionError("Qdrant should not be accessed.")

    monkeypatch.setattr(
        bm25_module,
        "list_completed_document_ids",
        lambda: [],
    )
    monkeypatch.setattr(
        bm25_module,
        "get_vector_client",
        unexpected_client_call,
    )

    assert bm25_module.load_chunks_from_qdrant() == []


def test_bm25_index_is_reused_between_searches(monkeypatch):
    points = [
    make_point(
        "completed-document",
        "Defect tracking improves software quality",
    ),
    make_point(
        "completed-document",
        "Software testing identifies defects",
    ),
    make_point(
        "completed-document",
        "Gardening flowers grow during spring",
    ),
    ]
    client = FakeClient(points)

    monkeypatch.setattr(
        bm25_module,
        "list_completed_document_ids",
        lambda: ["completed-document"],
    )
    monkeypatch.setattr(
        bm25_module,
        "get_vector_client",
        lambda: client,
    )

    # These functions will be implemented in the next step.
    bm25_module.refresh_bm25_index()

    first_results = bm25_module.search_bm25("defect tracking")
    calls_after_first_search = client.calls

    second_results = bm25_module.search_bm25("software testing")

    assert first_results
    assert second_results
    assert calls_after_first_search > 0
    assert client.calls == calls_after_first_search


def test_refresh_bm25_index_reloads_chunks(monkeypatch):
    points = [
        make_point(
            "completed-document",
            "Defect tracking improves software quality",
        ),
    ]

    client = FakeClient(points)

    monkeypatch.setattr(
        bm25_module,
        "list_completed_document_ids",
        lambda: ["completed-document"],
    )
    monkeypatch.setattr(
        bm25_module,
        "get_vector_client",
        lambda: client,
    )

    bm25_module.refresh_bm25_index()
    calls_after_first_refresh = client.calls

    bm25_module.refresh_bm25_index()

    assert client.calls > calls_after_first_refresh


def test_bm25_cache_rebuilds_after_refresh_failure(monkeypatch):
    monkeypatch.setattr(
        bm25_module,
        "list_completed_document_ids",
        lambda: ["completed-document"],
    )

    points = [
        make_point(
            "completed-document",
            "Defect tracking improves software quality",
        )
    ]
    fake_client = FakeClient(points)

    monkeypatch.setattr(
        bm25_module,
        "get_vector_client",
        lambda: fake_client,
    )

    # Build the initial index successfully.
    bm25_module.refresh_bm25_index()

    # Simulate a failure while refreshing the same document set.
    def fail_to_load_chunks():
        raise RuntimeError("Simulated Qdrant failure")

    monkeypatch.setattr(
        bm25_module,
        "load_chunks_from_qdrant",
        fail_to_load_chunks,
    )

    try:
        bm25_module.refresh_bm25_index()
        assert False, "Expected the refresh to fail"
    except RuntimeError as exc:
        assert str(exc) == "Simulated Qdrant failure"

    # Restore chunk loading. The next search should retry the rebuild.
    monkeypatch.setattr(
    bm25_module,
    "load_chunks_from_qdrant",
    lambda: [
        {
            "chunk_id": "chunk-completed-document",
            "document_id": "completed-document",
            "filename": "example.pdf",
            "text": "Defect tracking improves software quality",
            "start_page": 1,
            "end_page": 1,
            "source_pages": [1],
        },
        {
            "chunk_id": "chunk-unrelated-document",
            "document_id": "completed-document",
            "filename": "example.pdf",
            "text": "Gardening flowers grow during spring",
            "start_page": 2,
            "end_page": 2,
            "source_pages": [2],
        },
    ],
)

    

    results = bm25_module.search_bm25("defect tracking")

# The failed refresh was recovered: the cache was rebuilt and is clean.
    assert bm25_module._bm25_cache_dirty is False
    assert len(bm25_module._cached_chunks) == 2
    assert bm25_module._cached_bm25 is not None

   