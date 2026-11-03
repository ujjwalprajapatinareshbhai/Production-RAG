from types import SimpleNamespace
from uuid import UUID

import pytest

import app.bm25_retriever as bm25_module


TEST_TENANT_ID = UUID("22222222-2222-2222-2222-222222222222")
OTHER_TENANT_ID = UUID("33333333-3333-3333-3333-333333333333")


@pytest.fixture(autouse=True)
def reset_bm25_caches(monkeypatch):
    """Give every test an isolated set of tenant BM25 caches."""
    monkeypatch.setattr(bm25_module, "_bm25_caches", {})


class FakeClient:
    def __init__(self, points):
        self.points = points
        self.calls = 0
        self.last_kwargs = None

    def scroll(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return self.points, None


def make_point(document_id, text, tenant_id=TEST_TENANT_ID):
    return SimpleNamespace(
        payload={
            "tenant_id": str(tenant_id),
            "document_id": document_id,
            "chunk_id": f"chunk-{document_id}",
            "filename": "example.pdf",
            "text": text,
            "start_page": 1,
            "end_page": 1,
            "source_pages": [1],
        }
    )


def make_chunk(document_id, text):
    return {
        "chunk_id": f"chunk-{document_id}",
        "document_id": document_id,
        "filename": "example.pdf",
        "text": text,
        "start_page": 1,
        "end_page": 1,
        "source_pages": [1],
    }


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
        lambda tenant_id: ["completed-document"],
    )
    monkeypatch.setattr(
        bm25_module,
        "get_vector_client",
        lambda: client,
    )

    chunks = bm25_module.load_chunks_from_qdrant(TEST_TENANT_ID)

    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "completed-document"
    assert chunks[0]["text"] == "Defect tracking process"

    # Qdrant must receive a tenant-specific filter.
    assert str(TEST_TENANT_ID) in str(
        client.last_kwargs["scroll_filter"]
    )


def test_load_chunks_skips_qdrant_when_no_documents_completed(
    monkeypatch,
):
    def unexpected_client_call():
        raise AssertionError("Qdrant should not be accessed.")

    monkeypatch.setattr(
        bm25_module,
        "list_completed_document_ids",
        lambda tenant_id: [],
    )
    monkeypatch.setattr(
        bm25_module,
        "get_vector_client",
        unexpected_client_call,
    )

    assert bm25_module.load_chunks_from_qdrant(TEST_TENANT_ID) == []


def test_bm25_index_is_reused_between_searches(monkeypatch):
    load_calls = []

    chunks = [
        make_chunk(
            "completed-document",
            "Defect tracking improves software quality",
        ),
        make_chunk(
            "completed-document",
            "Software testing identifies defects",
        ),
        make_chunk(
            "completed-document",
            "Gardening flowers grow during spring",
        ),
    ]

    def fake_load_chunks(tenant_id):
        load_calls.append(tenant_id)
        return chunks

    monkeypatch.setattr(
        bm25_module,
        "load_chunks_from_qdrant",
        fake_load_chunks,
    )

    bm25_module.refresh_bm25_index(TEST_TENANT_ID)

    first_results = bm25_module.search_bm25(
        TEST_TENANT_ID,
        "defect tracking",
    )

    calls_after_first_search = len(load_calls)

    second_results = bm25_module.search_bm25(
        TEST_TENANT_ID,
        "software testing",
    )

    assert first_results
    assert second_results
    assert calls_after_first_search == 1
    assert len(load_calls) == calls_after_first_search


def test_bm25_caches_are_isolated_between_tenants(monkeypatch):
    tenant_chunks = {
        TEST_TENANT_ID: [
            make_chunk(
                "tenant-a-document",
                "Alpha project defect tracking and software testing",
            ),
        ],
        OTHER_TENANT_ID: [
            make_chunk(
                "tenant-b-document",
                "Bravo gardening flowers and spring plants",
            ),
        ],
    }

    def fake_load_chunks(tenant_id):
        return tenant_chunks[tenant_id]

    monkeypatch.setattr(
        bm25_module,
        "load_chunks_from_qdrant",
        fake_load_chunks,
    )

    bm25_module.refresh_bm25_index(TEST_TENANT_ID)
    bm25_module.refresh_bm25_index(OTHER_TENANT_ID)

    tenant_a_results = bm25_module.search_bm25(
        TEST_TENANT_ID,
        "alpha defect tracking",
    )

    tenant_b_results = bm25_module.search_bm25(
        OTHER_TENANT_ID,
        "bravo gardening",
    )

    assert tenant_a_results
    assert tenant_b_results

    assert all(
        result["document_id"] == "tenant-a-document"
        for result in tenant_a_results
    )

    assert all(
        result["document_id"] == "tenant-b-document"
        for result in tenant_b_results
    )

    assert TEST_TENANT_ID in bm25_module._bm25_caches
    assert OTHER_TENANT_ID in bm25_module._bm25_caches
    assert (
        bm25_module._bm25_caches[TEST_TENANT_ID]
        is not bm25_module._bm25_caches[OTHER_TENANT_ID]
    )


def test_refresh_bm25_index_reloads_chunks(monkeypatch):
    load_calls = []

    chunks = [
        make_chunk(
            "completed-document",
            "Defect tracking improves software quality",
        ),
    ]

    def fake_load_chunks(tenant_id):
        load_calls.append(tenant_id)
        return chunks

    monkeypatch.setattr(
        bm25_module,
        "load_chunks_from_qdrant",
        fake_load_chunks,
    )

    bm25_module.refresh_bm25_index(TEST_TENANT_ID)

    calls_after_first_refresh = len(load_calls)

    bm25_module.refresh_bm25_index(TEST_TENANT_ID)

    assert calls_after_first_refresh == 1
    assert len(load_calls) > calls_after_first_refresh


def test_bm25_cache_rebuilds_after_refresh_failure(monkeypatch):
    chunks = [
        make_chunk(
            "completed-document",
            "Defect tracking improves software quality",
        ),
    ]

    monkeypatch.setattr(
        bm25_module,
        "load_chunks_from_qdrant",
        lambda tenant_id: chunks,
    )

    # Build the initial tenant index successfully.
    bm25_module.refresh_bm25_index(TEST_TENANT_ID)

    cache = bm25_module._bm25_caches[TEST_TENANT_ID]

    assert cache.dirty is False
    assert cache.bm25 is not None

    # Simulate a failure while refreshing the index.
    def fail_to_load_chunks(tenant_id):
        raise RuntimeError("Simulated Qdrant failure")

    monkeypatch.setattr(
        bm25_module,
        "load_chunks_from_qdrant",
        fail_to_load_chunks,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated Qdrant failure",
    ):
        bm25_module.refresh_bm25_index(TEST_TENANT_ID)

    # Restore chunk loading with updated data.
    updated_chunks = [
        make_chunk(
            "completed-document",
            "Defect tracking improves software quality",
        ),
        make_chunk(
            "completed-document",
            "Gardening flowers grow during spring",
        ),
    ]

    monkeypatch.setattr(
        bm25_module,
        "load_chunks_from_qdrant",
        lambda tenant_id: updated_chunks,
    )

    # The next search should retry the failed rebuild.
    results = bm25_module.search_bm25(
        TEST_TENANT_ID,
        "defect tracking",
    )

    cache = bm25_module._bm25_caches[TEST_TENANT_ID]

    assert results
    assert cache.dirty is False
    assert len(cache.chunks) == 2
    assert cache.bm25 is not None