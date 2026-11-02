
from uuid import uuid4, uuid5, NAMESPACE_URL
from types import SimpleNamespace
from qdrant_client import models
import numpy as np
import pytest
import app.vector_store as vector_store
from app.models import Chunk
from app.vector_store import (
    COLLECTION_NAME,
    get_vector_client,
    upsert_chunks,
)


def test_upsert_stores_dense_and_sparse_vectors():
    client = get_vector_client()
    test_chunk_id = f"test-{uuid4().hex}"

    chunk = Chunk(
        chunk_id=test_chunk_id,
        document_id="test-document",
        filename="test-vector-store.txt",
        chunk_index=0,
        text=(
            "Software testing helps teams identify defects, "
            "verify requirements, and improve software quality."
        ),
        start_page=1,
        end_page=1,
        source_pages=[1],
    )

    try:
        stored_count = upsert_chunks([chunk])

        assert stored_count == 1

        point_id = str(
            __import__("uuid").uuid5(
                __import__("uuid").NAMESPACE_URL,
                test_chunk_id,
            )
        )

        response = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )

        assert len(response) == 1

        point = response[0]
        assert point.payload["chunk_id"] == test_chunk_id
        assert point.payload["text"] == chunk.text

        assert isinstance(point.vector, dict)
        assert "" in point.vector
        assert "bm25" in point.vector

        assert len(point.vector[""]) == 384
        assert len(point.vector["bm25"].indices) > 0
        assert len(point.vector["bm25"].values) > 0

    finally:
        # Delete only this test's uniquely identified point.
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[point_id] if "point_id" in locals() else [],
            wait=True,
        )


def test_upserting_same_chunk_twice_does_not_create_duplicates():
    client = get_vector_client()
    test_chunk_id = f"idempotency-test-{uuid4().hex}"

    chunk = Chunk(
        chunk_id=test_chunk_id,
        document_id="b" * 64,
        filename="idempotency-test.pdf",
        chunk_index=0,
        text="This is a test chunk for idempotent storage.",
        start_page=1,
        end_page=1,
        source_pages=[1],
    )

    point_id = str(uuid5(NAMESPACE_URL, test_chunk_id))

    try:
        first_count = upsert_chunks([chunk])
        second_count = upsert_chunks([chunk])

        assert first_count == 1
        assert second_count == 1

        result = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_id",
                        match=models.MatchValue(value=test_chunk_id),
                    )
                ]
            ),
            exact=True,
        )

        assert result.count == 1

    finally:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[point_id]),
            wait=True,
        )



def test_partial_batch_failure_leaves_only_first_batch_stored(
    monkeypatch,
):
    client = get_vector_client()
    vector_store.initialize_vector_store()

    chunks = [
        Chunk(
            chunk_id=f"partial-test-{uuid4().hex}",
            document_id="c" * 64,
            filename="partial-test.pdf",
            chunk_index=index,
            text=f"Test chunk number {index}.",
            start_page=1,
            end_page=1,
            source_pages=[1],
        )
        for index in range(2)
    ]

    point_ids = [
        str(uuid5(NAMESPACE_URL, chunk.chunk_id))
        for chunk in chunks
    ]

    original_upsert = client.upsert
    upsert_calls = 0

    def fail_on_second_upsert(*args, **kwargs):
        nonlocal upsert_calls
        upsert_calls += 1

        if upsert_calls == 2:
            raise RuntimeError("Simulated second-batch failure")

        return original_upsert(*args, **kwargs)

    def fake_sparse_embeddings(texts):
        return iter([
            SimpleNamespace(
                indices=np.array([1]),
                values=np.array([0.5]),
            )
            for _ in texts
        ])

    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts: [
            [1.0] + [0.0] * 383
            for _ in texts
        ],
    )
    monkeypatch.setattr(
        vector_store._sparse_model,
        "embed",
        fake_sparse_embeddings,
    )
    monkeypatch.setattr(
        client,
        "upsert",
        fail_on_second_upsert,
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="Simulated second-batch failure",
        ):
            upsert_chunks(chunks, batch_size=1)

        first_point = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_ids[0]],
            with_payload=True,
        )
        second_point = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_ids[1]],
            with_payload=True,
        )

        assert len(first_point) == 1
        assert len(second_point) == 0

    finally:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )

def test_delete_document_chunks_removes_only_target_document():
    from app.vector_store import (
        delete_document_chunks,
        get_vector_client,
        upsert_chunks,
    )

    from app.models import Chunk

    client = get_vector_client()
    collection_name = "document_chunks"

    document_a = "a" * 64
    document_b = "b" * 64

    chunk_a = Chunk(
        chunk_id="chunk-a",
        document_id=document_a,
        filename="document_a.pdf",
        chunk_index=0,
        text="Unique test content for document A.",
        start_page=1,
        end_page=1,
        source_pages=[1],
    )

    chunk_b = Chunk(
        chunk_id="chunk-b",
        document_id=document_b,
        filename="document_b.pdf",
        chunk_index=0,
        text="Unique test content for document B.",
        start_page=1,
        end_page=1,
        source_pages=[1],
    )

    try:
        upsert_chunks([chunk_a, chunk_b])

        delete_document_chunks(document_a)

        remaining_a = client.scroll(
            collection_name=collection_name,
            scroll_filter=__import__(
                "qdrant_client"
            ).models.Filter(
                must=[
                    __import__("qdrant_client").models.FieldCondition(
                        key="document_id",
                        match=__import__(
                            "qdrant_client"
                        ).models.MatchValue(value=document_a),
                    )
                ]
            ),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )[0]

        remaining_b = client.scroll(
            collection_name=collection_name,
            scroll_filter=__import__(
                "qdrant_client"
            ).models.Filter(
                must=[
                    __import__("qdrant_client").models.FieldCondition(
                        key="document_id",
                        match=__import__(
                            "qdrant_client"
                        ).models.MatchValue(value=document_b),
                    )
                ]
            ),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )[0]

        assert remaining_a == []
        assert len(remaining_b) == 1

    finally:
        delete_document_chunks(document_a)
        delete_document_chunks(document_b)

