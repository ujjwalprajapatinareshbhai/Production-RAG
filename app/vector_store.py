import os
import uuid
from collections.abc import Sequence
from uuid import UUID

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from app.config import settings
from app.embeddings import embed_texts
from app.models import Chunk


COLLECTION_NAME = settings.qdrant_collection
VECTOR_SIZE = settings.vector_size
SPARSE_MODEL_NAME = settings.sparse_model

_client = QdrantClient(url=settings.qdrant_url)
_sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def initialize_vector_store() -> None:
    """Create the collection with dense and sparse vector support if needed."""
    if _client.collection_exists(COLLECTION_NAME):
        return

    _client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "": models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


def get_vector_client() -> QdrantClient:
    """Return the shared Qdrant client."""
    return _client


def _build_payload(chunk: Chunk) -> dict:
    """Build the metadata stored alongside a chunk's vectors."""
    return {
        "chunk_id": chunk.chunk_id,
        "tenant_id": str(chunk.tenant_id),
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "source_pages": chunk.source_pages,
    }


def upsert_chunk(chunk: Chunk) -> None:
    """Embed and store one chunk with both dense and sparse vectors."""
    upsert_chunks([chunk])


def upsert_chunks(
    chunks: Sequence[Chunk],
    batch_size: int = 32,
) -> int:
    """Embed and store chunks in batches with dense and BM25 vectors."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if not chunks:
        return 0

    initialize_vector_store()
    stored_count = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]

        # Dense vectors: semantic similarity.
        dense_vectors = embed_texts(texts)

        # Sparse vectors: keyword relevance using BM25.
        sparse_embeddings = list(_sparse_model.embed(texts))

        if len(dense_vectors) != len(batch):
            raise RuntimeError(
                "Dense embedding count does not match chunk count."
            )

        if len(sparse_embeddings) != len(batch):
            raise RuntimeError(
                "Sparse embedding count does not match chunk count."
            )

        points = []

        for chunk, dense_vector, sparse_embedding in zip(
            batch,
            dense_vectors,
            sparse_embeddings,
        ):
            sparse_vector = models.SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist(),
            )

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    chunk.chunk_id,
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "": dense_vector,
                        "bm25": sparse_vector,
                    },
                    payload=_build_payload(chunk),
                )
            )

        _client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        stored_count += len(points)

    return stored_count


def delete_document_chunks(
    tenant_id: UUID,
    document_id: str,
) -> None:
    """Delete document chunks belonging to a specific tenant."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    if not document_id or len(document_id) != 64:
        raise ValueError(
            "document_id must be a 64-character SHA-256 hash."
        )

    if not _client.collection_exists(COLLECTION_NAME):
        return

    _client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(
                            value=str(tenant_id),
                        ),
                    ),
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(
                            value=document_id,
                        ),
                    ),
                ]
            )
        ),
        wait=True,
    )