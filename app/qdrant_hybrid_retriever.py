from uuid import UUID

from fastembed import SparseTextEmbedding
from qdrant_client import models

from app.registry import get_document, list_completed_document_ids
from app.embeddings import embed_text
from app.vector_store import COLLECTION_NAME, get_vector_client
from app.config import settings


_sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)


def search_qdrant_hybrid(
    tenant_id: UUID,
    query: str,
    limit: int = 3,
    candidate_limit: int = 10,
    document_id: str | None = None,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Search completed documents using dense + sparse retrieval and RRF."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if limit < 1 or candidate_limit < 1:
        raise ValueError("Search limits must be at least 1.")

    query = query.strip()

    # Every Qdrant query must be restricted to this tenant.
    tenant_condition = models.FieldCondition(
        key="tenant_id",
        match=models.MatchValue(value=str(tenant_id)),
    )

    # Only search documents that have completed ingestion.
    if document_id is not None:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        document_id = document_id.strip()

        # Check document ownership inside the authenticated tenant.
        document = get_document(tenant_id, document_id)

        if document is None or document["status"] != "completed":
            return []

        query_filter = models.Filter(
            must=[
                tenant_condition,
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
            ]
        )

    else:
        # Retrieve only completed documents belonging to this tenant.
        completed_ids = list_completed_document_ids(tenant_id)

        if not completed_ids:
            return []

        query_filter = models.Filter(
            must=[
                tenant_condition,
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=completed_ids),
                ),
            ]
        )

    # Generate dense and sparse query vectors.
    dense_vector = embed_text(query)
    sparse_embedding = next(_sparse_model.query_embed([query]))

    sparse_vector = models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )

    client = get_vector_client()

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            # Dense vector search.
            models.Prefetch(
                query=dense_vector,
                limit=candidate_limit,
                filter=query_filter,
            ),
            # Sparse BM25 search.
            models.Prefetch(
                query=sparse_vector,
                using="bm25",
                limit=candidate_limit,
                filter=query_filter,
            ),
        ],
        # Reciprocal Rank Fusion combines dense and sparse results.
        query=models.FusionQuery(fusion=models.Fusion.RRF),

        # Apply the same restriction to the final fusion query.
        query_filter=query_filter,

        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    results = []

    for point in response.points:
        payload = point.payload or {}

        # Defense in depth: reject missing or mismatched tenant ownership.
        if payload.get("tenant_id") != str(tenant_id):
            continue

        results.append(
            {
                "tenant_id": payload.get("tenant_id"),
                "score": point.score,
                "filename": payload.get("filename"),
                "document_id": payload.get("document_id"),
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text"),
                "start_page": payload.get("start_page"),
                "end_page": payload.get("end_page"),
                "source_pages": payload.get("source_pages", []),
                "section_type": payload.get("section_type"),
            }
        )

    return results