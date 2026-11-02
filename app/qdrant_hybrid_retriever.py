
from fastembed import SparseTextEmbedding
from qdrant_client import models
from app.registry import get_document, list_completed_document_ids
from app.embeddings import embed_text
from app.vector_store import COLLECTION_NAME, get_vector_client
from app.config import settings

_sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)


def search_qdrant_hybrid(
    query: str,
    limit: int = 3,
    candidate_limit: int = 10,
    document_id: str | None = None,
    collection_name: str = COLLECTION_NAME, 
) -> list[dict]:
    """Search completed documents using dense + sparse retrieval and RRF."""
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if limit < 1 or candidate_limit < 1:
        raise ValueError("Search limits must be at least 1.")

    query = query.strip()

    # Only search documents that have completed ingestion.
    if document_id is not None:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        document_id = document_id.strip()
        document = get_document(document_id)

        if document is None or document["status"] != "completed":
            return []

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )
    else:
        completed_ids = list_completed_document_ids()

        # No completed documents means there is nothing searchable.
        if not completed_ids:
            return []

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=completed_ids),
                )
            ]
        )

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
            models.Prefetch(
                query=dense_vector,
                limit=candidate_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=sparse_vector,
                using="bm25",
                limit=candidate_limit,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    results = []

    for point in response.points:
        payload = point.payload or {}

        results.append(
            {
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
