
from qdrant_client import models


from app.embeddings import embed_text
from app.registry import get_document, list_completed_document_ids
from app.vector_store import COLLECTION_NAME, get_vector_client


def search_chunks(
    query: str,
    limit: int = 3,
    document_id: str | None = None,
) -> list[dict]:
    """Search dense vectors belonging to completed documents."""
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    query = query.strip()

    # Restrict search to completed documents.
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

    query_vector = embed_text(query)

    response = get_vector_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
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
            }
        )

    return results
