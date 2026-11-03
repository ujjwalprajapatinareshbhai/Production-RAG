from uuid import UUID

from qdrant_client import models

from app.embeddings import embed_text
from app.registry import get_document, list_completed_document_ids
from app.vector_store import COLLECTION_NAME, get_vector_client


def search_chunks(
    tenant_id: UUID,
    query: str,
    limit: int = 3,
    document_id: str | None = None,
) -> list[dict]:
    """Search dense vectors belonging to completed documents in one tenant."""

    if not isinstance(tenant_id, UUID):
        raise ValueError("tenant_id must be a UUID.")

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    query = query.strip()

    # Every query must be restricted to the authenticated tenant.
    tenant_filter = models.FieldCondition(
        key="tenant_id",
        match=models.MatchValue(value=str(tenant_id)),
    )

    # Restrict search to completed documents.
    if document_id is not None:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        document_id = document_id.strip()

        # Look up the document inside this tenant only.
        document = get_document(tenant_id, document_id)

        if document is None or document["status"] != "completed":
            return []

        query_filter = models.Filter(
            must=[
                tenant_filter,
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
            ]
        )

    else:
        # Get completed document IDs belonging only to this tenant.
        completed_ids = list_completed_document_ids(tenant_id)

        if not completed_ids:
            return []

        query_filter = models.Filter(
            must=[
                tenant_filter,
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=completed_ids),
                ),
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

        # Do not return a result if its tenant ownership is missing
        # or does not match the requested tenant.
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
            }
        )

    return results