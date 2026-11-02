
from fastembed import SparseTextEmbedding
from qdrant_client import models

from app.vector_store import COLLECTION_NAME, get_vector_client
from app.config import settings


MODEL_NAME = settings.sparse_model
BATCH_SIZE = 32


def migrate_sparse_vectors() -> int:
    """Add BM25 sparse vectors to existing Qdrant points."""
    client = get_vector_client()
    embedder = SparseTextEmbedding(model_name=MODEL_NAME)

    offset = None
    updated_count = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        texts = []
        for point in points:
            payload = point.payload or {}
            text = payload.get("text", "")

            if not text.strip():
                raise ValueError(
                    f"Point {point.id} has no text; migration stopped."
                )

            texts.append(text)

        sparse_embeddings = list(embedder.embed(texts))

        if len(sparse_embeddings) != len(points):
            raise RuntimeError(
                "Embedding count does not match point count."
            )

        updates = []

        for point, embedding in zip(points, sparse_embeddings):
            sparse_vector = models.SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )

            updates.append(
                models.PointVectors(
                    id=point.id,
                    vector={"bm25": sparse_vector},
                )
            )

        # Update only the named sparse vector.
        # Existing dense vectors and payloads are preserved.
        client.update_vectors(
            collection_name=COLLECTION_NAME,
            points=updates,
            wait=True,
        )

        updated_count += len(updates)
        print(f"Updated {updated_count} points...")

        if next_offset is None:
            break

        offset = next_offset

    return updated_count


if __name__ == "__main__":
    count = migrate_sparse_vectors()
    print(f"Migration complete. Sparse vectors updated: {count}")
