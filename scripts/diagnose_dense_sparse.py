from fastembed import SparseTextEmbedding
from qdrant_client import models

from app.embeddings import embed_texts
from app.vector_store import (
    SPARSE_MODEL_NAME,
    get_vector_client,
)


COLLECTION = "document_chunks_eval_abstract"
QUERY = "What was the main goal of the report for Company X?"
ABSTRACT_CHUNK_ID = (
    "157fccdd123e2e64856a7d816d9dd5076853ec353afdce092e146f602d88d239"
)


def main():
    client = get_vector_client()
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    dense_query = embed_texts([QUERY])[0]
    sparse_query = next(sparse_model.query_embed([QUERY]))

    searches = {
        "DENSE": {
            "query": dense_query,
        },
        "SPARSE": {
            "query": models.SparseVector(
                indices=sparse_query.indices.tolist(),
                values=sparse_query.values.tolist(),
            ),
            "using": "bm25",
        },
    }

    for method, arguments in searches.items():
        response = client.query_points(
            collection_name=COLLECTION,
            query=arguments["query"],
            using=arguments.get("using"),
            limit=60,
            with_payload=True,
            with_vectors=False,
        )

        rank = None

        for position, point in enumerate(response.points, start=1):
            payload = point.payload or {}

            if payload.get("chunk_id") == ABSTRACT_CHUNK_ID:
                rank = position
                break

        print(f"{method}: abstract rank in top 60 = {rank}")


if __name__ == "__main__":
    main()