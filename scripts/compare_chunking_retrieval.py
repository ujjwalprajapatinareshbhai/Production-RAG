import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import models

from app.chunker import chunk_pages, chunk_pages_independently
from app.embeddings import embed_texts, embed_text
from app.pdf_loader import load_pdf
from app.reranker import rerank_chunks
from app.vector_store import get_vector_client


PDF_PATH = "data/pdfs/Testing-in-a-Small-Company1.pdf"

EXPERIMENTS = {
    "Current chunking": "document_chunks_eval_current_v1",
    "Page-aware chunking": "document_chunks_eval_page_v1",
}

QUERY = "Which software development lifecycle did Company X use?"
TARGET_PHRASE = "current software process used in company x"
EXPECTED_PAGES = {11}

VECTOR_SIZE = 384
BATCH_SIZE = 32
CANDIDATE_LIMIT = 60

client = get_vector_client()
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def create_experiment_collection(collection_name):
    """Create a new, isolated evaluation collection."""
    if client.collection_exists(collection_name):
        raise RuntimeError(
            f"Collection {collection_name!r} already exists. "
            "Nothing was changed. Choose a new collection name."
        )

    client.create_collection(
        collection_name=collection_name,
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


def store_chunks(collection_name, chunks):
    """Embed chunks and store them only in the named evaluation collection."""
    stored = 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [chunk.text for chunk in batch]

        dense_vectors = embed_texts(texts)
        sparse_embeddings = list(sparse_model.embed(texts))

        points = []

        for chunk, dense, sparse in zip(
            batch,
            dense_vectors,
            sparse_embeddings,
        ):
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "": dense,
                        "bm25": models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "start_page": chunk.start_page,
                        "end_page": chunk.end_page,
                        "source_pages": chunk.source_pages,
                    },
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        stored += len(points)

    return stored


def hybrid_search(collection_name):
    """Run dense + BM25 retrieval with reciprocal-rank fusion."""
    dense_query = embed_text(QUERY)
    sparse_result = next(sparse_model.query_embed([QUERY]))

    sparse_query = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                limit=CANDIDATE_LIMIT,
            ),
            models.Prefetch(
                query=sparse_query,
                using="bm25",
                limit=CANDIDATE_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=CANDIDATE_LIMIT,
        with_payload=True,
        with_vectors=False,
    )

    results = []

    for point in response.points:
        payload = point.payload or {}

        results.append(
            {
                "score": point.score,
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text", ""),
                "source_pages": payload.get("source_pages", []),
                "start_page": payload.get("start_page"),
                "end_page": payload.get("end_page"),
            }
        )

    return results


def find_target_rank(results):
    """Find the rank of a retrieved chunk containing the target statement."""
    for rank, result in enumerate(results, start=1):
        if TARGET_PHRASE in result["text"].lower():
            return rank, result

    return None, None


def evaluate_strategy(name, collection_name, chunks):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Chunks:", len(chunks))

    create_experiment_collection(collection_name)
    stored = store_chunks(collection_name, chunks)

    print("Chunks stored:", stored)

    retrieved = hybrid_search(collection_name)
    hybrid_rank, target = find_target_rank(retrieved)

    print("\nHybrid retrieval:")
    print("Candidates returned:", len(retrieved))
    print("Target statement rank:", hybrid_rank or "Not in top 60")

    if target:
        print("Target pages:", target["source_pages"])

    reranked = rerank_chunks(
        query=QUERY,
        candidates=retrieved,
        limit=len(retrieved) or 1,
    )

    rerank_rank, reranked_target = find_target_rank(reranked)

    print("\nAfter cross-encoder reranking:")
    print("Target statement rank:", rerank_rank or "Not in reranked candidates")

    if reranked_target:
        print("Target pages:", reranked_target["source_pages"])

    relevant_hybrid = [
        rank
        for rank, result in enumerate(retrieved, start=1)
        if EXPECTED_PAGES.intersection(result["source_pages"])
    ]

    relevant_reranked = [
        rank
        for rank, result in enumerate(reranked, start=1)
        if EXPECTED_PAGES.intersection(result["source_pages"])
    ]

    print("\nAny chunk containing page 11:")
    print("Hybrid ranks:", relevant_hybrid[:10] or "None")
    print("Reranked ranks:", relevant_reranked[:10] or "None")

    print("\nTop 5 after reranking:")
    for rank, result in enumerate(reranked[:5], start=1):
        print(
            f"{rank}. pages={result['source_pages']}, "
            f"score={result.get('reranker_score')}, "
            f"text={result['text'][:140]!r}"
        )


def main():
    pages = load_pdf(PDF_PATH)

    strategies = {
        "Current chunking": chunk_pages(
            pages,
            chunk_size=400,
            overlap=60,
        ),
        "Page-aware chunking": chunk_pages_independently(
            pages,
            chunk_size=400,
            overlap=60,
        ),
    }

    print("PDF pages:", len(pages))
    print("Question:", QUERY)

    # Safety check: refuse to proceed if either destination already exists.
    existing = [
        name
        for name in EXPERIMENTS.values()
        if client.collection_exists(name)
    ]

    if existing:
        raise RuntimeError(
            f"Evaluation collections already exist: {existing}. "
            "No collections were modified. Choose fresh names first."
        )

    for strategy_name, chunks in strategies.items():
        evaluate_strategy(
            name=strategy_name,
            collection_name=EXPERIMENTS[strategy_name],
            chunks=chunks,
        )


if __name__ == "__main__":
    main()