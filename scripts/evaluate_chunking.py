from __future__ import annotations

from pathlib import Path

from qdrant_client import models

from app.bm25_retriever import load_chunks_from_qdrant
from app.chunker import chunk_pages_independently
from app.embeddings import embed_texts
from app.pdf_loader import load_pdf
from app.reranker import rerank_chunks
from app.vector_store import (
    SPARSE_MODEL_NAME,
    VECTOR_SIZE,
    get_vector_client,
    upsert_chunks,
)

# Keep this experiment isolated from production.
PDF_PATH = Path("data/pdfs/Testing-in-a-Small-Company1.pdf")
EVAL_COLLECTION = "document_chunks_eval_page_aware"
PRODUCTION_COLLECTION = "document_chunks"

QUESTIONS = [
    {
        "question": "What challenges do small software companies face when improving software quality?",
        "expected_pages": {6},
    },
    {
        "question": "What was the main goal of the report for Company X?",
        "expected_pages": {5},
    },
    {
        "question": "How did the researchers identify testing problems at Company X?",
        "expected_pages": {9},
    },
    {
        "question": "What testing knowledge were the developers missing?",
        "expected_pages": {10},
    },
]


def create_eval_collection() -> None:
    """Create a fresh, isolated collection for experimental chunks."""
    client = get_vector_client()

    if client.collection_exists(EVAL_COLLECTION):
        raise RuntimeError(
            f"Collection '{EVAL_COLLECTION}' already exists. "
            "Stopping to avoid overwriting evaluation data."
        )

    client.create_collection(
        collection_name=EVAL_COLLECTION,
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


def index_experimental_chunks(chunks) -> None:
    """Insert page-aware chunks into the isolated evaluation collection."""
    from fastembed import SparseTextEmbedding
    import uuid

    client = get_vector_client()
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    batch_size = 32

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]

        dense_vectors = embed_texts(texts)
        sparse_vectors = list(sparse_model.embed(texts))

        points = []

        for chunk, dense, sparse in zip(
            batch,
            dense_vectors,
            sparse_vectors,
        ):
            points.append(
                models.PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            chunk.chunk_id,
                        )
                    ),
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
            collection_name=EVAL_COLLECTION,
            points=points,
            wait=True,
        )


def search_collection(
    collection_name: str,
    query: str,
    candidate_limit: int = 30,
) -> list[dict]:
    """Run dense + sparse RRF retrieval against a specific collection."""
    from fastembed import SparseTextEmbedding

    client = get_vector_client()
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    dense_query = embed_texts([query])[0]
    sparse_query = next(sparse_model.query_embed([query]))

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                limit=candidate_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist(),
                ),
                using="bm25",
                limit=candidate_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidate_limit,
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
                "document_id": payload.get("document_id"),
                "filename": payload.get("filename"),
                "text": payload.get("text", ""),
                "start_page": payload.get("start_page"),
                "end_page": payload.get("end_page"),
                "source_pages": payload.get("source_pages", []),
            }
        )

    return results


def main() -> None:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    client = get_vector_client()

    if not client.collection_exists(PRODUCTION_COLLECTION):
        raise RuntimeError(
            f"Production collection '{PRODUCTION_COLLECTION}' not found."
        )

    # Read original production chunks, but never change them.
    original_chunks = load_chunks_from_qdrant()
    original_document_id = original_chunks[0]["document_id"]

    # Load the PDF and build experimental chunks only in memory.
    pages = load_pdf(PDF_PATH)

    if pages[0].document_id != original_document_id:
        raise RuntimeError(
            "PDF hash does not match the existing indexed document. "
            "Stopping before creating the evaluation collection."
        )

    experimental_chunks = chunk_pages_independently(
        pages,
        chunk_size=400,
        overlap=60,
    )

    print("Original chunks:", len(original_chunks))
    print("Page-aware chunks:", len(experimental_chunks))

    # Create the evaluation collection only after validation.
    create_eval_collection()

    try:
        index_experimental_chunks(experimental_chunks)

        print("\nEVALUATION RESULTS")
        print("=" * 80)

        for item in QUESTIONS:
            query = item["question"]
            expected_pages = item["expected_pages"]

            original_candidates = search_collection(
                PRODUCTION_COLLECTION,
                query,
            )
            experimental_candidates = search_collection(
                EVAL_COLLECTION,
                query,
            )

            original_ranked = rerank_chunks(
                query,
                original_candidates,
                limit=10,
            )
            experimental_ranked = rerank_chunks(
                query,
                experimental_candidates,
                limit=10,
            )

            print(f"\nQuestion: {query}")
            print(f"Expected evidence pages: {sorted(expected_pages)}")

            for label, results in (
                ("ORIGINAL", original_ranked),
                ("PAGE-AWARE", experimental_ranked),
            ):
                found_rank = None

                for rank, result in enumerate(results, start=1):
                    result_pages = set(result.get("source_pages") or [])
                    if result_pages & expected_pages:
                        found_rank = rank
                        break

                print(f"\n{label}: expected-page evidence rank = {found_rank}")

                for rank, result in enumerate(results[:3], start=1):
                    print(
                        f"  {rank}. pages={result.get('source_pages')} "
                        f"reranker={result.get('reranker_score')}"
                    )
                    print(
                        "     "
                        + result.get("text", "")[:180].replace("\n", " ")
                    )

    finally:
        print(
            "\nThe evaluation collection is retained for inspection:",
            EVAL_COLLECTION,
        )
        print("Production collection was not modified.")


if __name__ == "__main__":
    main()