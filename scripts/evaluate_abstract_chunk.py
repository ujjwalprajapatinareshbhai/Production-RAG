from pathlib import Path
import uuid

from qdrant_client import models

from app.bm25_retriever import load_chunks_from_qdrant
from app.chunker import extract_abstract_chunk
from app.embeddings import embed_texts
from app.pdf_loader import load_pdf
from app.reranker import rerank_chunks
from app.vector_store import (
    SPARSE_MODEL_NAME,
    VECTOR_SIZE,
    get_vector_client,
)

PDF_PATH = Path("data/pdfs/Testing-in-a-Small-Company1.pdf")
PRODUCTION_COLLECTION = "document_chunks"
EVAL_COLLECTION = "document_chunks_eval_abstract"

QUERY = "What was the main goal of the report for Company X?"


def main() -> None:
    client = get_vector_client()

    if not client.collection_exists(PRODUCTION_COLLECTION):
        raise RuntimeError("Production collection not found.")

    # Stop rather than overwrite an existing evaluation collection.
    if client.collection_exists(EVAL_COLLECTION):
        raise RuntimeError(
            f"Collection '{EVAL_COLLECTION}' already exists. "
            "No data was changed. Choose a new name for another experiment."
        )

    # Read the original chunks without modifying them.
    original_chunks = load_chunks_from_qdrant()

    if not original_chunks:
        raise RuntimeError("No original chunks were found.")

    pages = load_pdf(PDF_PATH)

    if pages[0].document_id != original_chunks[0]["document_id"]:
        raise RuntimeError("PDF hash mismatch. Stopping safely.")

    abstract_chunk = extract_abstract_chunk(pages)

    if abstract_chunk is None:
        raise RuntimeError("Could not extract the abstract.")

    # Copy original chunks into dictionaries matching the existing payload.
    chunks = list(original_chunks)

    # Add the clean abstract chunk as an extra evaluation-only item.
    chunks.append(
        {
            "chunk_id": abstract_chunk.chunk_id,
            "document_id": abstract_chunk.document_id,
            "filename": abstract_chunk.filename,
            "chunk_index": abstract_chunk.chunk_index,
            "text": abstract_chunk.text,
            "start_page": abstract_chunk.start_page,
            "end_page": abstract_chunk.end_page,
            "source_pages": abstract_chunk.source_pages,
        }
    )

    # Create a separate collection with the same vector configuration.
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

    try:
        from fastembed import SparseTextEmbedding

        sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME
        )

        batch_size = 32

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk["text"] for chunk in batch]

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
                                chunk["chunk_id"],
                            )
                        ),
                        vector={
                            "": dense,
                            "bm25": models.SparseVector(
                                indices=sparse.indices.tolist(),
                                values=sparse.values.tolist(),
                            ),
                        },
                        payload=chunk,
                    )
                )

            client.upsert(
                collection_name=EVAL_COLLECTION,
                points=points,
                wait=True,
            )

        print("Original chunks:", len(original_chunks))
        print("Standalone abstract chunks added: 1")
        print(
            "Evaluation collection points:",
            client.count(
                collection_name=EVAL_COLLECTION,
                exact=True,
            ).count,
        )

        # Search production and evaluation with the same query.
        from scripts.evaluate_chunking import search_collection

        for collection in (
            PRODUCTION_COLLECTION,
            EVAL_COLLECTION,
        ):
            candidates = search_collection(
                collection,
                QUERY,
                candidate_limit=30,
            )
            ranked = rerank_chunks(
                QUERY,
                candidates,
                limit=10,
            )

            print("\n" + "=" * 70)
            print("COLLECTION:", collection)
            print("Query:", QUERY)

            abstract_rank = None

            for rank, result in enumerate(ranked, start=1):
                if 5 in (result.get("source_pages") or []):
                    abstract_rank = rank
                    print("Abstract-containing result rank:", rank)
                    print("Pages:", result.get("source_pages"))
                    print(
                        "Reranker score:",
                        result.get("reranker_score"),
                    )
                    print(
                        "Text:",
                        result.get("text", "")[:500].replace("\n", " "),
                    )
                    break

            if abstract_rank is None:
                print("No page-5 evidence found in reranked top 10.")

    finally:
        print("\nProduction collection was not modified.")
        print("Evaluation collection:", EVAL_COLLECTION)


if __name__ == "__main__":
    main()