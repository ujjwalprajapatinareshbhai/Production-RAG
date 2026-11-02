import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import models

from app.chunker import chunk_pages_by_section
from app.embeddings import embed_texts
from app.pdf_loader import load_pdf
from app.registry import get_document
from app.search import search_documents
from app.vector_store import get_vector_client
from scripts.evaluate_context_selection import QUESTIONS


PDF_PATH = "data/pdfs/Testing-in-a-Small-Company1.pdf"

CURRENT_COLLECTION = "document_chunks_eval_current_v1"
SECTION_COLLECTION = "document_chunks_eval_section_v1"

VECTOR_SIZE = 384
BATCH_SIZE = 32
CANDIDATE_LIMIT = 60
EVALUATION_K = (3, 5, 10)
MAX_K = max(EVALUATION_K)

SPARSE_MODEL_NAME = "Qdrant/bm25"

client = get_vector_client()
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def create_section_collection(collection_name: str) -> None:
    """Create a new isolated collection without overwriting existing data."""

    if client.collection_exists(collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' already exists. "
            "Refusing to overwrite it. Choose a new collection name."
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

    print(f"Created isolated collection: {collection_name}")


def store_section_chunks(chunks: list) -> int:
    """Embed and store section-aware chunks in the experimental collection."""

    if not chunks:
        raise ValueError("No section-aware chunks were generated.")

    stored_count = 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [chunk.text for chunk in batch]

        dense_vectors = embed_texts(texts)
        sparse_embeddings = list(sparse_model.embed(texts))

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
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "source_pages": chunk.source_pages,
            }

            # Store section metadata only if the Chunk model provides it.
            section_type = getattr(chunk, "section_type", None)
            if section_type is not None:
                payload["section_type"] = section_type

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
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=SECTION_COLLECTION,
            points=points,
            wait=True,
        )

        stored_count += len(points)
        print(
            f"Stored {stored_count}/{len(chunks)} "
            "section-aware chunks"
        )

    return stored_count


def evaluate_question(
    question_item: dict,
    collection_name: str,
) -> dict:
    """Run the normal hybrid-search, reranking, and context-selection pipeline."""

    results = search_documents(
        query=question_item["question"],
        limit=MAX_K,
        candidate_limit=CANDIDATE_LIMIT,
        collection_name=collection_name,
    )

    expected_pages = set(question_item["expected_pages"])

    relevant_ranks = []

    for rank, result in enumerate(results, start=1):
        result_pages = set(result.get("source_pages") or [])

        if expected_pages.intersection(result_pages):
            relevant_ranks.append(rank)

    first_relevant_rank = (
        relevant_ranks[0] if relevant_ranks else None
    )

    hits_at_k = {
        k: any(rank <= k for rank in relevant_ranks)
        for k in EVALUATION_K
    }

    reciprocal_rank_at_10 = (
        1 / first_relevant_rank
        if first_relevant_rank is not None
        and first_relevant_rank <= 10
        else 0.0
    )

    return {
        "hits_at_k": hits_at_k,
        "rank": first_relevant_rank,
        "rr10": reciprocal_rank_at_10,
    }


def main() -> None:
    print("=" * 80)
    print("SECTION-AWARE CHUNKING EVALUATION")
    print("=" * 80)

    # Safety check 1: the existing baseline must be available.
    if not client.collection_exists(CURRENT_COLLECTION):
        raise RuntimeError(
            f"Baseline collection '{CURRENT_COLLECTION}' does not exist."
        )

    # Safety check 2: build and validate the experimental chunks first.
    pages = load_pdf(PDF_PATH)

    if not pages:
        raise RuntimeError("No PDF pages were extracted.")

    chunks = chunk_pages_by_section(
        pages,
        chunk_size=400,
        overlap=60,
    )

    if not chunks:
        raise RuntimeError("Section-aware chunking returned no chunks.")

    document_id = chunks[0].document_id

    # search_documents filters results using completed registry records.
    # Verify that the existing document is completed; do not change its status.
    document = get_document(document_id)

    if document is None or document["status"] != "completed":
        raise RuntimeError(
            "The document must already be marked completed in the registry. "
            "This experiment will not modify registry records."
        )

    print(f"PDF pages: {len(pages)}")
    print(f"Section-aware chunks: {len(chunks)}")
    print(f"Document ID: {document_id}")
    print(f"Evaluation questions: {len(QUESTIONS)}")
    print(f"Evaluation cutoffs: {EVALUATION_K}")

    # Safety check 3: this function refuses to overwrite an existing collection.
    create_section_collection(SECTION_COLLECTION)

    stored_count = store_section_chunks(chunks)

    if stored_count != len(chunks):
        raise RuntimeError(
            f"Expected {len(chunks)} chunks, stored {stored_count}."
        )

    print(f"\nVerified stored chunk count: {stored_count}")

    collections = {
        "current": CURRENT_COLLECTION,
        "section_aware": SECTION_COLLECTION,
    }

    stats = {
        name: {
            "evaluated": 0,
            "hits_at_k": {k: 0 for k in EVALUATION_K},
            "mrr10_sum": 0.0,
            "details": [None] * len(QUESTIONS),
        }
        for name in collections
    }

    # Evaluate both strategies with the same questions and search pipeline.
    for index, question_item in enumerate(QUESTIONS):
        print("\n" + "=" * 80)
        print(f"Question {index + 1}/{len(QUESTIONS)}")
        print(question_item["question"])

        for strategy_name, collection_name in collections.items():
            try:
                outcome = evaluate_question(
                    question_item,
                    collection_name,
                )
            except Exception as exc:
                print(
                    f"{strategy_name}: ERROR "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            rank = outcome["rank"]
            rank_text = (
                str(rank) if rank is not None else "not in top 10"
            )

            print(
                f"{strategy_name:14} | "
                + " | ".join(
                    f"Hit@{k}={outcome['hits_at_k'][k]}"
                    for k in EVALUATION_K
                )
                + f" | first relevant rank={rank_text}"
            )

            stats[strategy_name]["details"][index] = {
                "rank": rank,
                "rr10": outcome["rr10"],
                "hits_at_k": outcome["hits_at_k"],
            }

            # Unanswerable questions are excluded from the main metrics.
            if question_item["answerable"]:
                current_stats = stats[strategy_name]
                current_stats["evaluated"] += 1

                for k in EVALUATION_K:
                    current_stats["hits_at_k"][k] += int(
                        outcome["hits_at_k"][k]
                    )

                current_stats["mrr10_sum"] += outcome["rr10"]

    print("\n" + "=" * 80)
    print("FINAL METRICS — ANSWERABLE QUESTIONS")
    print("=" * 80)

    for strategy_name, strategy_stats in stats.items():
        count = strategy_stats["evaluated"]

        print(f"\nStrategy: {strategy_name}")
        print(f"Questions evaluated: {count}")

        if count == 0:
            print("No questions evaluated successfully.")
            continue

        for k in EVALUATION_K:
            hits = strategy_stats["hits_at_k"][k]
            print(
                f"Hit@{k}: {hits}/{count} "
                f"({hits / count * 100:.2f}%)"
            )

        mrr10 = strategy_stats["mrr10_sum"] / count
        print(f"MRR@10: {mrr10:.4f}")

    print("\n" + "=" * 80)
    print("QUESTION-LEVEL COMPARISON")
    print("=" * 80)

    improved = []
    regressed = []
    unchanged = []
    missing = []

    current_details = stats["current"]["details"]
    section_details = stats["section_aware"]["details"]

    for index, question_item in enumerate(QUESTIONS):
        if not question_item["answerable"]:
            continue

        old = current_details[index]
        new = section_details[index]

        if old is None or new is None:
            missing.append((index + 1, question_item["question"]))
            continue

        if old["rr10"] < new["rr10"]:
            improved.append((index + 1, question_item["question"]))
        elif old["rr10"] > new["rr10"]:
            regressed.append((index + 1, question_item["question"]))
        else:
            unchanged.append((index + 1, question_item["question"]))

    for label, items in (
        ("Improved", improved),
        ("Regressed", regressed),
        ("Unchanged", unchanged),
        ("Missing / failed", missing),
    ):
        print(f"\n{label}: {len(items)}")

        for number, question in items:
            print(f"  Q{number}: {question}")

    print("\nEvaluation complete.")
    print(f"Baseline collection: {CURRENT_COLLECTION}")
    print(f"Experimental collection: {SECTION_COLLECTION}")
    print("No production vectors or registry records were modified.")


if __name__ == "__main__":
    main()