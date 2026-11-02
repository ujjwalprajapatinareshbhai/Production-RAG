from scripts.evaluate_chunking import (
    EVAL_COLLECTION as PAGE_AWARE_COLLECTION,
    QUESTIONS,
    search_collection,
)
from app.reranker import rerank_chunks

PRODUCTION_COLLECTION = "document_chunks"
ABSTRACT_COLLECTION = "document_chunks_eval_abstract"

COLLECTIONS = {
    "BASELINE": PRODUCTION_COLLECTION,
    "PAGE_AWARE": PAGE_AWARE_COLLECTION,
    "ABSTRACT_AWARE": ABSTRACT_COLLECTION,
}


def find_evidence_rank(results, expected_pages):
    """Return the first rank containing any expected evidence page."""
    for rank, result in enumerate(results, start=1):
        result_pages = set(result.get("source_pages") or [])

        if result_pages & expected_pages:
            return rank

    return None


def main():
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 80)

    for item in QUESTIONS:
        query = item["question"]
        expected_pages = item["expected_pages"]

        print(f"\nQuestion: {query}")
        print(f"Expected evidence pages: {sorted(expected_pages)}")

        for strategy, collection in COLLECTIONS.items():
            candidates = search_collection(
                collection_name=collection,
                query=query,
                candidate_limit=30,
            )

            ranked = rerank_chunks(
                query=query,
                candidates=candidates,
                limit=10,
            )

            evidence_rank = find_evidence_rank(
                ranked,
                expected_pages,
            )

            print(
                f"  {strategy:14} "
                f"evidence_rank={evidence_rank}"
            )


if __name__ == "__main__":
    main()