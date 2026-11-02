from scripts.evaluate_chunking import search_collection
from app.reranker import rerank_chunks


COLLECTION = "document_chunks_eval_abstract"

ORIGINAL_QUERY = "What was the main goal of the report for Company X?"

EXPANDED_QUERY = (
    "What was the objective of the software testing improvement research?"
)

ABSTRACT_CHUNK_ID = (
    "157fccdd123e2e64856a7d816d9dd5076853ec353afdce092e146f602d88d239"
)


def merge_candidates(*candidate_lists):
    """Merge candidate lists without duplicating chunks."""
    merged = {}

    for candidates in candidate_lists:
        for candidate in candidates:
            chunk_id = candidate.get("chunk_id")

            if chunk_id and chunk_id not in merged:
                merged[chunk_id] = candidate

    return list(merged.values())


def main():
    # Retrieve candidates from both query variants.
    original_candidates = search_collection(
        collection_name=COLLECTION,
        query=ORIGINAL_QUERY,
        candidate_limit=60,
    )

    expanded_candidates = search_collection(
        collection_name=COLLECTION,
        query=EXPANDED_QUERY,
        candidate_limit=60,
    )

    candidates = merge_candidates(
        original_candidates,
        expanded_candidates,
    )

    print("Unique candidates:", len(candidates))

    # Inspect the exact standalone abstract.
    abstract = next(
        (
            item
            for item in candidates
            if item.get("chunk_id") == ABSTRACT_CHUNK_ID
        ),
        None,
    )

    print("\n" + "=" * 80)
    print("STANDALONE ABSTRACT")
    print("=" * 80)

    if abstract is None:
        print("Abstract was not retrieved.")
    else:
        print("Chunk ID:", abstract.get("chunk_id"))
        print("Source pages:", abstract.get("source_pages"))
        print("Full text:\n")
        print(abstract.get("text", ""))

    # Rank the same candidates using the original question.
    ranked = rerank_chunks(
        query=ORIGINAL_QUERY,
        candidates=candidates,
        limit=len(candidates),
    )

    print("\n" + "=" * 80)
    print("TOP 5 RESULTS FOR THE ORIGINAL QUESTION")
    print("=" * 80)

    for rank, item in enumerate(ranked[:5], start=1):
        print(f"\nRank: {rank}")
        print("Pages:", item.get("source_pages"))
        print("Score:", item.get("reranker_score"))
        print("Chunk ID:", item.get("chunk_id"))
        print("Text:\n")
        print(item.get("text", "")[:1200])
        print("-" * 80)


if __name__ == "__main__":
    main()