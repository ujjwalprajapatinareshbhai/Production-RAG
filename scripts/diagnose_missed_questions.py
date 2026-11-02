from app.qdrant_hybrid_retriever import search_qdrant_hybrid
from app.reranker import rerank_chunks


EVAL_COLLECTION = "document_chunks_eval_abstract"
CANDIDATE_LIMIT = 60
RERANK_LIMIT = 60

QUESTIONS = [
    {
        "question": (
            "What broad improvements does the report propose for "
            "Company X's testing process?"
        ),
        "expected_pages": {66, 67, 68, 70},
    },
    {
        "question": (
            "Which software development lifecycle did Company X use?"
        ),
        "expected_pages": {6, 11},
    },
]


def overlaps_expected_pages(chunk, expected_pages):
    """Return True if a chunk contains any expected source page."""
    return bool(
        expected_pages.intersection(
            chunk.get("source_pages", [])
        )
    )


def print_chunk(label, chunk, expected_pages):
    """Print the key diagnostic details for one chunk."""
    pages = chunk.get("source_pages", [])
    relevant = overlaps_expected_pages(chunk, expected_pages)

    print(f"\n{label}")
    print("Pages:", pages)
    print("Relevant:", relevant)

    if "retrieval_score" in chunk:
        print("Retrieval score:", chunk["retrieval_score"])

    if "reranker_score" in chunk:
        print("Reranker score:", chunk["reranker_score"])

    print("Section type:", chunk.get("section_type"))
    print("Text:", chunk.get("text", "")[:500])


def main():
    for item in QUESTIONS:
        query = item["question"]
        expected_pages = item["expected_pages"]

        print("\n" + "=" * 80)
        print("QUESTION:", query)
        print("EXPECTED PAGES:", sorted(expected_pages))
        print("COLLECTION:", EVAL_COLLECTION)
        print("=" * 80)

        candidates = search_qdrant_hybrid(
            query=query,
            limit=CANDIDATE_LIMIT,
            candidate_limit=CANDIDATE_LIMIT,
            collection_name=EVAL_COLLECTION,
        )

        print("\nRetrieved candidates:", len(candidates))

        # Stage 1: Did hybrid retrieval find relevant evidence?
        candidate_ranks = [
            rank
            for rank, chunk in enumerate(candidates, start=1)
            if overlaps_expected_pages(chunk, expected_pages)
        ]

        print("Relevant candidate ranks:", candidate_ranks)

        if not candidate_ranks:
            print(
                "\nDIAGNOSIS: No candidate overlaps the expected pages. "
                "Investigate candidate retrieval, collection contents, "
                "and source-page metadata before tuning the reranker."
            )
        else:
            print("\nRelevant chunks BEFORE reranking:")
            for rank, chunk in enumerate(candidates, start=1):
                if overlaps_expected_pages(chunk, expected_pages):
                    print_chunk(
                        f"Candidate rank {rank}",
                        chunk,
                        expected_pages,
                    )

        # Stage 2: Does reranking preserve and prioritize the evidence?
        ranked = rerank_chunks(
            query=query,
            candidates=candidates,
            limit=min(RERANK_LIMIT, len(candidates)),
        )

        relevant_ranks = [
            rank
            for rank, chunk in enumerate(ranked, start=1)
            if overlaps_expected_pages(chunk, expected_pages)
        ]

        print("\nTOP 10 AFTER RERANKING")

        for rank, chunk in enumerate(ranked[:10], start=1):
            print_chunk(
                f"Reranked rank {rank}",
                chunk,
                expected_pages,
            )

        print("\nRelevant ranks AFTER reranking:", relevant_ranks)

        top3_relevant = any(
            overlaps_expected_pages(chunk, expected_pages)
            for chunk in ranked[:3]
        )

        if not candidate_ranks:
            print("RESULT: Candidate retrieval miss.")
        elif top3_relevant:
            print("RESULT: Relevant evidence is in the reranked top 3.")
        else:
            print(
                "RESULT: Relevant evidence was retrieved, "
                "but none of it reached the reranked top 3."
            )

        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()