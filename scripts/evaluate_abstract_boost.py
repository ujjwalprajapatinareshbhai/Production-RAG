from scripts.evaluate_chunking import search_collection
from app.reranker import rerank_chunks


COLLECTION = "document_chunks_eval_abstract"
CANDIDATE_LIMIT = 60

ABSTRACT_CHUNK_ID = (
    "157fccdd123e2e64856a7d816d9dd5076853ec353afdce092e146f602d88d239"
)

QUESTIONS = [
    {
        "name": "Q1: Small-company quality challenges",
        "query": "What challenges do small software companies face when improving software quality?",
        "expected_pages": {6},
    },
    {
        "name": "Q2: Main goal of the report",
        "query": "What was the main goal of the report for Company X?",
        "expected_pages": {5},
    },
    {
        "name": "Q3: How testing problems were identified",
        "query": "How did the researchers identify testing problems at Company X?",
        "expected_pages": {9},
    },
    {
        "name": "Q4: Missing testing knowledge",
        "query": "What testing knowledge were the developers missing?",
        "expected_pages": {10},
    },
]

BONUSES = [0.0, 1.0, 2.0]


def merge_candidates(*candidate_lists):
    merged = {}

    for candidate_list in candidate_lists:
        for candidate in candidate_list:
            chunk_id = candidate.get("chunk_id")

            if chunk_id and chunk_id not in merged:
                merged[chunk_id] = candidate

    return list(merged.values())


def rank_expected_evidence(ranked, expected_pages):
    for rank, item in enumerate(ranked, start=1):
        pages = set(item.get("source_pages", []))

        if pages.intersection(expected_pages):
            return rank

    return None


def main():
    for question in QUESTIONS:
        query = question["query"]
        expected_pages = question["expected_pages"]

        candidates = search_collection(
            collection_name=COLLECTION,
            query=query,
            candidate_limit=CANDIDATE_LIMIT,
        )

        # Also search with a targeted variant for the abstract-related question.
        if question["name"].startswith("Q2"):
            expanded = search_collection(
                collection_name=COLLECTION,
                query="What was the objective of the software testing improvement research?",
                candidate_limit=CANDIDATE_LIMIT,
            )
            candidates = merge_candidates(candidates, expanded)

        ranked = rerank_chunks(
            query=query,
            candidates=candidates,
            limit=len(candidates),
        )

        print("\n" + "=" * 75)
        print(question["name"])
        print("Unique candidates:", len(candidates))
        print("=" * 75)

        for bonus in BONUSES:
            adjusted = []

            for item in ranked:
                result = item.copy()

                if result.get("chunk_id") == ABSTRACT_CHUNK_ID:
                    result["adjusted_score"] = (
                        result["reranker_score"] + bonus
                    )
                else:
                    result["adjusted_score"] = result["reranker_score"]

                adjusted.append(result)

            adjusted.sort(
                key=lambda item: item["adjusted_score"],
                reverse=True,
            )

            evidence_rank = rank_expected_evidence(
                adjusted,
                expected_pages,
            )

            abstract_rank = next(
                (
                    rank
                    for rank, item in enumerate(adjusted, start=1)
                    if item.get("chunk_id") == ABSTRACT_CHUNK_ID
                ),
                None,
            )

            print(
                f"Bonus={bonus:.1f} | "
                f"Expected evidence rank={evidence_rank} | "
                f"Abstract rank={abstract_rank}"
            )

            print("Top 3 pages:")
            for item in adjusted[:3]:
                print(
                    "  Pages:",
                    item.get("source_pages"),
                    "| Adjusted score:",
                    round(item["adjusted_score"], 3),
                )


if __name__ == "__main__":
    main()