from scripts.evaluate_chunking import search_collection
from app.reranker import rerank_chunks


COLLECTION = "document_chunks_eval_abstract"
CANDIDATE_LIMIT = 60

ABSTRACT_CHUNK_ID = (
    "157fccdd123e2e64856a7d816d9dd5076853ec353afdce092e146f602d88d239"
)

ABSTRACT_BONUS = 1.0

QUESTIONS = [
    {
        "name": "Q1: Small-company quality challenges",
        "query": (
            "What challenges do small software companies face "
            "when improving software quality?"
        ),
        "expected_pages": {6},
    },
    {
        "name": "Q2: Main goal of the report",
        "query": "What was the main goal of the report for Company X?",
        "expected_pages": {5},
        "expanded_query": (
            "What was the objective of the software testing improvement research?"
        ),
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


def merge_candidates(*candidate_lists):
    merged = {}

    for candidate_list in candidate_lists:
        for candidate in candidate_list:
            chunk_id = candidate.get("chunk_id")

            if chunk_id and chunk_id not in merged:
                merged[chunk_id] = candidate

    return list(merged.values())


def find_evidence_rank(ranked, expected_pages):
    for rank, item in enumerate(ranked, start=1):
        pages = set(item.get("source_pages", []))

        if pages.intersection(expected_pages):
            return rank

    return None


def apply_abstract_bonus(ranked, bonus):
    adjusted = []

    for item in ranked:
        result = item.copy()
        score = result["reranker_score"]

        if result.get("chunk_id") == ABSTRACT_CHUNK_ID:
            score += bonus

        result["adjusted_score"] = score
        adjusted.append(result)

    adjusted.sort(
        key=lambda item: item["adjusted_score"],
        reverse=True,
    )

    return adjusted


def main():
    total_queries = len(QUESTIONS)

    for question in QUESTIONS:
        query = question["query"]
        expected_pages = question["expected_pages"]

        original_candidates = search_collection(
            collection_name=COLLECTION,
            query=query,
            candidate_limit=CANDIDATE_LIMIT,
        )

        candidate_lists = [original_candidates]

        expanded_query = question.get("expanded_query")

        if expanded_query:
            expanded_candidates = search_collection(
                collection_name=COLLECTION,
                query=expanded_query,
                candidate_limit=CANDIDATE_LIMIT,
            )
            candidate_lists.append(expanded_candidates)

        candidates = merge_candidates(*candidate_lists)

        baseline = rerank_chunks(
            query=query,
            candidates=candidates,
            limit=len(candidates),
        )

        # Apply the bonus only to overall-purpose questions.
        purpose_query = any(
            word in query.lower().split()
            for word in ("goal", "purpose", "aim", "objective")
        )

        bonus = ABSTRACT_BONUS if purpose_query else 0.0

        intent_aware = apply_abstract_bonus(baseline, bonus)

        baseline_rank = find_evidence_rank(
            baseline,
            expected_pages,
        )

        intent_rank = find_evidence_rank(
            intent_aware,
            expected_pages,
        )

        abstract_rank = next(
            (
                rank
                for rank, item in enumerate(intent_aware, start=1)
                if item.get("chunk_id") == ABSTRACT_CHUNK_ID
            ),
            None,
        )

        print("\n" + "=" * 70)
        print(question["name"])
        print("Unique candidates:", len(candidates))
        print("Purpose intent detected:", purpose_query)
        print("Applied abstract bonus:", bonus)
        print("Baseline expected-evidence rank:", baseline_rank)
        print("Intent-aware evidence rank:", intent_rank)
        print("Intent-aware abstract rank:", abstract_rank)
        print("Intent-aware top 3:")

        for item in intent_aware[:3]:
            print(
                "  Pages:",
                item.get("source_pages"),
                "| Score:",
                round(item["adjusted_score"], 3),
            )

    print("\n" + "=" * 70)
    print("Evaluation complete:", total_queries, "questions")
    print("No production data or application code was changed.")


if __name__ == "__main__":
    main()
    