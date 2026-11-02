from scripts.evaluate_chunking import search_collection
from app.reranker import rerank_chunks

COLLECTION = "document_chunks_eval_abstract"
CANDIDATE_LIMIT = 60

ABSTRACT_CHUNK_ID = (
    "157fccdd123e2e64856a7d816d9dd5076853ec353afdce092e146f602d88d239"
)

QUESTIONS = [
    {
        "original": (
            "What challenges do small software companies face "
            "when improving software quality?"
        ),
        "expanded": (
            "What difficulties do small software companies face "
            "when improving software quality and testing practices?"
        ),
        "targeted": (
            "What challenges prevent small software companies "
            "from improving software testing and quality?"
        ),
        "expected_page": 6,
    },
    {
        "original": "What was the main goal of the report for Company X?",
        "expanded": (
            "What was the objective of the software testing "
            "improvement research?"
        ),
        "targeted": (
            "What was the aim of the research to improve "
            "Company X's software testing process?"
        ),
        "expected_page": 5,
    },
    {
        "original": (
            "How did the researchers identify testing problems "
            "at Company X?"
        ),
        "expanded": (
            "How did the researchers investigate and identify "
            "problems in Company X's current software testing process?"
        ),
        "targeted": (
            "Which methods did researchers use to investigate "
            "and identify testing problems at Company X?"
        ),
        "expected_page": 9,
    },
    {
        "original": "What testing knowledge were the developers missing?",
        "expanded": (
            "What testing skills, knowledge, and training did "
            "the developers at Company X lack?"
        ),
        "targeted": (
            "What software testing concepts, techniques, or "
            "practices did Company X's developers lack knowledge of?"
        ),
        "expected_page": 10,
    },
]


def merge_candidates(*candidate_lists):
    merged = {}

    for candidates in candidate_lists:
        for candidate in candidates:
            chunk_id = candidate.get("chunk_id")

            if chunk_id and chunk_id not in merged:
                merged[chunk_id] = candidate

    return list(merged.values())


def find_rank(results, predicate):
    for rank, item in enumerate(results, start=1):
        if predicate(item):
            return rank

    return None


def main():
    for number, question in enumerate(QUESTIONS, start=1):
        original = question["original"]
        expanded = question["expanded"]
        targeted = question["targeted"]
        expected_page = question["expected_page"]

        original_candidates = search_collection(
            collection_name=COLLECTION,
            query=original,
            candidate_limit=CANDIDATE_LIMIT,
        )

        expanded_candidates = search_collection(
            collection_name=COLLECTION,
            query=expanded,
            candidate_limit=CANDIDATE_LIMIT,
        )

        targeted_candidates = search_collection(
            collection_name=COLLECTION,
            query=targeted,
            candidate_limit=CANDIDATE_LIMIT,
        )

        candidates = merge_candidates(
            original_candidates,
            expanded_candidates,
            targeted_candidates,
        )

        print("\n" + "=" * 75)
        print(f"QUESTION {number}: {original}")
        print(f"Expected evidence page: {expected_page}")
        print(f"Unique merged candidates: {len(candidates)}")

        for label, query in [
            ("Original", original),
            ("Expanded", expanded),
            ("Targeted", targeted),
        ]:
            ranked = rerank_chunks(
                query=query,
                candidates=candidates,
                limit=len(candidates),
            )

            evidence_rank = find_rank(
                ranked,
                lambda item: expected_page in (
                    item.get("source_pages") or []
                ),
            )

            abstract_rank = find_rank(
                ranked,
                lambda item: (
                    item.get("chunk_id") == ABSTRACT_CHUNK_ID
                ),
            )

            print(f"\n{label} query:")
            print(f"  Expected evidence rank: {evidence_rank}")
            print(f"  Standalone abstract rank: {abstract_rank}")

            print("  Top 3:")
            for rank, item in enumerate(ranked[:3], start=1):
                print(
                    f"    {rank}. pages={item.get('source_pages')} "
                    f"score={item.get('reranker_score')}"
                )


if __name__ == "__main__":
    main()