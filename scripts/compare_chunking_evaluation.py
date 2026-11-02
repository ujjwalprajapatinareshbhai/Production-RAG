from scripts.evaluate_context_selection import QUESTIONS
from app.search import search_documents


COLLECTIONS = {
    "current": "document_chunks_eval_current_v1",
    "page_aware": "document_chunks_eval_page_v1",
}

EVALUATION_K = (3, 5, 10)
MAX_K = max(EVALUATION_K)
CANDIDATE_LIMIT = 60


def evaluate_question(question_item, collection_name):
    """Evaluate one question using up to 10 final selected chunks."""

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
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank_at_10": reciprocal_rank_at_10,
        "results": results,
    }


def main():
    strategies = {
        name: {
            "evaluated": 0,
            "hits_at_k": {k: 0 for k in EVALUATION_K},
            "mrr10_sum": 0.0,
            # Preallocate to preserve question positions on errors.
            "details": [None] * len(QUESTIONS),
        }
        for name in COLLECTIONS
    }

    print("Questions:", len(QUESTIONS))
    print("Evaluation cutoffs:", EVALUATION_K)
    print("Candidate limit:", CANDIDATE_LIMIT)
    print("Evaluation is read-only.")
    print("Collections:", COLLECTIONS)

    for index, item in enumerate(QUESTIONS):
        question_number = index + 1

        print("\n" + "=" * 80)
        print(f"Question {question_number}/{len(QUESTIONS)}")
        print(item["question"])

        for strategy_name, collection_name in COLLECTIONS.items():
            try:
                outcome = evaluate_question(
                    question_item=item,
                    collection_name=collection_name,
                )
            except Exception as exc:
                print(
                    f"{strategy_name}: ERROR "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            stats = strategies[strategy_name]

            rank = outcome["first_relevant_rank"]
            rank_text = str(rank) if rank is not None else "not in top 10"

            if item["answerable"]:
                stats["evaluated"] += 1

                for k in EVALUATION_K:
                    stats["hits_at_k"][k] += int(
                        outcome["hits_at_k"][k]
                    )

                stats["mrr10_sum"] += (
                    outcome["reciprocal_rank_at_10"]
                )

            stats["details"][index] = {
                "question": item["question"],
                "answerable": item["answerable"],
                "hits_at_k": outcome["hits_at_k"],
                "rank": rank,
                "rr10": outcome["reciprocal_rank_at_10"],
            }

            hit_summary = " | ".join(
                f"Hit@{k}={outcome['hits_at_k'][k]}"
                for k in EVALUATION_K
            )

            print(
                f"{strategy_name:12} | "
                f"{hit_summary} | "
                f"first relevant rank={rank_text}"
            )

    print("\n" + "=" * 80)
    print("FINAL COMPARISON — ANSWERABLE QUESTIONS")
    print("=" * 80)

    for name, stats in strategies.items():
        count = stats["evaluated"]

        print(f"\nStrategy: {name}")
        print(f"Answerable questions evaluated: {count}")

        if count == 0:
            print("No questions evaluated successfully.")
            continue

        for k in EVALUATION_K:
            hits = stats["hits_at_k"][k]
            hit_rate = hits / count * 100

            print(
                f"Hit@{k}: {hits}/{count} "
                f"({hit_rate:.2f}%)"
            )

        mrr10 = stats["mrr10_sum"] / count
        print(f"MRR@10: {mrr10:.4f}")

    print("\n" + "=" * 80)
    print("QUESTION-LEVEL COMPARISON")
    print("=" * 80)

    improved = []
    regressed = []
    unchanged = []
    missing = []

    current = strategies["current"]["details"]
    page_aware = strategies["page_aware"]["details"]

    for index, item in enumerate(QUESTIONS):
        if not item["answerable"]:
            continue

        old = current[index]
        new = page_aware[index]

        if old is None or new is None:
            missing.append((index + 1, item["question"]))
            continue

        if old["rr10"] < new["rr10"]:
            improved.append((index + 1, item["question"]))
        elif old["rr10"] > new["rr10"]:
            regressed.append((index + 1, item["question"]))
        else:
            unchanged.append((index + 1, item["question"]))

    print(f"Improved by MRR@10: {len(improved)}")
    for number, question in improved:
        print(f"  + Q{number}: {question}")

    print(f"\nRegressed by MRR@10: {len(regressed)}")
    for number, question in regressed:
        print(f"  - Q{number}: {question}")

    print(f"\nUnchanged by MRR@10: {len(unchanged)}")
    for number, question in unchanged:
        print(f"  = Q{number}: {question}")

    print(f"\nMissing evaluations: {len(missing)}")
    for number, question in missing:
        print(f"  ! Q{number}: {question}")


if __name__ == "__main__":
    main()