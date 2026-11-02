from app.context_selector import classify_query_intent
from app.qdrant_hybrid_retriever import search_qdrant_hybrid
from app.reranker import rerank_chunks

EVAL_COLLECTION = "document_chunks_eval_abstract"

REWRITTEN_QUERY = (
    "What changes are recommended for Company X's software testing process, "
    "including the proposed improved workflow, testing practices, "
    "and implementation recommendations?"
)

# Import the existing evaluation questions.
from scripts.evaluate_context_selection import QUESTIONS


def chunk_key(chunk):
    return chunk.get("chunk_id") or chunk["text"]


def evaluate_question(question_data):
    query = question_data["question"]
    expected_pages = set(question_data["expected_pages"])

    # Retrieve candidates once so both strategies use the same evidence.
    candidates = search_qdrant_hybrid(
        query=query,
        limit=60,
        candidate_limit=60,
        collection_name=EVAL_COLLECTION,
    )

    # Strategy A: baseline reranking.
    baseline = rerank_chunks(
        query=query,
        candidates=candidates,
        limit=len(candidates),
    )

    # Strategy B: average scores from the original and rewritten queries.
    rewritten = rerank_chunks(
        query=REWRITTEN_QUERY,
        candidates=candidates,
        limit=len(candidates),
    )

    rewritten_scores = {
        chunk_key(chunk): chunk["reranker_score"]
        for chunk in rewritten
    }

    combined = []

    for chunk in candidates:
        key = chunk_key(chunk)

        result = chunk.copy()
        result["average_score"] = (
            next(
                item["reranker_score"]
                for item in baseline
                if chunk_key(item) == key
            )
            + rewritten_scores[key]
        ) / 2

        combined.append(result)

    combined.sort(
        key=lambda item: item["average_score"],
        reverse=True,
    )

    baseline_top3 = baseline[:3]
    combined_top3 = combined[:3]

    def relevant(chunk):
        return bool(
            expected_pages.intersection(
                chunk.get("source_pages", [])
            )
        )

    return {
        "question": query,
        "answerable": question_data["answerable"],
        "baseline_hit": (
            any(relevant(chunk) for chunk in baseline_top3)
            if expected_pages
            else None
        ),
        "rewritten_hit": (
            any(relevant(chunk) for chunk in combined_top3)
            if expected_pages
            else None
        ),
        "baseline_pages": [
            chunk.get("source_pages", [])
            for chunk in baseline_top3
        ],
        "rewritten_pages": [
            chunk.get("source_pages", [])
            for chunk in combined_top3
        ],
    }


def main():
    results = []

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\nEvaluating {index}/{len(QUESTIONS)}")
        print(question["question"])

        result = evaluate_question(question)
        results.append(result)

        print("Baseline pages:", result["baseline_pages"])
        print("Rewritten pages:", result["rewritten_pages"])

    answerable = [
        result
        for result in results
        if result["answerable"]
    ]

    baseline_hits = sum(
        result["baseline_hit"] for result in answerable
    )

    rewritten_hits = sum(
        result["rewritten_hit"] for result in answerable
    )

    total = len(answerable)

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Answerable questions: {total}")
    print(
        f"Baseline Hit@3: {baseline_hits}/{total} "
        f"({baseline_hits / total * 100:.2f}%)"
    )
    print(
        f"Query-rewriting Hit@3: {rewritten_hits}/{total} "
        f"({rewritten_hits / total * 100:.2f}%)"
    )

    improved = sum(
        result["baseline_hit"] is False
        and result["rewritten_hit"] is True
        for result in answerable
    )

    regressed = sum(
        result["baseline_hit"] is True
        and result["rewritten_hit"] is False
        for result in answerable
    )

    unchanged = total - improved - regressed

    print(f"Questions improved: {improved}")
    print(f"Questions regressed: {regressed}")
    print(f"Questions unchanged: {unchanged}")


if __name__ == "__main__":
    main()
    