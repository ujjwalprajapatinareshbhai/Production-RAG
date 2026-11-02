from app.search import search_documents


EVAL_COLLECTION = "document_chunks_eval_abstract"
TOP_K = 3
CANDIDATE_LIMIT = 60


QUESTIONS = [
    # Overview questions
    {
        "question": "What is the overall purpose of this report?",
        "expected_pages": [5, 6,7,8],
        "intent": "overview",
        "answerable": True,
    },
    {
        "question": "What challenges do small software companies face when improving software quality?",
        "expected_pages": [5, 6, 7],
        "intent": "overview",
        "answerable": True,
    },
    {
        "question": "What are the main weaknesses of Company X's current testing workflow?",
        "expected_pages": [42, 43, 44,57],
        "intent": "overview",
        "answerable": True,
    },
    {
        "question": "Why does the report consider Company X's current testing workflow insufficient?",
        "expected_pages": [57],
        "intent": "overview",
        "answerable": True,
    },
    {
        "question": "What broad improvements does the report propose for Company X's testing process?",
        "expected_pages": [40, 41, 52, 53, 66, 67, 68, 70],
        "intent": "overview",
        "answerable": True,
    },

    # Specific-detail questions
    {
        "question": "How did the researchers investigate Company X's existing testing process?",
        "expected_pages": [9],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "Which software development lifecycle did Company X use?",
        "expected_pages": [6, 11, 32],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "What percentage of developers spent less than 10% of project time on testing?",
        "expected_pages": [12],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "What percentage of participants said they sometimes created testing documentation?",
        "expected_pages": [13],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "What characterizes an organization at TMMi Level 1?",
        "expected_pages": [23],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "Which process areas are included in TMMi Level 2?",
        "expected_pages": [24],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "What training and certification does the report recommend for Company X's developers?",
        "expected_pages": [30],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "Which factors can discourage organizations from using automated testing tools?",
        "expected_pages": [31],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "What problem did Company X have with defect tracking?",
        "expected_pages": [37],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "What activities begin the improved test-planning phase?",
        "expected_pages": [47],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "Why should regression testing be performed after a feature change or defect fix?",
        "expected_pages": [51],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "What is the purpose of test monitoring and control in the improved workflow?",
        "expected_pages": [52, 53],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "What report is created during test closure, and who receives it?",
        "expected_pages": [56],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "What is the purpose of the non-functional testing process area?",
        "expected_pages": [63],
        "intent": "specific_detail",
        "answerable": True,
    },
    {
        "question": "What management perspective does the report identify as missing from its research?",
        "expected_pages": [71],
        "intent": "specific_detail",
        "answerable": True,
    },

    # Comparison questions
    {
        "question": "How does TMMi Level 3 differ from TMMi Level 2?",
        "expected_pages": [24, 25],
        "intent": "comparison",
        "answerable": True,
    },
    {
        "question": "How does the report map the CTP model against TMMi?",
        "expected_pages": [27, 28],
        "intent": "comparison",
        "answerable": True,
    },
    {
        "question": "How do the roles in the existing workflow differ from those in the proposed workflow?",
        "expected_pages": [42, 43, 67, 68],
        "intent": "comparison",
        "answerable": True,
    },
    {
        "question": "How does the proposed testing workflow differ from Company X's existing workflow?",
        "expected_pages": [42, 43, 44, 47, 51, 56, 57, 66, 67],
        "intent": "comparison",
        "answerable": True,
    },

    # Unanswerable questions
    {
        "question": "What was Company X's annual revenue in 2009?",
        "expected_pages": [],
        "intent": "other",
        "answerable": False,
    },
    {
        "question": "What was the exact stock market price of Company X on 1 March 2010?",
        "expected_pages": [],
        "intent": "other",
        "answerable": False,
    },
    {
        "question": "What is the chief executive's date of birth?",
        "expected_pages": [],
        "intent": "other",
        "answerable": False,
    },
]


def main():
    total_questions = len(QUESTIONS)

    # Dataset totals
    total_answerable = sum(
        1 for item in QUESTIONS if item["answerable"]
    )
    total_unanswerable = total_questions - total_answerable

    # Successfully searched questions
    answerable_evaluated = 0
    unanswerable_evaluated = 0

    # Retrieval results
    answerable_hits = 0
    unanswerable_with_results = 0
    abstract_hits = 0

    # Errors
    search_failures = 0

    print("=" * 90)
    print("CONTEXT RETRIEVAL EVALUATION")
    print("=" * 90)
    print(f"Evaluation collection: {EVAL_COLLECTION}")
    print(f"Total questions: {total_questions}")
    print(f"Answerable questions: {total_answerable}")
    print(f"Unanswerable questions: {total_unanswerable}")
    print(f"Top-K: {TOP_K}")
    print(f"Candidate limit: {CANDIDATE_LIMIT}")

    for index, item in enumerate(QUESTIONS, start=1):
        query = item["question"]
        expected_pages = set(item["expected_pages"])
        expected_intent = item["intent"]
        answerable = item["answerable"]

        print("\n" + "=" * 90)
        print(f"Question {index}/{total_questions}: {query}")
        print(f"Expected intent: {expected_intent}")
        print(f"Expected pages: {sorted(expected_pages) or 'None'}")
        print(f"Answerable from report: {answerable}")

        try:
            results = search_documents(
                query=query,
                limit=TOP_K,
                candidate_limit=CANDIDATE_LIMIT,
                collection_name=EVAL_COLLECTION,
            )
        except Exception as exc:
            search_failures += 1
            print(
                f"Search failed: {type(exc).__name__}: {exc}"
            )
            continue

        # Count a question as evaluated only if the search call completed.
        if answerable:
            answerable_evaluated += 1
        else:
            unanswerable_evaluated += 1

        if not results:
            print("No results returned.")

            if answerable:
                print(
                    "Answerable question: counted as a retrieval miss."
                )
            else:
                print(
                    "Unanswerable question: no context was retrieved. "
                    "This alone does not prove correct abstention."
                )

            continue

        relevant_found = False
        abstract_found = False

        for rank, chunk in enumerate(results, start=1):
            pages = set(chunk.get("source_pages") or [])
            section_type = chunk.get("section_type")

            # Relevant-page hit: at least one expected page appears
            # in the source pages of one of the selected chunks.
            if expected_pages.intersection(pages):
                relevant_found = True

            if section_type == "abstract":
                abstract_found = True

            print(f"\nSelected context rank: {rank}")
            print(f"Pages: {sorted(pages)}")
            print(f"Section type: {section_type}")
            print(f"Reranker score: {chunk.get('reranker_score')}")
            print(f"Text: {(chunk.get('text') or '')[:250]}")

        if abstract_found:
            abstract_hits += 1

        if not answerable:
            unanswerable_with_results += 1

        print(
            "\nRelevant page found in selected context: "
            f"{relevant_found}"
        )
        print(f"Abstract found in selected context: {abstract_found}")

        if answerable and relevant_found:
            answerable_hits += 1

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("EVALUATION SUMMARY")
    print("=" * 90)

    print(f"Evaluation collection: {EVAL_COLLECTION}")
    print(f"Total questions: {total_questions}")
    print(f"Total answerable questions: {total_answerable}")
    print(f"Total unanswerable questions: {total_unanswerable}")

    print("\nSearch execution")
    print(f"Successful answerable searches: {answerable_evaluated}")
    print(f"Successful unanswerable searches: {unanswerable_evaluated}")
    print(f"Search failures: {search_failures}")

    print("\nAnswerable-question retrieval")
    print(f"Relevant-page hits in top {TOP_K}: {answerable_hits}")

    if answerable_evaluated:
        evaluated_hit_rate = (
            answerable_hits / answerable_evaluated * 100
        )
        print(
            "Hit rate among successfully searched answerable questions: "
            f"{evaluated_hit_rate:.2f}%"
        )

    # This metric treats search failures as no successful hit.
    if total_answerable:
        dataset_hit_rate = answerable_hits / total_answerable * 100
        print(
            "Hit rate across all answerable questions "
            "(failures included in denominator): "
            f"{dataset_hit_rate:.2f}%"
        )

    print("\nUnanswerable-question retrieval")
    print(
        "Unanswerable questions that returned results: "
        f"{unanswerable_with_results}"
    )
    print(f"Questions with an abstract chunk in top {TOP_K}: {abstract_hits}")

    print("\nEvaluation notes:")
    print("- A relevant-page hit means at least one expected page appears in the selected context.")
    print("- This script evaluates retrieval, not the factual correctness of generated answers.")
    print("- Returning no results for an unanswerable question does not by itself prove correct abstention.")
    print("- Search failures are reported separately and are not silently counted as successful evaluations.")
    print("=" * 90)


if __name__ == "__main__":
    main()