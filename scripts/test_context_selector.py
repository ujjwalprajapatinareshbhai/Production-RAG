from app.context_selector import classify_query_intent


TEST_QUERIES = [
    "What was the main goal of the report for Company X?",
    "Why did the researchers conduct this study?",
    "How did the researchers identify testing problems at Company X?",
    "What testing knowledge were the developers missing?",
    "Compare the two test process improvement models.",
    "What is the name of the company studied in this report?",
]


def main():
    for query in TEST_QUERIES:
        intent = classify_query_intent(query)

        print(f"\nQuestion: {query}")
        print(f"Predicted intent: {intent}")


if __name__ == "__main__":
    main()