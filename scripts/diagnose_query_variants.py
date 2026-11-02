from scripts.evaluate_chunking import search_collection

COLLECTION = "document_chunks_eval_abstract"
ABSTRACT_CHUNK_ID = (
    "157fccdd123e2e64856a7d816d9dd5076853ec353afdce092e146f602d88d239"
)

QUERIES = [
    "What was the main goal of the report for Company X?",
    "What was the purpose and objective of this research report?",
    "What did the report aim to achieve for Company X?",
    "What is the main objective of the study described in the abstract?",
    "What was the objective of the software testing improvement research?",
]


def main():
    for query in QUERIES:
        candidates = search_collection(
            collection_name=COLLECTION,
            query=query,
            candidate_limit=60,
        )

        abstract_rank = None

        for rank, item in enumerate(candidates, start=1):
            if item.get("chunk_id") == ABSTRACT_CHUNK_ID:
                abstract_rank = rank
                break

        print(f"\nQuery: {query}")
        print(f"Abstract hybrid rank: {abstract_rank}")


if __name__ == "__main__":
    main()