from scripts.evaluate_chunking import search_collection


QUERY = "What was the main goal of the report for Company X?"
COLLECTION = "document_chunks_eval_abstract"


def main():
    for candidate_limit in (10, 30, 60, 86):
        candidates = search_collection(
            collection_name=COLLECTION,
            query=QUERY,
            candidate_limit=candidate_limit,
        )

        abstract_rank = None

        for rank, item in enumerate(candidates, start=1):
            pages = item.get("source_pages") or []
            text = item.get("text", "")

            # Identify the standalone abstract, not the mixed chunk.
            if pages == [5] and len(text) > 500:
                abstract_rank = rank
                break

        print(
            f"candidate_limit={candidate_limit}: "
            f"abstract_hybrid_rank={abstract_rank}"
        )


if __name__ == "__main__":
    main()