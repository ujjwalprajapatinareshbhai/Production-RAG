from qdrant_client import QdrantClient

from app.context_selector import select_context
from app.reranker import rerank_chunks


COLLECTION_NAME = "document_chunks_eval_abstract"

TEST_QUESTIONS = [
    "What was the main goal of the report for Company X?",
    "Why did the researchers conduct this study?",
    "How did the researchers identify testing problems at Company X?",
    "What testing knowledge were the developers missing?",
]


def load_evaluation_chunks() -> list[dict]:
    """Load passages from the isolated evaluation collection."""

    client = QdrantClient(url="http://localhost:6333")

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    chunks = [
        point.payload
        for point in points
        if point.payload and point.payload.get("text")
    ]

    client.close()
    return chunks


def main():
    chunks = load_evaluation_chunks()

    print(f"Loaded {len(chunks)} passages from {COLLECTION_NAME}")

    for query in TEST_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"QUESTION: {query}")

        # Experiment: rerank all stored passages for this question.
        ranked_chunks = rerank_chunks(
            query=query,
            candidates=chunks,
            limit=len(chunks),
        )

        selected_chunks = select_context(
            query=query,
            ranked_chunks=ranked_chunks,
            limit=3,
        )

        print("\nSelected context:")

        for position, chunk in enumerate(selected_chunks, start=1):
            pages = chunk.get(
                "source_pages",
                chunk.get("pages", []),
            )

            section = chunk.get("section_type", "body")

            print(
                f"\n{position}. Pages: {pages} | "
                f"Section: {section}"
            )
            print(chunk["text"][:350].replace("\n", " "))

        abstract_available = any(
            chunk.get("section_type") == "abstract"
            for chunk in chunks
        )

        abstract_selected = any(
            chunk.get("section_type") == "abstract"
            for chunk in selected_chunks
        )

        print("\nAbstract available:", abstract_available)
        print("Abstract selected:", abstract_selected)


if __name__ == "__main__":
    main()
    