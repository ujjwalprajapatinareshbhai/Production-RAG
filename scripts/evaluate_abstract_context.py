from qdrant_client import QdrantClient
from app.reranker import rerank_chunks


COLLECTION = "document_chunks_eval_abstract"

ABSTRACT_CHUNK_ID = (
    "157fccdd123e2e64856a7d816d9dd5076853ec353afdce092e146f602d88d239"
)

QUERY = "What was the main goal of the report for Company X?"


def main():
    client = QdrantClient(url="http://localhost:6333")

    points, _ = client.scroll(
        collection_name=COLLECTION,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    candidates = [
        point.payload
        for point in points
        if point.payload and point.payload.get("text")
    ]

    print("Stored passages loaded:", len(candidates))

    abstract = next(
        (
            item
            for item in candidates
            if item.get("chunk_id") == ABSTRACT_CHUNK_ID
        ),
        None,
    )

    if abstract is None:
        raise RuntimeError("Standalone abstract was not found.")

    # Experimental baseline: rerank all stored evaluation passages.
    ranked = rerank_chunks(
        query=QUERY,
        candidates=candidates,
        limit=len(candidates),
    )

    print("\n" + "=" * 70)
    print("STRATEGY A: TOP 3 RERANKED PASSAGES")
    print("=" * 70)

    for rank, item in enumerate(ranked[:3], start=1):
        print(f"\nRank {rank} | Pages: {item.get('source_pages')}")
        print(item.get("text", "")[:800])

    # Experimental context selection: always include the abstract,
    # followed by the two highest-ranked non-abstract passages.
    context = [abstract]

    for item in ranked:
        if item.get("chunk_id") == ABSTRACT_CHUNK_ID:
            continue

        context.append(item)

        if len(context) == 3:
            break

    print("\n" + "=" * 70)
    print("STRATEGY B: ABSTRACT + TOP 2 OTHER PASSAGES")
    print("=" * 70)

    for rank, item in enumerate(context, start=1):
        print(f"\nContext {rank} | Pages: {item.get('source_pages')}")
        print(item.get("text", "")[:800])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Strategy A pages:", [
        item.get("source_pages") for item in ranked[:3]
    ])
    print("Strategy B pages:", [
        item.get("source_pages") for item in context
    ])
    print("No vectors, payloads, or production data were modified.")
    print(
        "Note: this is a context-selection experiment over all stored "
        "evaluation passages, not a comparison of hybrid retrieval quality."
    )


if __name__ == "__main__":
    main()