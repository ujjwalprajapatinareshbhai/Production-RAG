from app.qdrant_hybrid_retriever import search_qdrant_hybrid
from app.reranker import rerank_chunks


QUERIES = [
    (
        "What broad improvements does the report propose "
        "for Company X's testing process?"
    ),
    (
        "What changes are recommended for Company X's software "
        "testing process, including the proposed improved workflow, "
        "testing practices, and implementation recommendations?"
    ),
]

COLLECTION = "document_chunks_eval_abstract"


def main():
    candidates = search_qdrant_hybrid(
        query=QUERIES[0],
        limit=60,
        candidate_limit=60,
        collection_name=COLLECTION,
    )

    print(f"Retrieved candidates: {len(candidates)}")

    # Run the cross-encoder separately with each query.
    scores_by_chunk = {}

    for query_index, query in enumerate(QUERIES, start=1):
        ranked = rerank_chunks(
            query=query,
            candidates=candidates,
            limit=len(candidates),
        )

        for chunk in ranked:
            chunk_key = chunk.get("chunk_id")

            if chunk_key is None:
                chunk_key = chunk["text"]

            scores_by_chunk.setdefault(
                chunk_key,
                {"chunk": chunk, "scores": []},
            )

            scores_by_chunk[chunk_key]["scores"].append(
                chunk["reranker_score"]
            )

        print(f"Query {query_index} reranking complete.")

    # Average the scores from both query formulations.
    combined = []

    for item in scores_by_chunk.values():
        result = item["chunk"].copy()
        scores = item["scores"]

        result["combined_score"] = sum(scores) / len(scores)
        combined.append(result)

    combined.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    print("\nTOP 10 BY AVERAGE RERANKER SCORE")
    print("=" * 70)

    for rank, chunk in enumerate(combined[:10], start=1):
        print(f"\nRank: {rank}")
        print(f"Pages: {chunk.get('source_pages', [])}")
        print(f"Combined score: {chunk['combined_score']:.4f}")
        print(f"Text: {chunk.get('text', '')[:300]}")

    print("\nEXPECTED-PAGE CHECK")
    expected_pages = {66, 67, 68, 70}

    for rank, chunk in enumerate(combined[:3], start=1):
        pages = set(chunk.get("source_pages", []))

        print(
            f"Rank {rank}: pages={sorted(pages)}, "
            f"overlaps_expected={bool(pages & expected_pages)}"
        )


if __name__ == "__main__":
    main()