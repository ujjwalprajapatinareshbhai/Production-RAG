from app.qdrant_hybrid_retriever import search_qdrant_hybrid
from app.reranker import rerank_chunks

EVAL_COLLECTION = "document_chunks_eval_abstract"

QUERY = (
    "What broad improvements does the report propose for Company X's "
    "testing process?"
)

REWRITTEN_QUERY = (
    "What changes are recommended for Company X's software testing process, "
    "including the proposed improved workflow, testing practices, "
    "and implementation recommendations?"
)

candidates = search_qdrant_hybrid(
    query=QUERY,
    limit=60,
    candidate_limit=60,
    collection_name=EVAL_COLLECTION,
)

original_ranked = rerank_chunks(
    query=QUERY,
    candidates=candidates,
    limit=len(candidates),
)

rewritten_ranked = rerank_chunks(
    query=REWRITTEN_QUERY,
    candidates=candidates,
    limit=len(candidates),
)

def get_chunk_key(chunk):
    return chunk.get("chunk_id") or chunk["text"]

original_scores = {
    get_chunk_key(chunk): chunk["reranker_score"]
    for chunk in original_ranked
}

rewritten_scores = {
    get_chunk_key(chunk): chunk["reranker_score"]
    for chunk in rewritten_ranked
}

combined = []

for chunk in candidates:
    key = get_chunk_key(chunk)

    result = chunk.copy()
    result["original_score"] = original_scores[key]
    result["rewritten_score"] = rewritten_scores[key]
    result["average_score"] = (
        result["original_score"] + result["rewritten_score"]
    ) / 2

    combined.append(result)

combined.sort(
    key=lambda chunk: chunk["average_score"],
    reverse=True,
)

interesting_pages = [
    {40, 41},
    {52, 53},
    {66, 67, 68, 69},
]

for page_group in interesting_pages:
    print("\n" + "=" * 70)
    print(f"INSPECTING PAGES: {sorted(page_group)}")
    print("=" * 70)

    matches = [
        chunk
        for chunk in combined
        if page_group.intersection(chunk.get("source_pages", []))
    ]

    for chunk in matches:
        print(f"\nPages: {chunk.get('source_pages')}")
        print(f"Average score: {chunk['average_score']:.4f}")
        print(f"Original score: {chunk['original_score']:.4f}")
        print(f"Rewritten score: {chunk['rewritten_score']:.4f}")
        print(f"Section type: {chunk.get('section_type')}")
        print(f"Text:\n{chunk.get('text', '')[:1200]}")
        print("-" * 70)