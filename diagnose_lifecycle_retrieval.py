from app.qdrant_hybrid_retriever import search_qdrant_hybrid
from app.reranker import rerank_chunks

QUERY = "Which software development lifecycle did Company X use?"
TARGET = "current software process used in company x"

# Retrieve the original chunk from the evaluation collection.
candidates = search_qdrant_hybrid(
    query=QUERY,
    limit=60,
    candidate_limit=60,
    collection_name="document_chunks_eval_abstract",
)

target_chunk = next(
    (
        chunk
        for chunk in candidates
        if TARGET in (chunk.get("text") or "").lower()
    ),
    None,
)

if target_chunk is None:
    raise RuntimeError("Waterfall chunk was not found in hybrid candidates.")

# Keep the original retrieved chunk unchanged.
original_result = rerank_chunks(
    query=QUERY,
    candidates=[target_chunk],
    limit=1,
)[0]

# Diagnostic passage containing the same answer, without unrelated material.
short_passage = (
    "The current software process used in Company X is based on "
    "the Waterfall model. Its phases are requirements analysis, "
    "architecture, design, implementation, testing, release, "
    "and maintenance."
)

short_candidate = target_chunk.copy()
short_candidate["text"] = short_passage

short_result = rerank_chunks(
    query=QUERY,
    candidates=[short_candidate],
    limit=1,
)[0]

print("Query:", QUERY)
print("\nOriginal chunk")
print("Pages:", original_result.get("source_pages"))
print("Characters:", len(original_result["text"]))
print("Reranker score:", original_result["reranker_score"])

print("\nShort diagnostic passage")
print("Characters:", len(short_passage))
print("Reranker score:", short_result["reranker_score"])

print("\nScore difference (short - original):")
print(
    short_result["reranker_score"]
    - original_result["reranker_score"]
)