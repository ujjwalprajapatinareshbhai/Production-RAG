from app.qdrant_hybrid_retriever import search_qdrant_hybrid

query = "Which software development lifecycle did Company X use?"

results = search_qdrant_hybrid(
    query=query,
    limit=60,
    candidate_limit=60,
    collection_name="document_chunks_eval_abstract",
)

print("Candidates returned:", len(results))

matches = []

for rank, result in enumerate(results, start=1):
    text = result.get("text") or ""

    if "current software process used in company X" in text.lower():
        matches.append((rank, result))

print("Chunks containing the explicit page 11 statement:", len(matches))

for rank, result in matches:
    print(f"\n--- Hybrid rank {rank} ---")
    print("Pages:", result.get("source_pages"))
    print("Score:", result.get("score"))
    print("Text:", result.get("text", "")[:1000])
