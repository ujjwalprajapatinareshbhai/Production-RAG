
from sentence_transformers import CrossEncoder
from app.config import settings

_RERANKER_MODEL = settings.reranker_model

_reranker = CrossEncoder(_RERANKER_MODEL)


def rerank_chunks(
    query: str,
    candidates: list[dict],
    limit: int = 3,
) -> list[dict]:
    """Rerank retrieved chunks by query-passage relevance."""
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    if not candidates:
        return []

    pairs = [
        (query.strip(), candidate["text"])
        for candidate in candidates
    ]

    scores = _reranker.predict(pairs)

    reranked = []

    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["reranker_score"] = float(score)
        reranked.append(result)

    reranked.sort(
        key=lambda result: result["reranker_score"],
        reverse=True,
    )

    return reranked[:limit]
