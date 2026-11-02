import json
import logging

import ollama


logger = logging.getLogger(__name__)

INTENT_MODEL = "gemma4:cloud"

VALID_INTENTS = {
    "overview",
    "specific_detail",
    "comparison",
    "other",
}


def classify_query_intent(query: str) -> str:
    """Classify a document question using Gemma Cloud."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    try:
        response = ollama.chat(
            model=INTENT_MODEL,
            messages=[
                {
                     "role": "system",
                     "content": """You classify questions about a document into exactly one intent.

                Choose one of these intents:

- overview: asks for the document's overall purpose, main goal,
  motivation, broad challenges, key findings, major problems,
  overall conclusions, or a summary of the big picture.
  Questions asking about the main challenges faced by a group
  or organization should usually be overview when they ask broadly.

- specific_detail: asks for a particular fact, percentage, date,
  named method, specific process, individual finding, or precise
  detail that can be answered directly from a particular passage.

- comparison: asks to compare two or more methods, items, ideas,
  organizations, or approaches.

- other: questions that do not fit the categories above.

Examples:
Question: What are the main challenges discussed in this report?
Intent: overview

Question: What challenges do small software companies face when
improving software quality?
Intent: overview

Question: Summarize the key findings of this research.
Intent: overview

Question: What percentage of developers received formal testing training?
Intent: specific_detail

Question: Which testing process was implemented at Company X?
Intent: specific_detail

Question: What testing knowledge were the developers missing?
Intent: specific_detail

Question: Compare the two testing improvement models.
Intent: comparison

Return only a valid JSON object in this format:
{"intent": "overview"}

Do not include explanations or Markdown."""
                },
                {
                    "role": "user",
                    "content": query.strip(),
                },
            ],
        )

        raw_output = response.message.content.strip()
        result = json.loads(raw_output)

        if not isinstance(result, dict):
            logger.warning("Intent classifier returned a non-object JSON value.")
            return "other"

        intent = result.get("intent")

        if not isinstance(intent, str) or intent not in VALID_INTENTS:
            logger.warning("Intent classifier returned an invalid intent.")
            return "other"

        return intent

    except json.JSONDecodeError:
        logger.warning("Intent classifier returned invalid JSON.")
        return "other"

    except Exception:
        logger.exception(
            "Gemma Cloud intent classification failed; using fallback intent."
        )
        return "other"


def _query_requests_report_summary(query: str) -> bool:
    """Return True when the question asks for a report-level summary or purpose."""

    query_lower = query.lower()

    summary_phrases = (
        # Overall purpose / goal
        "overall purpose",
        "main purpose of this report",
        "main goal of this report",
        "purpose of this study",
        "purpose of this report",
        "purpose of this research",
        "objective of this study",
        "objective of this report",
        "objective of this research",
        "goal of this study",
        "goal of this research",
        # Why was it done
        "why was this study",
        "why was the study",
        "why was this research",
        "why was the research",
        "why was this report",
        "why was the report",
        # Summaries
        "summarize this report",
        "summarise this report",
        "summary of this report",
        "summarize the report",
        "summarise the report",
        "overall summary",
        # Findings / conclusions
        "key findings of this research",
        "overall conclusions",
    )

    return any(
        phrase in query_lower
        for phrase in summary_phrases
    )


def select_context(
    query: str,
    ranked_chunks: list[dict],
    limit: int = 3,
    intent: str | None = None,
) -> list[dict]:
    """Select passages for the final answer."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    if not ranked_chunks:
        return []

    abstracts = [
        chunk
        for chunk in ranked_chunks
        if chunk.get("section_type") == "abstract"
    ]

    if not abstracts:
        return ranked_chunks[:limit]

    if intent is None:
        intent = classify_query_intent(query)

    if intent not in VALID_INTENTS:
        intent = "other"

    should_prioritize_abstract = (
        intent == "overview"
        and _query_requests_report_summary(query)
    )

    if not should_prioritize_abstract:
        return ranked_chunks[:limit]

    selected = [abstracts[0]]

    for chunk in ranked_chunks:
        if chunk.get("section_type") == "abstract":
            continue

        selected.append(chunk)

        if len(selected) >= limit:
            break

    return selected[:limit]