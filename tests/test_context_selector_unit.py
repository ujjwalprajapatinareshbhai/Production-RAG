import json
from unittest.mock import Mock, patch

import pytest

from app.context_selector import (
    classify_query_intent,
    select_context,
)


def make_chunk(
    page: int,
    section_type: str = "body",
) -> dict:
    return {
        "text": f"Passage from page {page}",
        "source_pages": [page],
        "section_type": section_type,
        "reranker_score": 1.0,
    }


def mock_ollama_response(content: str):
    response = Mock()
    response.message.content = content
    return response


def test_valid_intent_response():
    with patch("app.context_selector.ollama.chat") as chat:
        chat.return_value = mock_ollama_response(
            json.dumps({"intent": "overview"})
        )

        result = classify_query_intent("Why was this study conducted?")

    assert result == "overview"
    chat.assert_called_once()


def test_invalid_json_falls_back_to_other():
    with patch("app.context_selector.ollama.chat") as chat:
        chat.return_value = mock_ollama_response("not valid JSON")

        result = classify_query_intent("Why was this study conducted?")

    assert result == "other"


def test_unknown_intent_falls_back_to_other():
    with patch("app.context_selector.ollama.chat") as chat:
        chat.return_value = mock_ollama_response(
            '{"intent": "unknown_intent"}'
        )

        result = classify_query_intent("Why was this study conducted?")

    assert result == "other"


def test_cloud_failure_falls_back_to_other():
    with patch("app.context_selector.ollama.chat") as chat:
        chat.side_effect = RuntimeError("Simulated cloud failure")

        result = classify_query_intent("Why was this study conducted?")

    assert result == "other"


def test_empty_query_raises_error():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        classify_query_intent("   ")


def test_no_abstract_skips_cloud_call():
    chunks = [
        make_chunk(58),
        make_chunk(63),
    ]

    with patch("app.context_selector.ollama.chat") as chat:
        result = select_context(
            query="Why was this study conducted?",
            ranked_chunks=chunks,
            limit=2,
        )

    assert [chunk["source_pages"] for chunk in result] == [[58], [63]]
    chat.assert_not_called()


def test_overview_prioritizes_abstract():
    chunks = [
        make_chunk(58),
        make_chunk(63),
        make_chunk(5, "abstract"),
    ]

    with patch("app.context_selector.ollama.chat") as chat:
        chat.return_value = mock_ollama_response(
            '{"intent": "overview"}'
        )

        result = select_context(
            query="What is the overall purpose of this report?",
            ranked_chunks=chunks,
            limit=3,
        )

    assert [chunk["source_pages"] for chunk in result] == [
        [5],
        [58],
        [63],
    ]

    chat.assert_called_once()

def test_overview_topic_question_preserves_ranking():
    chunks = [
        make_chunk(66),
        make_chunk(67),
        make_chunk(5, "abstract"),
    ]

    result = select_context(
        query="What broad improvements does the report propose for Company X's testing process?",
        ranked_chunks=chunks,
        limit=3,
        intent="overview",
    )

    assert [chunk["source_pages"] for chunk in result] == [
        [66],
        [67],
        [5],
    ]


def test_specific_detail_preserves_ranking():
    chunks = [
        make_chunk(58),
        make_chunk(63),
        make_chunk(5, "abstract"),
    ]

    with patch("app.context_selector.ollama.chat") as chat:
        chat.return_value = mock_ollama_response(
            '{"intent": "specific_detail"}'
        )

        result = select_context(
            query="What testing practices were discussed?",
            ranked_chunks=chunks,
            limit=3,
        )

    assert [chunk["source_pages"] for chunk in result] == [
        [58],
        [63],
        [5],
    ]

def test_classifier_prompt_covers_broad_challenges_as_overview():
    with patch("app.context_selector.ollama.chat") as chat:
        chat.return_value = mock_ollama_response(
            '{"intent": "overview"}'
        )

        result = classify_query_intent(
            "What challenges do small software companies face "
            "when improving software quality?"
        )

    assert result == "overview"
    chat.assert_called_once()

    messages = chat.call_args.kwargs["messages"]
    system_prompt = messages[0]["content"]

    assert "broad challenges" in system_prompt
    assert "What challenges do small software companies face" in system_prompt


def test_classifier_prompt_distinguishes_specific_detail():
    with patch("app.context_selector.ollama.chat") as chat:
        chat.return_value = mock_ollama_response(
            '{"intent": "specific_detail"}'
        )

        result = classify_query_intent(
            "What percentage of developers received formal testing training?"
        )

    assert result == "specific_detail"
    chat.assert_called_once()

    messages = chat.call_args.kwargs["messages"]
    system_prompt = messages[0]["content"]

    assert "specific_detail" in system_prompt
    assert "What percentage of developers received formal testing training?" in system_prompt