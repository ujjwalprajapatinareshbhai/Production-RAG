import json

import pytest

from app.answer import _normalize_generated_answer
from app.answer import _source_ids_in_answer


def test_normal_json_answer():
    content = json.dumps({
        "answer": "Company X had no defect tracking system [S1].",
        "insufficient_evidence": False,
    })

    result = _normalize_generated_answer(content)

    assert result == {
        "answer": "Company X had no defect tracking system [S1].",
        "insufficient_evidence": False,
    }


def test_unwraps_nested_json_inside_markdown_fence():
    nested_answer = {
        "answer": "Company X had no defect tracking system [S1].",
        "insufficient_evidence": False,
    }

    content = json.dumps({
        "answer": (
            "Here is the answer:\n```json\n"
            + json.dumps(nested_answer)
            + "\n```"
        ),
        "insufficient_evidence": False,
    })

    result = _normalize_generated_answer(content)

    assert result == nested_answer


def test_preserves_true_insufficient_evidence_flag():
    nested_answer = {
        "answer": (
            "I could not find sufficient evidence in the documents."
        ),
        "insufficient_evidence": True,
    }

    content = json.dumps({
        "answer": (
            "```json\n"
            + json.dumps(nested_answer)
            + "\n```"
        ),
        "insufficient_evidence": False,
    })

    result = _normalize_generated_answer(content)

    assert result == nested_answer


def test_accepts_json_inside_outer_markdown_fence():
    content = (
        '```json\n'
        '{"answer": "Supported answer.", '
        '"insufficient_evidence": false}\n'
        '```'
    )

    result = _normalize_generated_answer(content)

    assert result == {
        "answer": "Supported answer.",
        "insufficient_evidence": False,
    }

def test_extracts_json_fence_after_leading_plain_text():
    content = (
        "Here is the answer in plain text.\n\n"
        "```json\n"
        '{"answer": "Company X lacked a defect tracking system [S1].", '
        '"insufficient_evidence": false}\n'
        "```"
    )

    result = _normalize_generated_answer(content)

    assert result == {
        "answer": "Company X lacked a defect tracking system [S1].",
        "insufficient_evidence": False,
    }


def test_plain_text_fallback():
    content = "Company X had no defect tracking system."

    result = _normalize_generated_answer(content)

    assert result == {
        "answer": content,
        "insufficient_evidence": False,
    }


def test_plain_text_insufficient_evidence_fallback():
    content = (
        "I could not find sufficient evidence in the available documents."
    )

    result = _normalize_generated_answer(content)

    assert result == {
        "answer": content,
        "insufficient_evidence": True,
    }


@pytest.mark.parametrize("content", ["", "   ", None])
def test_rejects_empty_or_non_text_content(content):
    with pytest.raises(ValueError):
        _normalize_generated_answer(content)


@pytest.mark.parametrize(
    "content",
    [
        "not valid JSON {",
        "[]",
        "null",
    ],
)
def test_invalid_json_uses_plain_text_fallback(content):
    result = _normalize_generated_answer(content)

    assert result == {
        "answer": content,
        "insufficient_evidence": False,
    }


def test_source_ids_in_single_citation():
    assert _source_ids_in_answer(
        "Company X used Waterfall [S1]."
    ) == {"S1"}


def test_source_ids_in_grouped_citation():
    assert _source_ids_in_answer(
        "Company X used Waterfall [S1, S2]."
    ) == {"S1", "S2"}


def test_source_ids_in_multiple_citation_groups():
    assert _source_ids_in_answer(
        "The process was sequential [S1, S2]. "
        "Testing occurred late [S3]."
    ) == {"S1", "S2", "S3"}


def test_source_ids_ignore_unbracketed_ids():
    assert _source_ids_in_answer(
        "S9 is mentioned in prose, but the evidence is [S1]."
    ) == {"S1"}