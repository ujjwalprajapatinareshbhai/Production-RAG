import pytest
from fastapi.testclient import TestClient

import app.api as api


client = TestClient(api.app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_results(monkeypatch):
    expected_results = [
        {
            "score": 0.5,
            "filename": "example.pdf",
            "source_pages": [7, 8],
            "text": "Software testing helps improve quality.",
        }
    ]

    def fake_search_documents(**kwargs):
        assert kwargs["query"] == "Why is testing important?"
        assert kwargs["limit"] == 3
        return expected_results

    monkeypatch.setattr(
        api,
        "search_documents",
        fake_search_documents,
    )

    response = client.post(
        "/search",
        json={
            "query": "Why is testing important?",
            "limit": 3,
            "candidate_limit": 10,
        },
    )

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["results"] == expected_results


def test_search_rejects_whitespace_query():
    response = client.post(
        "/search",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_search_rejects_invalid_limit():
    response = client.post(
        "/search",
        json={
            "query": "Why is testing important?",
            "limit": 0,
        },
    )

    assert response.status_code == 422


def test_search_handles_value_error(monkeypatch):
    def fake_search_documents(**kwargs):
        raise ValueError("Invalid search parameters.")

    monkeypatch.setattr(
        api,
        "search_documents",
        fake_search_documents,
    )

    response = client.post(
        "/search",
        json={"query": "Why is testing important?"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid search parameters."


def test_search_handles_internal_error(monkeypatch):
    def fake_search_documents(**kwargs):
        raise RuntimeError("Internal test failure")

    monkeypatch.setattr(
        api,
        "search_documents",
        fake_search_documents,
    )

    response = client.post(
        "/search",
        json={"query": "Why is testing important?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Search failed due to an internal server error."
    )


def test_readiness_check(monkeypatch):
    class FakeQdrantClient:
        def get_collection(self, collection_name):
            assert collection_name == api.settings.qdrant_collection

    monkeypatch.setattr(
        api,
        "get_vector_client",
        lambda: FakeQdrantClient(),
    )
    monkeypatch.setattr(
        api,
        "list_completed_document_ids",
        lambda: [],
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_ask_returns_answer(monkeypatch):
    expected_result = {
        "question": "What problem did Company X have with defect tracking?",
        "answer": "Company X had no defect tracking system [S1].",
        "sources": [
            {
                "source_id": "S1",
                "filename": "example.pdf",
                "document_id": "test-document",
                "pages": [37],
                "chunk_id": "test-chunk",
            }
        ],
        "insufficient_evidence": False,
    }

    def fake_answer_question(**kwargs):
        assert kwargs["query"] == (
            "What problem did Company X have with defect tracking?"
        )
        assert kwargs["limit"] == 3
        assert kwargs["candidate_limit"] == 10
        assert kwargs["document_id"] == "test-document"
        return expected_result

    monkeypatch.setattr(
        api,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/ask",
        json={
            "query": (
                "What problem did Company X have with defect tracking?"
            ),
            "limit": 3,
            "candidate_limit": 10,
            "document_id": "test-document",
        },
    )

    assert response.status_code == 200
    assert response.json() == expected_result


def test_ask_rejects_whitespace_query():
    response = client.post(
        "/ask",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_ask_handles_value_error(monkeypatch):
    def fake_answer_question(**kwargs):
        raise ValueError("Question cannot be empty.")

    monkeypatch.setattr(
        api,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/ask",
        json={"query": "A valid question"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_ask_handles_internal_error(monkeypatch):
    def fake_answer_question(**kwargs):
        raise RuntimeError("Internal test failure")

    monkeypatch.setattr(
        api,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/ask",
        json={"query": "A valid question"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Question answering failed. Check that Ollama is running "
        "and the configured model is available."
    )