from contextlib import nullcontext
from uuid import UUID
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import app.api as api


client = TestClient(api.app)

TEST_USER_ID = UUID("11111111-1111-1111-1111-111111111111")
TEST_TENANT_ID = UUID("22222222-2222-2222-2222-222222222222")

TEST_USER = {
    "user_id": TEST_USER_ID,
    "tenant_id": TEST_TENANT_ID,
    "email": "test@example.com",
    "role": "owner",
}


@pytest.fixture
def authenticated_client():
    """Provide a client with an authenticated test user."""

    api.app.dependency_overrides[api.get_current_user] = (
        lambda: TEST_USER
    )

    yield client

    api.app.dependency_overrides.pop(api.get_current_user, None)


# ---------------------------------------------------------
# Health and readiness
# ---------------------------------------------------------

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_check(monkeypatch):
    class FakeQdrantClient:
        def get_collection(self, collection_name):
            assert collection_name == api.settings.qdrant_collection

    class FakePostgresConnection:
        def execute(self, query):
            assert query == "SELECT 1"

    monkeypatch.setattr(
        api,
        "get_vector_client",
        lambda: FakeQdrantClient(),
    )

    monkeypatch.setattr(
        api,
        "get_connection",
        lambda: nullcontext(FakePostgresConnection()),
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------

def test_search_rejects_missing_token():
    response = client.post(
        "/search",
        json={"query": "Why is testing important?"},
    )

    assert response.status_code == 401


def test_search_rejects_invalid_token():
    response = client.post(
        "/search",
        json={"query": "Why is testing important?"},
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401


def test_search_rejects_revoked_membership(monkeypatch):
    monkeypatch.setattr(
        api,
        "decode_access_token",
        lambda token: {
            "sub": str(TEST_USER_ID),
            "tenant_id": str(TEST_TENANT_ID),
            "role": "owner",
        },
    )

    monkeypatch.setattr(
        api,
        "get_user_by_id",
        lambda user_id: {
            "user_id": TEST_USER_ID,
            "email": "test@example.com",
            "is_active": True,
        },
    )

    # No active membership means access must be denied.
    monkeypatch.setattr(
        api,
        "get_active_membership",
        lambda tenant_id, user_id: None,
    )

    response = client.post(
        "/search",
        json={"query": "Why is testing important?"},
        headers={"Authorization": "Bearer valid-looking-token"},
    )

    assert response.status_code == 403


def test_login_successfully(monkeypatch):
    user = {
        "user_id": TEST_USER_ID,
        "email": "test@example.com",
        "password_hash": "fake-password-hash",
        "is_active": True,
    }

    # Verify that the email is normalized before lookup.
    def fake_get_user_by_email(email):
        assert email == "test@example.com"
        return user

    monkeypatch.setattr(
        api,
        "get_user_by_email",
        fake_get_user_by_email,
    )

    # Simulate successful password verification.
    def fake_verify_password(password, password_hash):
        assert password == "correct-password"
        assert password_hash == "fake-password-hash"
        return True

    monkeypatch.setattr(
        api,
        "verify_password",
        fake_verify_password,
    )

    # Simulate an active tenant membership.
    def fake_get_active_membership(tenant_id, user_id):
        assert tenant_id == TEST_TENANT_ID
        assert user_id == TEST_USER_ID

        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "role": "member",
        }

    monkeypatch.setattr(
        api,
        "get_active_membership",
        fake_get_active_membership,
    )

    # Verify that the JWT uses the trusted database role.
    def fake_create_access_token(user_id, tenant_id, role):
        assert user_id == str(TEST_USER_ID)
        assert tenant_id == str(TEST_TENANT_ID)
        assert role == "member"

        return "fake-access-token"

    monkeypatch.setattr(
        api,
        "create_access_token",
        fake_create_access_token,
    )

    response = client.post(
        "/auth/login",
        json={
            "email": " TEST@EXAMPLE.COM ",
            "password": "correct-password",
            "tenant_id": str(TEST_TENANT_ID),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["access_token"] == "fake-access-token"
    assert data["token_type"] == "bearer"
    assert data["tenant_id"] == str(TEST_TENANT_ID)
    assert data["role"] == "member"
    assert data["expires_in"] == (
        api.settings.access_token_expire_minutes * 60
    )

def test_login_rejects_incorrect_password(monkeypatch):
    user = {
        "user_id": TEST_USER_ID,
        "email": "test@example.com",
        "password_hash": "fake-password-hash",
        "is_active": True,
    }

    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: user,
    )

    monkeypatch.setattr(
        api,
        "verify_password",
        lambda password, password_hash: False,
    )

    def unexpected_token_creation(*args, **kwargs):
        pytest.fail("JWT must not be generated for incorrect credentials.")

    monkeypatch.setattr(
        api,
        "create_access_token",
        unexpected_token_creation,
    )

    response = client.post(
        "/auth/login",
        json={
            "email": "test@example.com",
            "password": "incorrect-password",
            "tenant_id": str(TEST_TENANT_ID),
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_login_rejects_unknown_user(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: None,
    )

    def unexpected_password_verification(*args, **kwargs):
        pytest.fail("Password verification must not run for unknown users.")

    monkeypatch.setattr(
        api,
        "verify_password",
        unexpected_password_verification,
    )

    response = client.post(
        "/auth/login",
        json={
            "email": "unknown@example.com",
            "password": "correct-password",
            "tenant_id": str(TEST_TENANT_ID),
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."

def test_login_rejects_inactive_user(monkeypatch):
    user = {
        "user_id": TEST_USER_ID,
        "email": "test@example.com",
        "password_hash": "fake-password-hash",
        "is_active": False,
    }

    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: user,
    )

    monkeypatch.setattr(
        api,
        "verify_password",
        lambda password, password_hash: True,
    )

    def unexpected_membership_lookup(*args, **kwargs):
        pytest.fail("Inactive users must not reach membership lookup.")

    monkeypatch.setattr(
        api,
        "get_active_membership",
        unexpected_membership_lookup,
    )

    response = client.post(
        "/auth/login",
        json={
            "email": "test@example.com",
            "password": "correct-password",
            "tenant_id": str(TEST_TENANT_ID),
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."

def test_login_rejects_missing_tenant_membership(monkeypatch):
    user = {
        "user_id": TEST_USER_ID,
        "email": "test@example.com",
        "password_hash": "fake-password-hash",
        "is_active": True,
    }

    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: user,
    )

    monkeypatch.setattr(
        api,
        "verify_password",
        lambda password, password_hash: True,
    )

    monkeypatch.setattr(
        api,
        "get_active_membership",
        lambda tenant_id, user_id: None,
    )

    def unexpected_token_creation(*args, **kwargs):
        pytest.fail("JWT must not be generated without tenant membership.")

    monkeypatch.setattr(
        api,
        "create_access_token",
        unexpected_token_creation,
    )

    response = client.post(
        "/auth/login",
        json={
            "email": "test@example.com",
            "password": "correct-password",
            "tenant_id": str(TEST_TENANT_ID),
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "No active membership in the selected tenant."
    )

# ---------------------------------------------------------
# Invitation acceptance
# ---------------------------------------------------------

def test_accept_invitation_successfully(monkeypatch):
    expected_user = {
        "user_id": TEST_USER_ID,
        "email": "invited@example.com",
        "password_hash": "fake-password-hash",
        "is_active": True,
    }

    expected_invitation = {
        "invitation_id": UUID(
            "44444444-4444-4444-4444-444444444444"
        ),
        "tenant_id": TEST_TENANT_ID,
        "email": "invited@example.com",
        "role": "member",
        "accepted_at": datetime.now(timezone.utc),
    }

    # Mock user lookup.
    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: expected_user,
    )

    # Mock password verification.
    monkeypatch.setattr(
        api,
        "verify_password",
        lambda password, password_hash: True,
    )

    # Mock invitation token hashing.
    monkeypatch.setattr(
        api,
        "hash_invitation_token",
        lambda token: "a" * 64,
    )

    # Mock invitation acceptance.
    def fake_accept_invitation(token_hash, user_id):
        assert token_hash == "a" * 64
        assert user_id == TEST_USER_ID

        return expected_invitation

    monkeypatch.setattr(
        api,
        "accept_invitation",
        fake_accept_invitation,
    )

    # Mock JWT generation.
    def fake_create_access_token(user_id, tenant_id, role):
        assert user_id == str(TEST_USER_ID)
        assert tenant_id == str(TEST_TENANT_ID)
        assert role == "member"

        return "fake-access-token"

    monkeypatch.setattr(
        api,
        "create_access_token",
        fake_create_access_token,
    )

    response = client.post(
        "/auth/accept-invitation",
        json={
            "token": "raw-invitation-token",
            "email": "invited@example.com",
            "password": "correct-password-123",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["access_token"] == "fake-access-token"
    assert data["token_type"] == "bearer"
    assert data["tenant_id"] == str(TEST_TENANT_ID)
    assert data["role"] == "member"
    assert data["expires_in"] == (
        api.settings.access_token_expire_minutes * 60
    )


def test_accept_invitation_rejects_incorrect_password(monkeypatch):
    expected_user = {
        "user_id": TEST_USER_ID,
        "email": "invited@example.com",
        "password_hash": "fake-password-hash",
        "is_active": True,
    }

    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: expected_user,
    )

    # Simulate an incorrect password.
    monkeypatch.setattr(
        api,
        "verify_password",
        lambda password, password_hash: False,
    )

    # Invitation acceptance must never be called.
    def unexpected_acceptance(*args, **kwargs):
        pytest.fail(
            "Invitation must not be accepted with an incorrect password."
        )

    monkeypatch.setattr(
        api,
        "accept_invitation",
        unexpected_acceptance,
    )

    response = client.post(
        "/auth/accept-invitation",
        json={
            "token": "raw-invitation-token",
            "email": "invited@example.com",
            "password": "incorrect-password-123",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_accept_invitation_rejects_email_mismatch(monkeypatch):
    expected_user = {
        "user_id": TEST_USER_ID,
        "email": "different@example.com",
        "password_hash": "fake-password-hash",
        "is_active": True,
    }

    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: expected_user,
    )

    monkeypatch.setattr(
        api,
        "verify_password",
        lambda password, password_hash: True,
    )

    monkeypatch.setattr(
        api,
        "hash_invitation_token",
        lambda token: "a" * 64,
    )

    def fake_accept_invitation(token_hash, user_id):
        raise PermissionError(
            "Invitation email does not match user email."
        )

    monkeypatch.setattr(
        api,
        "accept_invitation",
        fake_accept_invitation,
    )

    response = client.post(
        "/auth/accept-invitation",
        json={
            "token": "raw-invitation-token",
            "email": "different@example.com",
            "password": "correct-password-123",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "Invitation email does not match user email."
    )


def test_accept_invitation_rejects_invalid_token(monkeypatch):
    expected_user = {
        "user_id": TEST_USER_ID,
        "email": "invited@example.com",
        "password_hash": "fake-password-hash",
        "is_active": True,
    }

    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: expected_user,
    )

    monkeypatch.setattr(
        api,
        "verify_password",
        lambda password, password_hash: True,
    )

    monkeypatch.setattr(
        api,
        "hash_invitation_token",
        lambda token: "b" * 64,
    )

    def fake_accept_invitation(token_hash, user_id):
        raise ValueError("Invitation not found.")

    monkeypatch.setattr(
        api,
        "accept_invitation",
        fake_accept_invitation,
    )

    response = client.post(
        "/auth/accept-invitation",
        json={
            "token": "invalid-invitation-token",
            "email": "invited@example.com",
            "password": "correct-password-123",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invitation not found."


def test_accept_invitation_rejects_inactive_user(monkeypatch):
    expected_user = {
        "user_id": TEST_USER_ID,
        "email": "invited@example.com",
        "password_hash": "fake-password-hash",
        "is_active": False,
    }

    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: expected_user,
    )

    monkeypatch.setattr(
        api,
        "verify_password",
        lambda password, password_hash: True,
    )

    def unexpected_acceptance(*args, **kwargs):
        pytest.fail("Inactive users must not accept invitations.")

    monkeypatch.setattr(
        api,
        "accept_invitation",
        unexpected_acceptance,
    )

    response = client.post(
        "/auth/accept-invitation",
        json={
            "token": "raw-invitation-token",
            "email": "invited@example.com",
            "password": "correct-password-123",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_accept_invitation_rejects_already_accepted(
    monkeypatch,
):
    expected_user = {
        "user_id": TEST_USER_ID,
        "email": "invited@example.com",
        "password_hash": "fake-password-hash",
        "is_active": True,
    }

    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: expected_user,
    )

    monkeypatch.setattr(
        api,
        "verify_password",
        lambda password, password_hash: True,
    )

    monkeypatch.setattr(
        api,
        "hash_invitation_token",
        lambda token: "a" * 64,
    )

    def fake_accept_invitation(token_hash, user_id):
        raise ValueError(
            "Invitation has already been accepted."
        )

    monkeypatch.setattr(
        api,
        "accept_invitation",
        fake_accept_invitation,
    )

    response = client.post(
        "/auth/accept-invitation",
        json={
            "token": "already-used-token",
            "email": "invited@example.com",
            "password": "correct-password-123",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invitation has already been accepted."
    )


@pytest.mark.parametrize(
    "error_message",
    [
        "Invitation has expired.",
        "Invitation has been revoked.",
    ],
)
def test_accept_invitation_rejects_expired_or_revoked(
    monkeypatch,
    error_message,
):
    expected_user = {
        "user_id": TEST_USER_ID,
        "email": "invited@example.com",
        "password_hash": "fake-password-hash",
        "is_active": True,
    }

    monkeypatch.setattr(
        api,
        "get_user_by_email",
        lambda email: expected_user,
    )

    monkeypatch.setattr(
        api,
        "verify_password",
        lambda password, password_hash: True,
    )

    monkeypatch.setattr(
        api,
        "hash_invitation_token",
        lambda token: "a" * 64,
    )

    def fake_accept_invitation(token_hash, user_id):
        raise ValueError(error_message)

    monkeypatch.setattr(
        api,
        "accept_invitation",
        fake_accept_invitation,
    )

    response = client.post(
        "/auth/accept-invitation",
        json={
            "token": "test-invitation-token",
            "email": "invited@example.com",
            "password": "correct-password-123",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == error_message


# ---------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------

def test_search_returns_results(monkeypatch, authenticated_client):
    expected_results = [
        {
            "score": 0.5,
            "filename": "example.pdf",
            "source_pages": [7, 8],
            "text": "Software testing helps improve quality.",
        }
    ]

    def fake_search_documents(**kwargs):
        assert kwargs["tenant_id"] == TEST_TENANT_ID
        assert kwargs["query"] == "Why is testing important?"
        assert kwargs["limit"] == 3
        return expected_results

    monkeypatch.setattr(
        api,
        "search_documents",
        fake_search_documents,
    )

    response = authenticated_client.post(
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


def test_search_rejects_whitespace_query(authenticated_client):
    response = authenticated_client.post(
        "/search",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_search_rejects_invalid_limit(authenticated_client):
    response = authenticated_client.post(
        "/search",
        json={
            "query": "Why is testing important?",
            "limit": 0,
        },
    )

    assert response.status_code == 422


def test_search_handles_value_error(
    monkeypatch,
    authenticated_client,
):
    def fake_search_documents(**kwargs):
        raise ValueError("Invalid search parameters.")

    monkeypatch.setattr(
        api,
        "search_documents",
        fake_search_documents,
    )

    response = authenticated_client.post(
        "/search",
        json={"query": "Why is testing important?"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid search parameters."


def test_search_handles_internal_error(
    monkeypatch,
    authenticated_client,
):
    def fake_search_documents(**kwargs):
        raise RuntimeError("Internal test failure")

    monkeypatch.setattr(
        api,
        "search_documents",
        fake_search_documents,
    )

    response = authenticated_client.post(
        "/search",
        json={"query": "Why is testing important?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Search failed due to an internal server error."
    )


# ---------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------

def test_ask_returns_answer(monkeypatch, authenticated_client):
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
        assert kwargs["tenant_id"] == TEST_TENANT_ID
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

    response = authenticated_client.post(
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


def test_ask_rejects_whitespace_query(authenticated_client):
    response = authenticated_client.post(
        "/ask",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_ask_handles_value_error(
    monkeypatch,
    authenticated_client,
):
    def fake_answer_question(**kwargs):
        raise ValueError("Question cannot be empty.")

    monkeypatch.setattr(
        api,
        "answer_question",
        fake_answer_question,
    )

    response = authenticated_client.post(
        "/ask",
        json={"query": "A valid question"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_ask_handles_internal_error(
    monkeypatch,
    authenticated_client,
):
    def fake_answer_question(**kwargs):
        raise RuntimeError("Internal test failure")

    monkeypatch.setattr(
        api,
        "answer_question",
        fake_answer_question,
    )

    response = authenticated_client.post(
        "/ask",
        json={"query": "A valid question"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Question answering failed. Check that Ollama is running "
        "and the configured model is available."
    )


# ---------------------------------------------------------
# Document retrieval and deletion
# ---------------------------------------------------------

TEST_DOCUMENT_ID = "a" * 64

OTHER_TENANT_ID = UUID(
    "33333333-3333-3333-3333-333333333333"
)


def test_get_document_uses_authenticated_tenant(
    monkeypatch,
    authenticated_client,
):
    expected_document = {
        "tenant_id": TEST_TENANT_ID,
        "document_id": TEST_DOCUMENT_ID,
        "filename": "sample.pdf",
        "status": "completed",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    def fake_get_document(tenant_id, document_id):
        assert tenant_id == TEST_TENANT_ID
        assert document_id == TEST_DOCUMENT_ID
        return expected_document

    monkeypatch.setattr(
        api,
        "get_document",
        fake_get_document,
    )

    response = authenticated_client.get(
        f"/documents/{TEST_DOCUMENT_ID}"
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == TEST_DOCUMENT_ID


def test_get_document_does_not_access_another_tenant_document(
    monkeypatch,
    authenticated_client,
):
    calls = []

    def fake_get_document(tenant_id, document_id):
        calls.append((tenant_id, document_id))

        # Simulate PostgreSQL tenant-scoped lookup.
        if tenant_id == OTHER_TENANT_ID:
            return {
                "tenant_id": OTHER_TENANT_ID,
                "document_id": TEST_DOCUMENT_ID,
                "filename": "other-tenant.pdf",
                "status": "completed",
            }

        return None

    monkeypatch.setattr(
        api,
        "get_document",
        fake_get_document,
    )

    response = authenticated_client.get(
        f"/documents/{TEST_DOCUMENT_ID}"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

    assert calls == [
        (TEST_TENANT_ID, TEST_DOCUMENT_ID)
    ]


def test_delete_document_uses_authenticated_tenant(
    monkeypatch,
    authenticated_client,
):
    expected_result = {
        "tenant_id": str(TEST_TENANT_ID),
        "document_id": TEST_DOCUMENT_ID,
        "filename": "sample.pdf",
        "status": "deleted",
    }

    def fake_delete_document(tenant_id, document_id):
        assert tenant_id == TEST_TENANT_ID
        assert document_id == TEST_DOCUMENT_ID
        return expected_result

    monkeypatch.setattr(
        api,
        "delete_document",
        fake_delete_document,
    )

    response = authenticated_client.delete(
        f"/documents/{TEST_DOCUMENT_ID}"
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": TEST_DOCUMENT_ID,
        "filename": "sample.pdf",
        "status": "deleted",
    }


def test_delete_document_cannot_delete_another_tenant_document(
    monkeypatch,
    authenticated_client,
):
    calls = []

    def fake_delete_document(tenant_id, document_id):
        calls.append((tenant_id, document_id))

        if tenant_id == OTHER_TENANT_ID:
            return {
                "tenant_id": str(OTHER_TENANT_ID),
                "document_id": document_id,
                "filename": "other-tenant.pdf",
                "status": "deleted",
            }

        raise ValueError("Document not found.")

    monkeypatch.setattr(
        api,
        "delete_document",
        fake_delete_document,
    )

    response = authenticated_client.delete(
        f"/documents/{TEST_DOCUMENT_ID}"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

    assert calls == [
        (TEST_TENANT_ID, TEST_DOCUMENT_ID)
    ]