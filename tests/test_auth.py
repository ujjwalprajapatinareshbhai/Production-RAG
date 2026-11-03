import jwt
import pytest

from app.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hashing():
    password = "StrongPassword123!"

    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("WrongPassword123!", hashed)


def test_empty_password_is_rejected():
    with pytest.raises(ValueError):
        hash_password("")


def test_create_and_decode_access_token():
    token = create_access_token(
        user_id="user-123",
        tenant_id="tenant-456",
        role="admin",
    )

    payload = decode_access_token(token)

    assert payload["sub"] == "user-123"
    assert payload["tenant_id"] == "tenant-456"
    assert payload["role"] == "admin"
    assert payload["type"] == "access"


def test_invalid_token_is_rejected():
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token("invalid.token.value")


def test_modified_token_is_rejected():
    token = create_access_token(
        user_id="user-123",
        tenant_id="tenant-456",
        role="member",
    )

    modified_token = token[:-5] + "abcde"

    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(modified_token)