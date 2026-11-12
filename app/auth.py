from datetime import datetime, timedelta, timezone
from typing import Any
import hashlib
import secrets
import jwt
from pwdlib import PasswordHash

from app.config import settings


# Password hashing using Argon2id.
password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using Argon2id.

    The returned hash is safe to store in PostgreSQL.
    """
    if not password:
        raise ValueError("Password cannot be empty.")

    return password_hasher.hash(password)

def generate_invitation_token() -> str:
    """
    Generate a cryptographically secure invitation token.

    The original token is returned only once so it can
    be delivered to the invited user.
    """
    return secrets.token_urlsafe(32)


def hash_invitation_token(token: str) -> str:
    """
    Generate a SHA-256 hash of an invitation token.

    Only this hash should be stored in PostgreSQL.
    """
    if not token:
        raise ValueError("Invitation token cannot be empty.")

    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """
    Verify a plain-text password against its stored hash.
    """
    if not plain_password or not hashed_password:
        return False

    return password_hasher.verify(
        plain_password,
        hashed_password,
    )


def create_access_token(
    user_id: str,
    tenant_id: str,
    role: str,
) -> str:
    """
    Create a signed JWT access token for a tenant member.
    """
    if not user_id or not tenant_id or not role:
        raise ValueError(
            "user_id, tenant_id, and role are required."
        )

    now = datetime.now(timezone.utc)

    expires_at = now + timedelta(
        minutes=settings.access_token_expire_minutes
    )

    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": expires_at,
        "type": "access",
    }

    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    return token


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Validate a JWT access token and return its claims.

    Raises jwt.InvalidTokenError when the token is invalid,
    expired, incorrectly signed, or has invalid claims.
    """
    if not token:
        raise jwt.InvalidTokenError("Token cannot be empty.")

    payload = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        options={
            "require": [
                "sub",
                "tenant_id",
                "role",
                "iss",
                "aud",
                "iat",
                "exp",
                "type",
            ]
        },
    )

    if payload["type"] != "access":
        raise jwt.InvalidTokenError("Invalid token type.")

    return payload