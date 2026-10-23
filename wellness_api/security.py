"""Password hashing and opaque bearer-session management."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import get_session
from .models import AuthSession, User

bearer = HTTPBearer(auto_error=False)
SESSION_DAYS = 30
PBKDF2_ROUNDS = 310_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Verify new hashes and legacy SQLite hashes created by earlier MoodMentor versions."""
    try:
        if encoded.startswith("pbkdf2_sha256$"):
            _algorithm, rounds, salt_hex, digest_hex = encoded.split("$", 3)
            iterations = int(rounds)
        else:
            # Legacy representation: <salt hex>$<digest hex>, 120,000 SHA-256 rounds.
            salt_hex, digest_hex = encoded.split("$", 1)
            iterations = 120_000
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), iterations
        )
        return hmac.compare_digest(actual, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(36)
    db.add(
        AuthSession(
            user_id=user.id,
            token_hash=token_digest(token),
            expires_at=datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS),
        )
    )
    db.commit()
    return token


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_session),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in is required.")
    session = db.scalar(
        select(AuthSession).where(AuthSession.token_hash == token_digest(credentials.credentials))
    )
    now = datetime.now(timezone.utc)
    if session is None:
        raise HTTPException(status_code=401, detail="Session is invalid or expired.")
    expires = session.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires <= now:
        db.delete(session)
        db.commit()
        raise HTTPException(status_code=401, detail="Session is invalid or expired.")
    return session.user
