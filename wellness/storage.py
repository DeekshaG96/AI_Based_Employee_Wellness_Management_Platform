"""SQLite persistence boundary for user authentication and mood telemetry."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from pathlib import Path
from typing import Any

from .errors import InputValidationError


def connect_database(path: str | Path = "wellness.db") -> sqlite3.Connection:
    """Open and initialize the local SQLite database with auto-migration."""
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS mood_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            source_type TEXT NOT NULL,
            source_name TEXT NOT NULL,
            original_text TEXT NOT NULL,
            cleaned_text TEXT NOT NULL,
            tokens TEXT NOT NULL,
            sentiment_label TEXT NOT NULL,
            neg_score REAL NOT NULL,
            neu_score REAL NOT NULL,
            pos_score REAL NOT NULL,
            compound_score REAL NOT NULL,
            dominant_emotion TEXT NOT NULL DEFAULT 'neutral',
            emotion_scores TEXT NOT NULL DEFAULT '{}',
            recommendation TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    connection.commit()

    # Auto-migrate any columns that might be absent from previous runs
    cursor = connection.execute("PRAGMA table_info(mood_entries)")
    existing_cols = {row["name"] for row in cursor.fetchall()}

    if "dominant_emotion" not in existing_cols:
        connection.execute(
            "ALTER TABLE mood_entries ADD COLUMN dominant_emotion TEXT NOT NULL DEFAULT 'neutral'"
        )
    if "emotion_scores" not in existing_cols:
        connection.execute(
            "ALTER TABLE mood_entries ADD COLUMN emotion_scores TEXT NOT NULL DEFAULT '{}'"
        )
    if "recommendation" not in existing_cols:
        connection.execute(
            "ALTER TABLE mood_entries ADD COLUMN recommendation TEXT NOT NULL DEFAULT ''"
        )
    connection.commit()

    return connection


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"{salt.hex()}${digest.hex()}"


def _check_password(password: str, encoded: str) -> bool:
    try:
        salt_hex, digest_hex = encoded.split("$", 1)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 120_000)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def register_user(connection: sqlite3.Connection, name: str, email: str, password: str) -> int:
    """Register a new user with salted PBKDF2 hashing."""
    name, email = name.strip(), email.strip().lower()
    if not name or not email or not password:
        raise InputValidationError("Name, email, and password are required.")
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise InputValidationError("Enter a valid email address.")
    if len(password) < 6:
        raise InputValidationError("Password must contain at least 6 characters.")
    try:
        cursor = connection.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, _hash_password(password)),
        )
        connection.commit()
        return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise InputValidationError("An account with that email already exists.") from exc


def authenticate_user(
    connection: sqlite3.Connection, email: str, password: str
) -> tuple[int, str] | None:
    """Verify user credentials against stored PBKDF2 hash."""
    row = connection.execute(
        "SELECT id, name, password_hash FROM users WHERE email = ?",
        (email.strip().lower(),),
    ).fetchone()
    if row and _check_password(password, row["password_hash"]):
        return int(row["id"]), str(row["name"])
    return None


def save_mood_entry(
    connection: sqlite3.Connection,
    user_id: int,
    source_type: str,
    source_name: str,
    original_text: str,
    cleaned_text: str,
    tokens: tuple[str, ...] | list[str],
    sentiment_label: str,
    neg_score: float,
    neu_score: float,
    pos_score: float,
    compound_score: float,
    dominant_emotion: str = "neutral",
    emotion_scores: dict[str, float] | str = "{}",
    recommendation: dict[str, Any] | str = "",
) -> int:
    """Insert a parsed mood, sentiment, and emotion analysis record into SQLite."""
    token_str = ", ".join(tokens)
    if isinstance(emotion_scores, dict):
        emotion_str = json.dumps(emotion_scores)
    else:
        emotion_str = str(emotion_scores)

    if isinstance(recommendation, dict):
        rec_str = json.dumps(recommendation)
    else:
        rec_str = str(recommendation)

    cursor = connection.execute(
        """
        INSERT INTO mood_entries (
            user_id, source_type, source_name, original_text, cleaned_text,
            tokens, sentiment_label, neg_score, neu_score, pos_score, compound_score,
            dominant_emotion, emotion_scores, recommendation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            source_type,
            source_name,
            original_text,
            cleaned_text,
            token_str,
            sentiment_label,
            neg_score,
            neu_score,
            pos_score,
            compound_score,
            dominant_emotion,
            emotion_str,
            rec_str,
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def get_user_mood_entries(
    connection: sqlite3.Connection, user_id: int | None = None
) -> list[dict[str, Any]]:
    """Retrieve mood entries for a specific user or all entries if user_id is None."""
    query = """
        SELECT id, user_id, source_type, source_name, original_text, cleaned_text,
               tokens, sentiment_label, neg_score, neu_score, pos_score, compound_score,
               dominant_emotion, emotion_scores, recommendation, created_at
        FROM mood_entries
        """
    params: tuple = ()
    if user_id is not None:
        query += " WHERE user_id = ?"
        params = (user_id,)
    query += " ORDER BY created_at DESC, id DESC"

    cursor = connection.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def delete_mood_entry(connection: sqlite3.Connection, entry_id: int, user_id: int) -> bool:
    """Delete a mood entry belonging to the specified user."""
    cursor = connection.execute(
        "DELETE FROM mood_entries WHERE id = ? AND user_id = ?",
        (entry_id, user_id),
    )
    connection.commit()
    return cursor.rowcount > 0
