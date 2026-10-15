"""Small SQLite persistence boundary for the Streamlit milestone flow."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from pathlib import Path

from .errors import InputValidationError


def connect_database(path: str | Path = "wellness.db") -> sqlite3.Connection:
    """Open and initialize the local SQLite database."""
    connection = sqlite3.connect(path)
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
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), 120_000
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def register_user(connection: sqlite3.Connection, name: str, email: str, password: str) -> None:
    name, email = name.strip(), email.strip().lower()
    if not name or not email or not password:
        raise InputValidationError("Name, email, and password are required.")
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise InputValidationError("Enter a valid email address.")
    if len(password) < 8:
        raise InputValidationError("Password must contain at least 8 characters.")
    try:
        connection.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, _hash_password(password)),
        )
        connection.commit()
    except sqlite3.IntegrityError as exc:
        raise InputValidationError("An account with that email already exists.") from exc


def authenticate_user(
    connection: sqlite3.Connection, email: str, password: str
) -> tuple[int, str] | None:
    row = connection.execute(
        "SELECT id, name, password_hash FROM users WHERE email = ?",
        (email.strip().lower(),),
    ).fetchone()
    if row and _check_password(password, row[2]):
        return int(row[0]), str(row[1])
    return None
