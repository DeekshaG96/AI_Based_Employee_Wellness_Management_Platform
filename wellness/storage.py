"""SQLite persistence boundary for user authentication and mood telemetry."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Any

from .errors import InputValidationError


def normalize_role(role: str | None) -> str:
    """Normalize stored role names to 'admin' or 'employee'."""
    return "admin" if str(role or "").strip().lower() == "admin" else "employee"


def admin_email_allowlist() -> set[str]:
    """Return trusted operator-configured admin emails from ADMIN_EMAILS env or Streamlit secrets."""
    raw = os.getenv("ADMIN_EMAILS", "")
    if not raw:
        try:
            import streamlit as st

            raw = str(st.secrets.get("ADMIN_EMAILS", "") or "")
        except Exception:
            raw = ""
    return {
        part.strip().lower()
        for part in raw.replace(";", ",").split(",")
        if part.strip()
    }


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
            role TEXT NOT NULL DEFAULT 'employee',
            profile_details TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            journal_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            report_data TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_trails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT NOT NULL,
            journal_id INTEGER,
            recommendation TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recommendation_id INTEGER,
            report_data TEXT NOT NULL DEFAULT '{}',
            accepted_or_rejected TEXT NOT NULL DEFAULT 'accepted',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT NOT NULL,
            report_data TEXT NOT NULL DEFAULT '{}',
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
    user_cursor = connection.execute("PRAGMA table_info(users)")
    existing_user_cols = {row["name"] for row in user_cursor.fetchall()}
    if "role" not in existing_user_cols:
        connection.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'employee'")
    if "profile_details" not in existing_user_cols:
        connection.execute("ALTER TABLE users ADD COLUMN profile_details TEXT NOT NULL DEFAULT '{}'")

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
    """Register a new user with salted PBKDF2 hashing (defaults strictly to employee unless allowlisted)."""
    name, email = name.strip(), email.strip().lower()
    if not name or not email or not password:
        raise InputValidationError("Name, email, and password are required.")
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise InputValidationError("Enter a valid email address.")
    if len(password) < 6:
        raise InputValidationError("Password must contain at least 6 characters.")
    assigned_role = "admin" if email in admin_email_allowlist() else "employee"
    try:
        cursor = connection.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (name, email, _hash_password(password), assigned_role),
        )
        connection.commit()
        return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise InputValidationError("An account with that email already exists.") from exc


def set_user_role(connection: sqlite3.Connection, email: str, role: str) -> bool:
    """Trusted server/CLI helper to promote or demote an existing user's persisted role."""
    clean_email = email.strip().lower()
    target_role = normalize_role(role)
    cursor = connection.execute(
        "UPDATE users SET role = ? WHERE email = ?",
        (target_role, clean_email),
    )
    connection.commit()
    return cursor.rowcount > 0


def bootstrap_admin_from_env(connection: sqlite3.Connection) -> None:
    """Provision or sync operator-configured admin accounts from environment or Streamlit secrets."""
    allowlist = admin_email_allowlist()
    for adm_email in allowlist:
        connection.execute("UPDATE users SET role = 'admin' WHERE email = ?", (adm_email,))

    boot_email = os.getenv("ADMIN_BOOTSTRAP_EMAIL", "").strip().lower()
    boot_pass = os.getenv("ADMIN_BOOTSTRAP_PASSWORD", "")
    boot_name = os.getenv("ADMIN_BOOTSTRAP_NAME", "Platform Administrator").strip() or "Platform Administrator"
    if not boot_email:
        try:
            import streamlit as st

            boot_email = str(st.secrets.get("ADMIN_BOOTSTRAP_EMAIL", "") or "").strip().lower()
            boot_pass = str(st.secrets.get("ADMIN_BOOTSTRAP_PASSWORD", "") or "")
            boot_name = str(st.secrets.get("ADMIN_BOOTSTRAP_NAME", boot_name) or boot_name).strip()
        except Exception:
            pass

    if boot_email and len(boot_pass) >= 6:
        row = connection.execute("SELECT id, role FROM users WHERE email = ?", (boot_email,)).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, 'admin')",
                (boot_name, boot_email, _hash_password(boot_pass)),
            )
        elif normalize_role(row["role"]) != "admin":
            connection.execute("UPDATE users SET role = 'admin' WHERE id = ?", (row["id"],))
    connection.commit()


def authenticate_user_with_role(
    connection: sqlite3.Connection, email: str, password: str
) -> tuple[int, str, str] | None:
    """Verify user credentials against stored PBKDF2 hash and return (id, name, role)."""
    clean_email = email.strip().lower()
    row = connection.execute(
        "SELECT id, name, password_hash, role FROM users WHERE email = ?",
        (clean_email,),
    ).fetchone()
    if row and _check_password(password, row["password_hash"]):
        persisted_role = normalize_role(row["role"])
        if clean_email in admin_email_allowlist() and persisted_role != "admin":
            connection.execute("UPDATE users SET role = 'admin' WHERE id = ?", (row["id"],))
            connection.commit()
            persisted_role = "admin"
        return int(row["id"]), str(row["name"]), persisted_role
    return None


def authenticate_user(
    connection: sqlite3.Connection, email: str, password: str
) -> tuple[int, str] | None:
    """Verify user credentials against stored PBKDF2 hash."""
    res = authenticate_user_with_role(connection, email, password)
    if res is None:
        return None
    return res[0], res[1]


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


# ---------------------------------------------------------------------------
# 6 DATABASE ENTITIES PERSISTENCE & RETRIEVAL
# ---------------------------------------------------------------------------

def save_journal_entry(connection: sqlite3.Connection, user_id: int, journal_text: str) -> int:
    """Save 2. Journal entries (Journal text)."""
    cursor = connection.execute(
        "INSERT INTO journal_entries (user_id, journal_text) VALUES (?, ?)",
        (user_id, journal_text),
    )
    connection.commit()
    return int(cursor.lastrowid)


def save_recommendation(connection: sqlite3.Connection, user_id: int | None, report_data: dict[str, Any] | str) -> int:
    """Save 3. Recommendations (Recommendation id, user id, report data)."""
    data_str = json.dumps(report_data) if isinstance(report_data, dict) else str(report_data)
    cursor = connection.execute(
        "INSERT INTO recommendations (user_id, report_data) VALUES (?, ?)",
        (user_id, data_str),
    )
    connection.commit()
    return int(cursor.lastrowid)


def save_audit_trail(
    connection: sqlite3.Connection,
    user_id: int | None,
    username: str,
    journal_id: int | None,
    recommendation: str,
) -> int:
    """Save 4. Audit trails (User id, username, journal id, recommendation)."""
    cursor = connection.execute(
        "INSERT INTO audit_trails (user_id, username, journal_id, recommendation) VALUES (?, ?, ?, ?)",
        (user_id, username, journal_id, recommendation),
    )
    connection.commit()
    return int(cursor.lastrowid)


def save_user_feedback(
    connection: sqlite3.Connection,
    recommendation_id: int | None,
    report_data: dict[str, Any] | str,
    accepted_or_rejected: str = "accepted",
) -> int:
    """Save 5. User feedback (recommendation id, report data, accepted/rejected)."""
    data_str = json.dumps(report_data) if isinstance(report_data, dict) else str(report_data)
    cursor = connection.execute(
        "INSERT INTO user_feedback (recommendation_id, report_data, accepted_or_rejected) VALUES (?, ?, ?)",
        (recommendation_id, data_str, accepted_or_rejected),
    )
    connection.commit()
    return int(cursor.lastrowid)


def save_report(
    connection: sqlite3.Connection,
    user_id: int | None,
    username: str,
    report_data: dict[str, Any] | str,
) -> int:
    """Save 6. Reports (Report id, user id, username, timestamp)."""
    data_str = json.dumps(report_data) if isinstance(report_data, dict) else str(report_data)
    cursor = connection.execute(
        "INSERT INTO reports (user_id, username, report_data) VALUES (?, ?, ?)",
        (user_id, username, data_str),
    )
    connection.commit()
    return int(cursor.lastrowid)


def _redact_report_dict(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    cleaned = dict(payload)
    for key in ("original_text", "cleaned_text", "processed_text", "vader_input_text"):
        if key in cleaned:
            cleaned[key] = "[Redacted]"
    return cleaned


def get_all_database_entities(
    connection: sqlite3.Connection,
    redact_private_text: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve stored database entities, optionally redacting private reflection text."""
    users_rows = connection.execute("SELECT id, name, email, role, profile_details, created_at FROM users ORDER BY id ASC").fetchall()
    journal_rows = connection.execute("SELECT id, user_id, journal_text, created_at AS timestamp FROM journal_entries ORDER BY id DESC").fetchall()
    rec_rows = connection.execute("SELECT id, user_id, report_data, created_at FROM recommendations ORDER BY id DESC").fetchall()
    audit_rows = connection.execute("SELECT id, user_id, username, journal_id, recommendation, created_at AS timestamp FROM audit_trails ORDER BY id DESC").fetchall()
    feedback_rows = connection.execute("SELECT id, recommendation_id, report_data, accepted_or_rejected, created_at AS timestamp FROM user_feedback ORDER BY id DESC").fetchall()
    report_rows = connection.execute("SELECT id, user_id, username, report_data, created_at AS timestamp FROM reports ORDER BY id DESC").fetchall()

    def _parse_json(val: Any) -> Any:
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return val
        return val

    return {
        "users": [
            {
                "id": r["id"],
                "name": r["name"],
                "email": r["email"],
                "role": normalize_role(r["role"]),
                "profile_details": _parse_json(r["profile_details"]),
                "created_at": r["created_at"],
            }
            for r in users_rows
        ],
        "journal_entries": [
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "journal_text": (
                    f"[Redacted private reflection · {len(str(r['journal_text'] or '').split())} words]"
                    if redact_private_text
                    else r["journal_text"]
                ),
                "timestamp": r["timestamp"],
            }
            for r in journal_rows
        ],
        "recommendations": [
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "report_data": _redact_report_dict(_parse_json(r["report_data"])) if redact_private_text else _parse_json(r["report_data"]),
                "created_at": r["created_at"],
            }
            for r in rec_rows
        ],
        "audit_trails": [
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "username": r["username"],
                "journal_id": r["journal_id"],
                "recommendation": r["recommendation"],
                "timestamp": r["timestamp"],
            }
            for r in audit_rows
        ],
        "user_feedback": [
            {
                "id": r["id"],
                "recommendation_id": r["recommendation_id"],
                "report_data": _parse_json(r["report_data"]),
                "accepted_or_rejected": r["accepted_or_rejected"],
                "timestamp": r["timestamp"],
            }
            for r in feedback_rows
        ],
        "reports": [
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "username": r["username"],
                "report_data": _redact_report_dict(_parse_json(r["report_data"])) if redact_private_text else _parse_json(r["report_data"]),
                "timestamp": r["timestamp"],
            }
            for r in report_rows
        ],
    }


def get_admin_analytics(connection: sqlite3.Connection) -> dict[str, Any]:
    """Calculate genuine SQL Admin Dashboard metrics without artificial floors or synthetic latencies."""
    total_users_row = connection.execute("SELECT COUNT(*) AS c FROM users").fetchone()
    total_users = int(total_users_row["c"]) if total_users_row else 0

    active_users_row = connection.execute("SELECT COUNT(DISTINCT user_id) AS c FROM mood_entries").fetchone()
    active_users = int(active_users_row["c"]) if active_users_row else 0

    logs_row = connection.execute("SELECT COUNT(*) AS c FROM mood_entries").fetchone()
    total_logs = int(logs_row["c"]) if logs_row else 0

    recs_row = connection.execute("SELECT COUNT(*) AS c FROM recommendations").fetchone()
    total_recs = int(recs_row["c"]) if recs_row else 0

    acc_row = connection.execute("SELECT COUNT(*) AS c FROM user_feedback WHERE accepted_or_rejected = 'accepted'").fetchone()
    rej_row = connection.execute("SELECT COUNT(*) AS c FROM user_feedback WHERE accepted_or_rejected = 'rejected'").fetchone()
    acc_count = int(acc_row["c"]) if acc_row else 0
    rej_count = int(rej_row["c"]) if rej_row else 0
    total_fb = acc_count + rej_count

    acceptance_rate = round((acc_count / total_fb) * 100, 2) if total_fb > 0 else None

    return {
        "total_registered_users": total_users,
        "active_users": active_users,
        "total_recommendations_analysis": total_logs + total_recs,
        "recommendation_usage": total_recs,
        "accepted_recommendations": acc_count,
        "rejected_recommendations": rej_count,
        "total_feedback": total_fb,
        "acceptance_rate": acceptance_rate,
        "api_performance": {
            "avg_response_time_ms": None,
            "p95_response_time_ms": None,
            "error_rate_pct": None,
            "uptime_pct": None,
            "status": "No live API telemetry",
        },
        "database_status": {
            "backend": "sqlite",
            "fallback_active": False,
            "warning": None,
        },
    }


