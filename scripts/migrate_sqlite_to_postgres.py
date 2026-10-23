"""Copy legacy MoodMentor SQLite accounts and mood logs into PostgreSQL.

Run first with --dry-run. The migration refuses to write into a non-empty destination.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, inspect, select, text

from wellness_api import database
from wellness_api.models import MoodLog, Recommendation, User


def _timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _json_value(value: Any, fallback: Any) -> Any:
    if not value:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def migrate(source: Path, database_url: str, dry_run: bool = False) -> tuple[int, int]:
    if not source.is_file():
        raise FileNotFoundError(f"SQLite database not found: {source}")
    database.configure_database(database_url)

    source_conn = sqlite3.connect(source)
    source_conn.row_factory = sqlite3.Row
    try:
        users = source_conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        logs = source_conn.execute("SELECT * FROM mood_entries ORDER BY id").fetchall()
    finally:
        source_conn.close()

    existing_tables = set(inspect(database.engine).get_table_names())
    required_tables = {"users", "mood_logs", "recommendations", "auth_sessions"}
    if dry_run and not existing_tables.intersection(required_tables):
        return len(users), len(logs)
    if dry_run and not required_tables.issubset(existing_tables):
        raise RuntimeError(
            "Destination schema is incomplete. Start the API to create its tables first."
        )
    if not dry_run:
        database.create_tables()

    session = database.SessionLocal()
    try:
        destination_users = session.scalar(select(func.count()).select_from(User)) or 0
        destination_logs = session.scalar(select(func.count()).select_from(MoodLog)) or 0
        if destination_users or destination_logs:
            raise RuntimeError(
                "Destination already contains users or mood logs. Migrate only into an empty database."
            )
        if dry_run:
            return len(users), len(logs)

        for row in users:
            session.add(
                User(
                    id=row["id"],
                    name=row["name"],
                    email=row["email"].strip().lower(),
                    password_hash=row["password_hash"],
                    created_at=_timestamp(row["created_at"]),
                )
            )

        session.flush()
        for row in logs:
            columns = set(row.keys())
            raw_tokens = row["tokens"] if "tokens" in columns else ""
            tokens = (
                raw_tokens
                if isinstance(raw_tokens, list)
                else [part.strip() for part in str(raw_tokens or "").split(",") if part.strip()]
            )
            emotion_scores = _json_value(
                row["emotion_scores"] if "emotion_scores" in columns else None, {}
            )
            recommendation_data = _json_value(
                row["recommendation"] if "recommendation" in columns else None, {}
            )
            original = row["original_text"] or ""
            cleaned = row["cleaned_text"] or ""
            entry = MoodLog(
                id=row["id"],
                user_id=row["user_id"],
                source_type=row["source_type"] or "text",
                source_name=(row["source_name"] or "Legacy Import")[:255],
                original_text=original,
                cleaned_text=cleaned,
                processed_text=" ".join(tokens),
                vader_input_text=original,
                tokens=tokens,
                counts={
                    "characters": len(original),
                    "words_before_filtering": len(cleaned.split()),
                    "tokens_after_lemmatization": len(tokens),
                    "lines": len(original.splitlines()),
                },
                sentiment_label=row["sentiment_label"],
                neg_score=float(row["neg_score"]),
                neu_score=float(row["neu_score"]),
                pos_score=float(row["pos_score"]),
                compound_score=float(row["compound_score"]),
                dominant_emotion=row["dominant_emotion"]
                if "dominant_emotion" in columns
                else "neutral",
                emotion_method="legacy_import",
                emotion_scores=emotion_scores,
                created_at=_timestamp(row["created_at"] if "created_at" in columns else None),
            )
            if recommendation_data:
                if isinstance(recommendation_data, str):
                    recommendation_data = {"title": recommendation_data}
                entry.recommendation = Recommendation(
                    title=str(recommendation_data.get("title", "Imported recommendation"))[:180],
                    category=str(recommendation_data.get("category", "Well-Being"))[:100],
                    urgency=str(recommendation_data.get("urgency", "Low"))[:32],
                    summary=str(recommendation_data.get("summary", "")),
                    action_items=list(recommendation_data.get("action_items", [])),
                    is_crisis=bool(recommendation_data.get("is_crisis", False)),
                )
            session.add(entry)

        session.commit()
        if database.engine.dialect.name == "postgresql":
            for table_name in ("users", "mood_logs", "recommendations", "auth_sessions"):
                session.execute(
                    text(
                        "SELECT setval(pg_get_serial_sequence(:table_name, 'id'), "
                        "COALESCE((SELECT MAX(id) FROM " + table_name + "), 1), "
                        "EXISTS (SELECT 1 FROM " + table_name + "))"
                    ),
                    {"table_name": table_name},
                )
            session.commit()
        return len(users), len(logs)
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("wellness.db"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows and validate destination without writing.",
    )
    args = parser.parse_args()
    if not args.database_url:
        parser.error("Set DATABASE_URL or pass --database-url.")
    users, logs = migrate(args.source, args.database_url, args.dry_run)
    action = "Would migrate" if args.dry_run else "Migrated"
    print(f"{action} {users} users and {logs} mood logs.")


if __name__ == "__main__":
    main()
