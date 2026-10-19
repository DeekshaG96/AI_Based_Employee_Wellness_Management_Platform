from pathlib import Path

import pytest

from wellness.errors import InputValidationError
from wellness.storage import (
    authenticate_user,
    connect_database,
    delete_mood_entry,
    get_user_mood_entries,
    register_user,
    save_mood_entry,
)


def test_connect_database_initializes_tables(tmp_path: Path):
    connection = connect_database(tmp_path / "wellness.db")
    tables = {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    connection.close()
    assert "users" in tables
    assert "mood_entries" in tables


def test_register_and_authenticate_user(tmp_path: Path):
    connection = connect_database(tmp_path / "wellness.db")
    register_user(connection, "Ada Lovelace", "ADA@example.com", "securepass123")
    auth_success = authenticate_user(connection, "ada@example.com", "securepass123")
    assert auth_success is not None
    assert auth_success[1] == "Ada Lovelace"
    assert authenticate_user(connection, "ada@example.com", "wrongpass") is None
    connection.close()


def test_registration_validates_input(tmp_path: Path):
    connection = connect_database(tmp_path / "wellness.db")
    with pytest.raises(InputValidationError, match="6 characters"):
        register_user(connection, "Ada", "ada@example.com", "123")
    connection.close()


def test_save_and_get_mood_entries_with_emotion_telemetry(tmp_path: Path):
    connection = connect_database(tmp_path / "wellness.db")
    register_user(connection, "User One", "user1@example.com", "password123")
    user = authenticate_user(connection, "user1@example.com", "password123")
    assert user is not None
    user_id = user[0]

    entry_id = save_mood_entry(
        connection=connection,
        user_id=user_id,
        source_type="text",
        source_name="direct text",
        original_text="I feel happy and productive.",
        cleaned_text="i feel happy and productive",
        tokens=("feel", "happy", "productive"),
        sentiment_label="positive",
        neg_score=0.0,
        neu_score=0.3,
        pos_score=0.7,
        compound_score=0.65,
        dominant_emotion="joy",
        emotion_scores={"joy": 0.85, "neutral": 0.15},
        recommendation={"title": "Keep up the momentum"},
    )
    assert entry_id > 0

    entries = get_user_mood_entries(connection, user_id)
    assert len(entries) == 1
    assert entries[0]["sentiment_label"] == "positive"
    assert entries[0]["compound_score"] == 0.65
    assert entries[0]["dominant_emotion"] == "joy"
    assert "happy" in entries[0]["tokens"]

    # Test delete
    assert delete_mood_entry(connection, entry_id, user_id) is True
    assert len(get_user_mood_entries(connection, user_id)) == 0
    connection.close()
