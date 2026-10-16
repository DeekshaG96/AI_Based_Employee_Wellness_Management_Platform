import sqlite3

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


def test_connect_database_initializes_tables(tmp_path):
    connection = connect_database(tmp_path / "wellness.db")
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    connection.close()
    assert "users" in tables
    assert "mood_entries" in tables


def test_register_and_authenticate_user():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, "
        "password_hash TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    register_user(connection, "Ada Lovelace", "ADA@example.com", "securepass")
    assert authenticate_user(connection, "ada@example.com", "securepass") == (1, "Ada Lovelace")
    assert authenticate_user(connection, "ada@example.com", "wrongpass") is None


def test_registration_validates_input():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, "
        "password_hash TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    with pytest.raises(InputValidationError, match="8 characters"):
        register_user(connection, "Ada", "ada@example.com", "short")


def test_save_and_get_mood_entries(tmp_path):
    connection = connect_database(tmp_path / "wellness.db")
    register_user(connection, "User One", "user1@example.com", "password123")
    user = authenticate_user(connection, "user1@example.com", "password123")
    assert user is not None
    user_id = user[0]

    entry_id = save_mood_entry(
        connection,
        user_id,
        "text",
        "direct text",
        "I feel happy and productive.",
        "i feel happy and productive",
        ("feel", "happy", "productive"),
        "positive",
        0.0,
        0.3,
        0.7,
        0.65,
    )
    assert entry_id > 0

    entries = get_user_mood_entries(connection, user_id)
    assert len(entries) == 1
    assert entries[0]["sentiment_label"] == "positive"
    assert entries[0]["compound_score"] == 0.65
    assert "happy" in entries[0]["tokens"]

    # Test delete
    assert delete_mood_entry(connection, entry_id, user_id) is True
    assert len(get_user_mood_entries(connection, user_id)) == 0
    connection.close()

