import sqlite3

import pytest

from wellness.errors import InputValidationError
from wellness.storage import authenticate_user, register_user


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
