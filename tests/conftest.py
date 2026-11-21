"""Pytest configuration and test isolation fixtures."""

import os
import sys
from pathlib import Path

# Mark test environment so all database calls use local SQLite
os.environ["TESTING"] = "1"
os.environ["WELLNESS_FAST_TEST"] = "1"

import pytest
import nltk

# Ensure local workspace NLTK directory is preferred
_local_nltk = Path(__file__).resolve().parents[1] / "nltk_data"
if _local_nltk.is_dir() and str(_local_nltk) not in nltk.data.path:
    nltk.data.path.insert(0, str(_local_nltk))

from wellness_api import database


@pytest.fixture(autouse=True)
def isolated_test_database(tmp_path):
    """Ensure every test uses a fast, isolated SQLite database without touching external resources."""
    test_db_url = f"sqlite:///{tmp_path / 'isolated_test.db'}"
    database.configure_database(test_db_url)
    database.create_tables()
    yield

