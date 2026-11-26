"""Database engine and session lifecycle for the MoodMentor API."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    _project_root = Path(__file__).resolve().parents[1]
    # Neon CLI writes project credentials to .env.local; keep .env as a fallback.
    # Existing process environment variables still take precedence.
    for _env_name in (".env.local", ".env"):
        _env_file = _project_root / _env_name
        if _env_file.is_file():
            load_dotenv(_env_file)
except ImportError:
    pass

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

_db_status: dict[str, Any] = {
    "backend": "unknown",
    "fallback_active": False,
    "warning": None,
}


def _is_test_runtime() -> bool:
    """Return True when running inside automated tests."""
    return bool(os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules or os.getenv("TESTING") == "1")


def is_production_runtime() -> bool:
    """Return True when running in a deployed production environment."""
    if _is_test_runtime():
        return False
    if os.getenv("RENDER"):
        return True
    for key in ("EQUIMIND_ENV", "ENVIRONMENT", "APP_ENV"):
        if os.getenv(key, "").strip().lower() == "production":
            return True
    return False


def get_database_status() -> dict[str, Any]:
    """Return current storage backend and any health/fallback warning."""
    return dict(_db_status)


class Base(DeclarativeBase):
    """Base class for persisted entities."""


def _database_url() -> str:
    if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules or os.getenv("TESTING") == "1":
        db_path = Path(__file__).resolve().parents[1] / "test_moodmentor.db"
        return f"sqlite:///{db_path}"
    url = os.getenv("DATABASE_URL")
    if not url:
        db_path = Path(__file__).resolve().parents[1] / "moodmentor.db"
        return f"sqlite:///{db_path}"
    # Render supplies standard PostgreSQL URLs; this project uses psycopg 3.
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def _make_engine(url: str):
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["connect_args"] = {"connect_timeout": 3}
    return create_engine(url, **kwargs)


engine = _make_engine(_database_url())
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def configure_database(url: str) -> None:
    """Replace the configured engine; used by tests and local SQLite demos."""
    global engine, SessionLocal
    old_engine = engine
    engine = _make_engine(url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    old_engine.dispose()
    _db_status["backend"] = "sqlite" if url.startswith("sqlite") else "postgresql"
    _db_status["fallback_active"] = False
    _db_status["warning"] = None


def create_tables() -> None:
    from . import models  # noqa: F401 - register mapped entities

    logger = logging.getLogger(__name__)
    configured_url = os.getenv("DATABASE_URL")

    try:
        with engine.connect():
            pass
        Base.metadata.create_all(bind=engine)
        _db_status["backend"] = engine.dialect.name
        _db_status["fallback_active"] = False
        _db_status["warning"] = None
    except Exception as exc:
        if configured_url and is_production_runtime():
            _db_status["backend"] = "unavailable"
            _db_status["fallback_active"] = False
            _db_status["warning"] = "Configured production database is unreachable; refusing SQLite fallback."
            logger.error("Production database is unreachable (%s); refusing silent SQLite fallback.", exc)
            raise
        logger.warning("Configured database is unreachable (%s); falling back to local SQLite.", exc)
        db_path = Path(__file__).resolve().parents[1] / "moodmentor.db"
        configure_database(f"sqlite:///{db_path}")
        Base.metadata.create_all(bind=engine)
        _db_status["backend"] = "sqlite"
        _db_status["fallback_active"] = bool(configured_url)
        _db_status["warning"] = (
            "Primary DATABASE_URL unreachable; using local SQLite fallback for development."
            if configured_url
            else None
        )

    tables = inspect(engine).get_table_names()
    if "users" in tables:
        user_cols = {column["name"] for column in inspect(engine).get_columns("users")}
        with engine.begin() as connection:
            if "role" not in user_cols:
                connection.exec_driver_sql(
                    "ALTER TABLE users ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'employee'"
                )
            if "profile_details" not in user_cols:
                default_dict = "'{}'::json" if engine.dialect.name == "postgresql" else "'{}'"
                connection.exec_driver_sql(
                    f"ALTER TABLE users ADD COLUMN profile_details JSON NOT NULL DEFAULT {default_dict}"
                )

    if "recommendations" in tables:
        rec_cols = {column["name"] for column in inspect(engine).get_columns("recommendations")}
        default_list = "'[]'::json" if engine.dialect.name == "postgresql" else "'[]'"
        default_dict = "'{}'::json" if engine.dialect.name == "postgresql" else "'{}'"
        with engine.begin() as connection:
            if "action_paths" not in rec_cols:
                connection.exec_driver_sql(
                    f"ALTER TABLE recommendations ADD COLUMN action_paths JSON NOT NULL DEFAULT {default_list}"
                )
            if "user_id" not in rec_cols:
                connection.exec_driver_sql(
                    "ALTER TABLE recommendations ADD COLUMN user_id INTEGER"
                )
            if "report_data" not in rec_cols:
                connection.exec_driver_sql(
                    f"ALTER TABLE recommendations ADD COLUMN report_data JSON NOT NULL DEFAULT {default_dict}"
                )


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
