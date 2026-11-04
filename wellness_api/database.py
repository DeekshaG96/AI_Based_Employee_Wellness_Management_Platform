"""Database engine and session lifecycle for the MoodMentor API."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

try:
    from dotenv import load_dotenv

    _env_file = Path(__file__).resolve().parents[1] / ".env"
    if _env_file.is_file():
        load_dotenv(_env_file)
except ImportError:
    pass

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class for persisted entities."""


def _database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        db_path = Path(__file__).resolve().parents[1] / "moodmentor.db"
        return f"sqlite:///{db_path.as_posix()}"
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


def create_tables() -> None:
    from . import models  # noqa: F401 - register mapped entities

    Base.metadata.create_all(bind=engine)
    if "recommendations" not in inspect(engine).get_table_names():
        return
    columns = {column["name"] for column in inspect(engine).get_columns("recommendations")}
    if "action_paths" in columns:
        return
    default = "'[]'::json" if engine.dialect.name == "postgresql" else "'[]'"
    with engine.begin() as connection:
        connection.exec_driver_sql(
            f"ALTER TABLE recommendations ADD COLUMN action_paths JSON NOT NULL DEFAULT {default}"
        )


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
