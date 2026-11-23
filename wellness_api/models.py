"""SQLAlchemy models for accounts, analysis history, recommendations, audit trails, and feedback."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    profile_details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    mood_logs: Mapped[list[MoodLog]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    sessions: Mapped[list[AuthSession]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    journal_entries: Mapped[list[JournalEntry]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    audit_trails: Mapped[list[AuditTrail]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    reports: Mapped[list[ReportRecord]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    user: Mapped[User] = relationship(back_populates="sessions")


class MoodLog(Base):
    __tablename__ = "mood_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="text")
    source_name: Mapped[str] = mapped_column(String(255), nullable=False, default="Direct Entry")
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    processed_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    vader_input_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tokens: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    counts: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False, default=dict)
    sentiment_label: Mapped[str] = mapped_column(String(16), nullable=False)
    neg_score: Mapped[float] = mapped_column(Float, nullable=False)
    neu_score: Mapped[float] = mapped_column(Float, nullable=False)
    pos_score: Mapped[float] = mapped_column(Float, nullable=False)
    compound_score: Mapped[float] = mapped_column(Float, nullable=False)
    dominant_emotion: Mapped[str] = mapped_column(String(32), nullable=False, default="neutral")
    emotion_method: Mapped[str] = mapped_column(
        String(32), nullable=False, default="legacy_unknown"
    )
    emotion_scores: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )
    user: Mapped[User] = relationship(back_populates="mood_logs")
    recommendation: Mapped[Recommendation | None] = relationship(
        back_populates="mood_log", cascade="all, delete-orphan", uselist=False
    )


class JournalEntry(Base):
    """Journal entries entity storing original reflection text."""
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    journal_text: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )
    user: Mapped[User] = relationship(back_populates="journal_entries")


class Recommendation(Base):
    """Personalized recommendations linked to users and telemetry reports."""
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True
    )
    mood_log_id: Mapped[int | None] = mapped_column(
        ForeignKey("mood_logs.id", ondelete="CASCADE"), unique=True, index=True, nullable=True
    )
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, default="Well-Being")
    urgency: Mapped[str] = mapped_column(String(32), nullable=False, default="Low")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    action_items: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    action_paths: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    is_crisis: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    report_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    mood_log: Mapped[MoodLog | None] = relationship(back_populates="recommendation")
    user: Mapped[User | None] = relationship()


class AuditTrail(Base):
    """Audit trails logging interactions with user ID, username, journal ID, and recommendation."""
    __tablename__ = "audit_trails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    username: Mapped[str] = mapped_column(String(120), nullable=False)
    journal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )
    user: Mapped[User | None] = relationship(back_populates="audit_trails")


class UserFeedback(Base):
    """User feedback on recommendations with accepted/rejected status."""
    __tablename__ = "user_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recommendation_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    user_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    report_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    accepted_or_rejected: Mapped[str] = mapped_column(String(32), nullable=False, default="accepted")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )


class ReportRecord(Base):
    """Formal generated wellness reports with timestamp, username, and user ID."""
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    username: Mapped[str] = mapped_column(String(120), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )
    report_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    user: Mapped[User | None] = relationship(back_populates="reports")


def recommendation_dict(recommendation: Recommendation | None) -> dict[str, Any]:
    if recommendation is None:
        return {}
    return {
        "id": recommendation.id,
        "title": recommendation.title,
        "category": recommendation.category,
        "urgency": recommendation.urgency,
        "summary": recommendation.summary,
        "action_items": recommendation.action_items or [],
        "action_paths": recommendation.action_paths or [],
        "is_crisis": recommendation.is_crisis,
        "report_data": recommendation.report_data or {},
    }
