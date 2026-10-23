"""MoodMentor FastAPI application."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from wellness.errors import InputValidationError, WellnessError
from wellness.pipeline import analyze_csv_records, analyze_text
from wellness.report import AnalysisReport

from .database import create_tables, get_session
from .models import AuthSession, MoodLog, Recommendation, User, recommendation_dict
from .schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    AuthResponse,
    HistoryResponse,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    UserResponse,
)
from .security import current_user, hash_password, issue_session, token_digest, verify_password

app = FastAPI(
    title="MoodMentor API",
    description="Authenticated API for emotional tone analysis, recommendations, and history.",
    version="1.0.0",
)
MAX_UPLOAD_BYTES = 2_000_000


@app.on_event("startup")
def initialize_database() -> None:
    if os.getenv("MOODMENTOR_SKIP_CREATE_TABLES") != "1":
        create_tables()


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    return {"name": "MoodMentor API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["service"])
def health(db: Session = Depends(get_session)) -> dict[str, str]:
    try:
        db.execute(select(func.count()).select_from(User)).scalar_one()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable.") from exc
    return {"status": "ok", "database": "connected"}


def _auth_response(db: Session, user: User) -> AuthResponse:
    return AuthResponse(
        access_token=issue_session(db, user),
        user=UserResponse(id=user.id, name=user.name, email=user.email),
    )


@app.post(
    "/auth/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
)
def register(payload: RegisterRequest, db: Session = Depends(get_session)) -> AuthResponse:
    email = str(payload.email).strip().lower()
    existing = db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(status_code=409, detail="An account with that email already exists.")
    user = User(name=payload.name, email=email, password_hash=hash_password(payload.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="An account with that email already exists."
        ) from exc
    db.refresh(user)
    return _auth_response(db, user)


@app.post("/auth/login", response_model=AuthResponse, tags=["auth"])
def login(payload: LoginRequest, db: Session = Depends(get_session)) -> AuthResponse:
    email = str(payload.email).strip().lower()
    user = db.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return _auth_response(db, user)


@app.get("/auth/me", response_model=UserResponse, tags=["auth"])
def me(user: User = Depends(current_user)) -> UserResponse:
    return UserResponse(id=user.id, name=user.name, email=user.email)


@app.post("/auth/logout", response_model=MessageResponse, tags=["auth"])
def logout(
    request: Request,
    db: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> MessageResponse:
    del user  # the dependency validates that the supplied bearer token belongs to an active user
    header = request.headers.get("authorization", "")
    token = header.partition(" ")[2].strip()
    session = db.scalar(select(AuthSession).where(AuthSession.token_hash == token_digest(token)))
    if session is not None:
        db.delete(session)
        db.commit()
    return MessageResponse(message="Signed out.")


def _persist_report(
    db: Session,
    user: User,
    report: AnalysisReport,
    source_type: str,
    source_name: str,
    commit: bool = True,
) -> MoodLog:
    scores = report.sentiment_scores
    log = MoodLog(
        user_id=user.id,
        source_type=source_type,
        source_name=source_name[:255],
        original_text=report.original_text,
        cleaned_text=report.cleaned_text,
        processed_text=report.processed_text,
        vader_input_text=report.vader_input_text,
        tokens=list(report.tokens),
        counts=report.counts,
        sentiment_label=report.sentiment_label.value,
        neg_score=scores.negative,
        neu_score=scores.neutral,
        pos_score=scores.positive,
        compound_score=scores.compound,
        dominant_emotion=report.dominant_emotion,
        emotion_method=report.emotion_method,
        emotion_scores=report.emotion_scores,
    )
    rec = report.recommendations
    log.recommendation = Recommendation(
        title=str(rec.get("title", "Well-Being Recommendation"))[:180],
        category=str(rec.get("category", "Well-Being"))[:100],
        urgency=str(rec.get("urgency", "Low"))[:32],
        summary=str(rec.get("summary", "")),
        action_items=[str(item) for item in rec.get("action_items", [])],
        action_paths=[dict(path) for path in rec.get("action_paths", [])],
        is_crisis=bool(rec.get("is_crisis", False)),
    )
    db.add(log)
    if commit:
        db.commit()
        db.refresh(log)
    else:
        db.flush()
    return log


def _analysis_response(log: MoodLog) -> dict[str, Any]:
    return {
        "id": log.id,
        "created_at": log.created_at,
        "input_summary": {
            "source_type": log.source_type,
            "source_name": log.source_name,
            "character_count": len(log.original_text),
            "line_count": len(log.original_text.splitlines()),
            "token_count": len(log.tokens or []),
        },
        "original_text": log.original_text,
        "cleaned_text": log.cleaned_text,
        "processed_text": log.processed_text,
        "tokens": log.tokens or [],
        "sentiment_scores": {
            "negative": log.neg_score,
            "neutral": log.neu_score,
            "positive": log.pos_score,
            "compound": log.compound_score,
        },
        "sentiment_label": log.sentiment_label,
        "counts": log.counts or {},
        "vader_input_text": log.vader_input_text,
        "dominant_emotion": log.dominant_emotion,
        "emotion_method": log.emotion_method,
        "emotion_scores": log.emotion_scores or {},
        "recommendations": recommendation_dict(log.recommendation),
    }


def _analyze_and_persist(
    text: str, source_name: str, source_type: str, db: Session, user: User
) -> dict[str, Any]:
    try:
        report = analyze_text(text, source_name)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WellnessError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    saved = _persist_report(db, user, report, source_type, source_name)
    return _analysis_response(saved)


@app.post("/analyze", response_model=AnalysisResponse, tags=["analysis"])
def analyze(
    payload: AnalyzeRequest,
    db: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    return _analyze_and_persist(payload.text, payload.source_name, "text", db, user)


@app.post("/analyze/file", tags=["analysis"])
def analyze_file(
    file: UploadFile = File(...),
    text_column: str | None = Query(default=None, max_length=100),
    db: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise HTTPException(status_code=415, detail="Upload a .csv or .txt file.")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload must be 2 MB or smaller.")
    if not content.strip():
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    reports: list[AnalysisReport]
    try:
        if suffix == ".txt":
            reports = [analyze_text(content.decode("utf-8-sig", errors="replace"), filename)]
        else:
            with tempfile.TemporaryDirectory() as temporary_directory:
                temporary_path = Path(temporary_directory) / "upload.csv"
                temporary_path.write_bytes(content)
                reports = analyze_csv_records(temporary_path, text_column)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (UnicodeError, csv.Error) as exc:
        raise HTTPException(
            status_code=422, detail="The uploaded file could not be parsed."
        ) from exc
    except WellnessError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not reports:
        raise HTTPException(status_code=422, detail="No non-empty feedback rows were found.")
    try:
        entries = [
            _persist_report(db, user, report, suffix[1:], filename, commit=False)
            for report in reports
        ]
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"items": [_analysis_response(entry) for entry in entries], "total": len(entries)}


def _history_entry(log: MoodLog) -> dict[str, Any]:
    return {
        "id": log.id,
        "user_id": log.user_id,
        "source_type": log.source_type,
        "source_name": log.source_name,
        "original_text": log.original_text,
        "cleaned_text": log.cleaned_text,
        "processed_text": log.processed_text,
        "vader_input_text": log.vader_input_text,
        "tokens": log.tokens or [],
        "counts": log.counts or {},
        "sentiment_label": log.sentiment_label,
        "neg_score": log.neg_score,
        "neu_score": log.neu_score,
        "pos_score": log.pos_score,
        "compound_score": log.compound_score,
        "dominant_emotion": log.dominant_emotion,
        "emotion_method": log.emotion_method,
        "emotion_scores": log.emotion_scores or {},
        "created_at": log.created_at,
        "recommendation": recommendation_dict(log.recommendation) or None,
    }


@app.get("/history/{user_id}", response_model=HistoryResponse, tags=["history"])
def history(
    user_id: int,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    if user_id != user.id:
        raise HTTPException(status_code=403, detail="You can only access your own history.")
    total = (
        db.scalar(select(func.count()).select_from(MoodLog).where(MoodLog.user_id == user.id)) or 0
    )
    rows = db.scalars(
        select(MoodLog)
        .options(selectinload(MoodLog.recommendation))
        .where(MoodLog.user_id == user.id)
        .order_by(MoodLog.created_at.desc(), MoodLog.id.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return {
        "items": [_history_entry(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.delete("/history/{entry_id}", response_model=MessageResponse, tags=["history"])
def delete_history(
    entry_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> MessageResponse:
    entry = db.scalar(select(MoodLog).where(MoodLog.id == entry_id, MoodLog.user_id == user.id))
    if entry is None:
        raise HTTPException(status_code=404, detail="Record not found.")
    db.delete(entry)
    db.commit()
    return MessageResponse(message="Record deleted.")
