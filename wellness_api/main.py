"""MoodMentor FastAPI application."""

from __future__ import annotations

import csv
import logging
import os
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from wellness.comparator import compare_models
from wellness.errors import InputValidationError, WellnessError
from wellness.food_planner import get_mood_food_plan, get_supported_mood_categories
from wellness.meditation import get_meditation_packs, get_soundscape_catalog
from wellness.pipeline import analyze_csv_records, analyze_text, generate_recommendations
from wellness.report import AnalysisReport, SentimentLabel
from wellness.team_analytics import calculate_team_burnout_risk, generate_synthetic_team_data
from wellness.wellness_tools import get_cognitive_reframes, get_guided_exercises
from wellness.yoga import get_all_yoga_remedies, get_yoga_remedy_for_emotion

from .database import SessionLocal, create_tables, get_database_status, get_session
from .models import (
    AuditTrail,
    AuthSession,
    JournalEntry,
    MoodLog,
    Recommendation,
    ReportRecord,
    User,
    UserFeedback,
    recommendation_dict,
    utc_now,
)
from .schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    AuditTrailResponse,
    AuthResponse,
    CompareRequest,
    DatabaseEntitiesResponse,
    FeedbackRequest,
    FeedbackResponse,
    HistoryResponse,
    JournalEntryResponse,
    LoginRequest,
    MessageResponse,
    MoodFoodPlanRequest,
    RecommendationRecordResponse,
    RecommendRequest,
    RegisterRequest,
    ReportRecordResponse,
    TeamBurnoutRequest,
    UserInfoResponse,
    UserResponse,
)
from .security import (
    admin_email_allowlist,
    current_user,
    hash_password,
    issue_session,
    normalize_role,
    require_admin,
    token_digest,
    verify_password,
)

logger = logging.getLogger(__name__)

_telemetry_lock = threading.Lock()
_latency_samples_ms: deque[float] = deque(maxlen=500)
_total_requests = 0
_error_5xx_count = 0


def _bootstrap_admin_accounts() -> None:
    """Apply operator-configured ADMIN_EMAILS and optional ADMIN_BOOTSTRAP_* credentials."""
    allowlist = admin_email_allowlist()
    boot_email = os.getenv("ADMIN_BOOTSTRAP_EMAIL", "").strip().lower()
    boot_pass = os.getenv("ADMIN_BOOTSTRAP_PASSWORD", "")
    boot_name = os.getenv("ADMIN_BOOTSTRAP_NAME", "Platform Administrator").strip() or "Platform Administrator"

    if not allowlist and not (boot_email and len(boot_pass) >= 8):
        return

    with SessionLocal() as db:
        for email in allowlist:
            existing = db.scalar(select(User).where(User.email == email))
            if existing and normalize_role(existing.role) != "admin":
                existing.role = "admin"
        if boot_email and len(boot_pass) >= 8:
            existing_boot = db.scalar(select(User).where(User.email == boot_email))
            if existing_boot is None:
                db.add(
                    User(
                        name=boot_name,
                        email=boot_email,
                        password_hash=hash_password(boot_pass),
                        role="admin",
                    )
                )
            elif normalize_role(existing_boot.role) != "admin":
                existing_boot.role = "admin"
        db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("MOODMENTOR_SKIP_CREATE_TABLES") != "1":
        try:
            create_tables()
            _bootstrap_admin_accounts()
        except Exception as exc:
            logger.error("Database initialization failed during startup: %s", exc)
    # Eagerly warm up NLP transformer and VADER in a background daemon thread (skip during pytest)
    import sys
    if not ("pytest" in sys.modules or os.getenv("TESTING") == "1"):
        from wellness.pipeline import _vader_analyzer, warm_up_emotion_pipeline

        _vader_analyzer()
        threading.Thread(target=warm_up_emotion_pipeline, daemon=True, name="nlp_prewarm").start()
    yield


app = FastAPI(
    title="MoodMentor API",
    description="Authenticated API for emotional tone analysis, recommendations, and history.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_api_telemetry(request: Request, call_next):
    """Measure real API response times and 5xx error counts."""
    global _total_requests, _error_5xx_count
    t0 = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        if request.url.path != "/health":
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            with _telemetry_lock:
                _total_requests += 1
                _latency_samples_ms.append(elapsed_ms)
                if status_code >= 500:
                    _error_5xx_count += 1


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MAX_UPLOAD_BYTES = 2_000_000

portal_dir = Path(__file__).resolve().parent.parent / "moodmentor-portal"
if portal_dir.is_dir():
    app.mount("/portal", StaticFiles(directory=str(portal_dir), html=True), name="portal")


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    return {
        "name": "MoodMentor API",
        "docs": "/docs",
        "health": "/health",
        "portal": "/portal",
        "streamlit": "http://localhost:8501",
    }


@app.get("/health", tags=["service"])
def health(db: Session = Depends(get_session)) -> dict[str, Any]:
    db_stat = get_database_status()
    try:
        db.execute(select(func.count()).select_from(User)).scalar_one()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable.") from exc
    payload: dict[str, Any] = {"status": "ok", "database": "connected"}
    if db_stat.get("fallback_active") and db_stat.get("warning"):
        payload["fallback_active"] = True
        payload["warning"] = db_stat["warning"]
    return payload


@app.get("/health/database", tags=["service"])
def database_health(db: Session = Depends(get_session)) -> dict[str, Any]:
    """Inspect active database backend and fallback health status."""
    try:
        db.execute(select(func.count()).select_from(User)).scalar_one()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable.") from exc
    return get_database_status()


def _auth_response(db: Session, user: User) -> AuthResponse:
    return AuthResponse(
        access_token=issue_session(db, user),
        user=UserResponse(
            id=user.id,
            name=user.name,
            email=user.email,
            role=normalize_role(getattr(user, "role", None)),
        ),
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
    assigned_role = "admin" if email in admin_email_allowlist() else "employee"
    user = User(
        name=payload.name,
        email=email,
        password_hash=hash_password(payload.password),
        role=assigned_role,
    )
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
    if email in admin_email_allowlist() and normalize_role(user.role) != "admin":
        user.role = "admin"
        db.commit()
        db.refresh(user)
    return _auth_response(db, user)


@app.get("/auth/me", response_model=UserResponse, tags=["auth"])
def me(user: User = Depends(current_user)) -> UserResponse:
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        role=normalize_role(getattr(user, "role", None)),
    )


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
        user_id=user.id,
        title=str(rec.get("title", "Well-Being Recommendation"))[:180],
        category=str(rec.get("category", "Well-Being"))[:100],
        urgency=str(rec.get("urgency", "Low"))[:32],
        summary=str(rec.get("summary", "")),
        action_items=[str(item) for item in rec.get("action_items", [])],
        action_paths=[dict(path) for path in rec.get("action_paths", [])],
        is_crisis=bool(rec.get("is_crisis", False)),
        report_data=report.to_dict(),
    )
    db.add(log)
    db.flush()

    # 1. Journal entries (Journal text)
    journal_entry = JournalEntry(
        user_id=user.id,
        journal_text=report.original_text,
        timestamp=utc_now(),
    )
    db.add(journal_entry)
    db.flush()

    # 4. Audit trails (User id, username, journal id, recommendation, timestamp)
    audit_trail = AuditTrail(
        user_id=user.id,
        username=user.name,
        journal_id=journal_entry.id,
        recommendation=str(rec.get("title", "Well-Being Recommendation")),
        timestamp=utc_now(),
    )
    db.add(audit_trail)

    # 6. Reports (Report id, user id, username, timestamp)
    report_record = ReportRecord(
        user_id=user.id,
        username=user.name,
        timestamp=utc_now(),
        report_data=report.to_dict(),
    )
    db.add(report_record)

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


@app.post("/analyze/quick", tags=["analysis"])
def analyze_quick(
    payload: AnalyzeRequest,
    db: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """Analyze single reflection text for the authenticated account and sync to database."""
    try:
        report = analyze_text(payload.text, payload.source_name or "web_portal")
        saved = _persist_report(
            db,
            user,
            report,
            "web_portal",
            payload.source_name or "web_portal",
        )
        res = report.to_dict()
        res["id"] = saved.id
        return res
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WellnessError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/analyze/compare", tags=["evaluation"])
def compare_models_endpoint(
    payload: CompareRequest,
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """Compare lexical baseline against fine-tuned DistilBERT transformer for an authenticated user."""
    del user
    return compare_models(payload.text)


@app.get("/wellness/exercises", tags=["wellness"])
def get_exercises_endpoint() -> list[dict[str, Any]]:
    """Return catalog of evidence-based somatic and cognitive micro-interventions."""
    return get_guided_exercises()


@app.get("/wellness/reframes/{emotion}", tags=["wellness"])
def get_reframe_endpoint(emotion: str) -> dict[str, Any]:
    """Return tailored CBT cognitive reframing prompt for a detected emotion."""
    return get_cognitive_reframes(emotion)


@app.post("/team/burnout-risk", tags=["wellness"])
def team_burnout_risk_endpoint(
    payload: TeamBurnoutRequest,
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """Calculate aggregated Team Burnout Risk Score and Resilience Index (Admin only)."""
    del user
    entries = payload.entries
    if not entries:
        entries = generate_synthetic_team_data(days=payload.days, team_size=payload.team_size, scenario=payload.scenario)
    return calculate_team_burnout_risk(entries)


@app.get("/wellness/food-plan/{emotion}", tags=["nutrition"])
def get_food_plan_endpoint(emotion: str, dietary_pref: str = Query(default="any")) -> dict[str, Any]:
    """Return tailored nutritional psychiatry recommendations for the detected emotion."""
    return get_mood_food_plan(emotion, dietary_pref)


@app.post("/wellness/food-plan", tags=["nutrition"])
def post_food_plan_endpoint(payload: MoodFoodPlanRequest) -> dict[str, Any]:
    """Return tailored nutritional psychiatry recommendations for the requested emotion and diet."""
    return get_mood_food_plan(payload.emotion, payload.dietary_preference)


@app.get("/wellness/food-plan-categories", tags=["nutrition"])
def get_food_plan_categories_endpoint() -> list[dict[str, str]]:
    """Return supported emotional states for nutritional planning."""
    return get_supported_mood_categories()


@app.get("/wellness/meditations", tags=["meditation"])
def get_meditations_endpoint() -> list[dict[str, Any]]:
    """Return catalog of structured, evidence-based Medito-inspired mindfulness packs."""
    return get_meditation_packs()


@app.get("/wellness/soundscapes", tags=["meditation"])
def get_soundscapes_endpoint() -> list[dict[str, str]]:
    """Return catalog of client-side ambient soundscapes and meditation timer presets."""
    return get_soundscape_catalog()


@app.get("/wellness/yoga-remedies", tags=["yoga"])
def get_yoga_remedies_endpoint() -> list[dict[str, Any]]:
    """Return catalog of evidence-based therapeutic yoga remedies and desk modifications."""
    return get_all_yoga_remedies()


@app.get("/wellness/yoga-remedies/{emotion}", tags=["yoga"])
def get_yoga_remedy_endpoint(emotion: str) -> dict[str, Any]:
    """Retrieve the targeted yoga remedy for a specific emotional or strain state."""
    return get_yoga_remedy_for_emotion(emotion)


@app.post("/recommend", tags=["recommendations"])
def recommend_endpoint(
    payload: RecommendRequest,
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """Generate tailored well-being recommendations and micro-interventions for any emotion."""
    del user
    lbl_str = payload.sentiment_label.lower()
    lbl = SentimentLabel(lbl_str) if lbl_str in ("positive", "neutral", "negative") else SentimentLabel.NEUTRAL
    return generate_recommendations(payload.emotion.lower(), lbl, payload.compound, payload.text)


@app.get("/recommend/{emotion}", tags=["recommendations"])
def get_recommend_by_emotion_endpoint(
    emotion: str,
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """Return curated recommendation strategies for a specified emotion."""
    del user
    return generate_recommendations(emotion.lower(), SentimentLabel.NEUTRAL, 0.0, "")


def _measured_api_performance() -> dict[str, Any]:
    with _telemetry_lock:
        samples = list(_latency_samples_ms)
        total = _total_requests
        errors = _error_5xx_count
    if not samples or total <= 0:
        return {
            "avg_response_time_ms": None,
            "p95_response_time_ms": None,
            "error_rate_pct": None,
            "uptime_pct": None,
            "status": "No data yet",
            "sample_count": 0,
        }
    samples.sort()
    avg_ms = round(sum(samples) / len(samples), 2)
    p95_idx = min(len(samples) - 1, int(0.95 * (len(samples) - 1)))
    p95_ms = round(samples[p95_idx], 2)
    err_pct = round((errors / total) * 100.0, 2)
    avail_pct = round(max(0.0, ((total - errors) / total) * 100.0), 2)
    return {
        "avg_response_time_ms": avg_ms,
        "p95_response_time_ms": p95_ms,
        "error_rate_pct": err_pct,
        "uptime_pct": avail_pct,
        "status": "Optimal" if err_pct < 5.0 else "Degraded",
        "sample_count": total,
    }


def _redact_report_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Strip raw private reflection text fields while retaining aggregated telemetry attributes."""
    if not isinstance(payload, dict):
        return {}
    cleaned = dict(payload)
    for secret_key in ("original_text", "cleaned_text", "processed_text", "vader_input_text"):
        if secret_key in cleaned:
            cleaned[secret_key] = "[Redacted]"
    return cleaned


@app.get("/admin/analytics", tags=["admin"])
def admin_analytics_endpoint(
    db: Session = Depends(get_session),
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """Retrieve server-enforced Admin Dashboard Analytics from live SQL counts and request telemetry."""
    del user
    total_users = db.scalar(select(func.count()).select_from(User)) or 0
    total_logs = db.scalar(select(func.count()).select_from(MoodLog)) or 0
    total_recs = db.scalar(select(func.count()).select_from(Recommendation)) or 0
    active_users = db.scalar(select(func.count(func.distinct(MoodLog.user_id)))) or 0

    accepted = db.scalar(
        select(func.count()).select_from(UserFeedback).where(UserFeedback.accepted_or_rejected == "accepted")
    ) or 0
    rejected = db.scalar(
        select(func.count()).select_from(UserFeedback).where(UserFeedback.accepted_or_rejected == "rejected")
    ) or 0
    total_fb = accepted + rejected
    acceptance_rate = round((accepted / total_fb) * 100, 2) if total_fb > 0 else None

    return {
        "total_registered_users": int(total_users),
        "active_users": int(active_users),
        "total_recommendations_analysis": int(total_logs + total_recs),
        "recommendation_usage": int(total_recs),
        "accepted_recommendations": int(accepted),
        "rejected_recommendations": int(rejected),
        "total_feedback": int(total_fb),
        "acceptance_rate": acceptance_rate,
        "api_performance": _measured_api_performance(),
        "database_status": get_database_status(),
    }


@app.get("/admin/overview", tags=["admin"])
def admin_privacy_filtered_overview(
    db: Session = Depends(get_session),
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """Return privacy-filtered aggregated organizational telemetry without raw reflection text or PII."""
    del user
    users = db.scalars(select(User)).all()
    logs = db.scalars(select(MoodLog).order_by(MoodLog.created_at.desc())).all()
    feedback_rows = db.scalars(select(UserFeedback).order_by(UserFeedback.id.desc())).all()

    user_dept: dict[int, str] = {}
    for u in users:
        prof = u.profile_details or {}
        if isinstance(prof, dict) and prof.get("department"):
            user_dept[u.id] = str(prof["department"])

    records = []
    for log in logs:
        src = log.source_name or ""
        dept = user_dept.get(log.user_id, "General")
        if "[" in src and "]" in src:
            parsed = src.split("[", 1)[1].split("]", 1)[0].strip()
            if parsed:
                dept = parsed
        records.append({
            "department": dept,
            "dominant_emotion": log.dominant_emotion,
            "sentiment_label": log.sentiment_label,
            "compound_score": float(log.compound_score),
            "created_at": log.created_at.isoformat() if log.created_at else "",
        })

    feedback_summary = []
    for fb in feedback_rows[:50]:
        rd = fb.report_data if isinstance(fb.report_data, dict) else {}
        feedback_summary.append({
            "decision": fb.accepted_or_rejected,
            "department": str(rd.get("department", "General")),
            "practice": str(rd.get("title") or rd.get("practice") or "Personalized Care Plan"),
            "decline_reason": rd.get("decline_reason", "") if fb.accepted_or_rejected == "rejected" else "",
            "timestamp": fb.timestamp.isoformat() if fb.timestamp else "",
        })

    return {
        "total_users": len(users),
        "records": records,
        "feedback_summary": feedback_summary,
        "database_status": get_database_status(),
    }


@app.post("/feedback", response_model=FeedbackResponse, tags=["database"])
def submit_feedback(
    payload: FeedbackRequest,
    db: Session = Depends(get_session),
    user: User = Depends(current_user),
) -> FeedbackResponse:
    """Store user feedback for recommendations (accepted/rejected status)."""
    feedback_record = UserFeedback(
        recommendation_id=payload.recommendation_id,
        user_id=user.id,
        report_data=payload.report_data,
        accepted_or_rejected=payload.accepted_or_rejected.lower(),
        timestamp=utc_now(),
    )
    db.add(feedback_record)
    db.commit()
    db.refresh(feedback_record)
    return FeedbackResponse(
        id=feedback_record.id,
        recommendation_id=feedback_record.recommendation_id,
        user_id=feedback_record.user_id,
        report_data=feedback_record.report_data,
        accepted_or_rejected=feedback_record.accepted_or_rejected,
        timestamp=feedback_record.timestamp,
    )


@app.get("/database/entities", response_model=DatabaseEntitiesResponse, tags=["admin"])
def get_all_database_entities(
    db: Session = Depends(get_session),
    user: User = Depends(require_admin),
) -> DatabaseEntitiesResponse:
    """Retrieve privacy-filtered entity metadata for verified administrators (raw reflection text redacted)."""
    del user
    users_rows = db.scalars(select(User).order_by(User.id.asc())).all()
    journals_rows = db.scalars(select(JournalEntry).order_by(JournalEntry.id.desc())).all()
    rec_rows = db.scalars(select(Recommendation).order_by(Recommendation.id.desc())).all()
    audit_rows = db.scalars(select(AuditTrail).order_by(AuditTrail.id.desc())).all()
    feedback_rows = db.scalars(select(UserFeedback).order_by(UserFeedback.id.desc())).all()
    reports_rows = db.scalars(select(ReportRecord).order_by(ReportRecord.id.desc())).all()

    return DatabaseEntitiesResponse(
        users=[
            UserInfoResponse(
                id=u.id,
                name=u.name,
                email=u.email,
                role=normalize_role(getattr(u, "role", None)),
                profile_details=u.profile_details or {},
                created_at=u.created_at,
            )
            for u in users_rows
        ],
        journal_entries=[
            JournalEntryResponse(
                id=j.id,
                user_id=j.user_id,
                journal_text=f"[Redacted private reflection · {len((j.journal_text or '').split())} words]",
                timestamp=j.timestamp,
            )
            for j in journals_rows
        ],
        recommendations=[
            RecommendationRecordResponse(
                id=r.id,
                user_id=r.user_id,
                title=r.title,
                category=r.category,
                urgency=r.urgency,
                summary=r.summary,
                report_data=_redact_report_payload(r.report_data),
                created_at=r.created_at,
            )
            for r in rec_rows
        ],
        audit_trails=[
            AuditTrailResponse(
                id=a.id,
                user_id=a.user_id,
                username=a.username,
                journal_id=a.journal_id,
                recommendation=a.recommendation,
                timestamp=a.timestamp,
            )
            for a in audit_rows
        ],
        user_feedback=[
            FeedbackResponse(
                id=f.id,
                recommendation_id=f.recommendation_id,
                user_id=f.user_id,
                report_data=f.report_data or {},
                accepted_or_rejected=f.accepted_or_rejected,
                timestamp=f.timestamp,
            )
            for f in feedback_rows
        ],
        reports=[
            ReportRecordResponse(
                id=rep.id,
                user_id=rep.user_id,
                username=rep.username,
                timestamp=rep.timestamp,
                report_data=_redact_report_payload(rep.report_data),
            )
            for rep in reports_rows
        ],
    )





