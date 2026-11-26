"""Small authenticated HTTP client used by the Streamlit frontend with resilient in-process SQLite fallback."""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Any

import requests

from .report import AnalysisReport, InputSummary, SentimentLabel, SentimentScores

logger = logging.getLogger(__name__)

_api_available: bool | None = None
_last_check_time: float = 0.0
_LOCAL_SESSIONS: dict[str, tuple[int, str]] = {}


class ApiError(RuntimeError):
    """An API request failed with a user-readable explanation."""


def api_url() -> str:
    try:
        import streamlit as st

        configured = st.secrets.get("API_URL")
    except Exception:
        configured = None
    return str(configured or os.getenv("API_URL", "http://127.0.0.1:8000")).rstrip("/")


def is_api_available() -> bool:
    """Check if the external FastAPI server is responsive, caching the status for 30s."""
    global _api_available, _last_check_time
    now = time.time()
    if _api_available is not None and (now - _last_check_time < 30.0):
        return _api_available
    try:
        resp = requests.get(f"{api_url()}/health", timeout=(0.4, 0.4))
        _api_available = resp.ok
    except Exception:
        _api_available = False
    _last_check_time = now
    return _api_available


def _get_current_user_id() -> int:
    try:
        import streamlit as st

        return int(st.session_state.get("user_id") or 1)
    except Exception:
        return 1


def _request(
    method: str,
    path: str,
    token: str | None = None,
    **kwargs: Any,
) -> Any:
    if not is_api_available():
        raise ApiError(f"API server at {api_url()} is inactive; fallback engaged.")

    headers = dict(kwargs.pop("headers", {}))
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.request(
            method,
            f"{api_url()}{path}",
            headers=headers,
            timeout=(1.0, 10.0),
            **kwargs,
        )
    except requests.RequestException as exc:
        global _api_available
        _api_available = False
        raise ApiError(
            f"MoodMentor API is not reachable at {api_url()}. Start the API and database, then retry."
        ) from exc
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not response.ok:
        detail = body.get("detail", f"API request failed ({response.status_code}).")
        if isinstance(detail, list):
            detail = "; ".join(item.get("msg", str(item)) for item in detail)
        raise ApiError(str(detail))
    return body


def _require_local_admin(token: str | None) -> int:
    """Verify administrator role for local SQLite fallback sessions."""
    from .storage import connect_database, normalize_role

    if not token or token not in _LOCAL_SESSIONS:
        raise ApiError("Administrator sign-in is required.")
    uid, cached_role = _LOCAL_SESSIONS[token]
    conn = connect_database()
    try:
        row = conn.execute("SELECT role FROM users WHERE id = ?", (uid,)).fetchone()
        db_role = normalize_role(row["role"]) if row else normalize_role(cached_role)
    finally:
        conn.close()
    if db_role != "admin":
        raise ApiError("Administrator privileges are required.")
    return uid


def _record_local_telemetry(
    method: str, endpoint: str, start_time: float, status_code: int = 200
) -> None:
    """Persist local execution latency to sqlite api_logs."""
    from .storage import connect_database, save_api_log

    latency_ms = max(0.1, (time.perf_counter() - start_time) * 1000.0)
    try:
        conn = connect_database()
        try:
            save_api_log(conn, method, endpoint, status_code, latency_ms)
        finally:
            conn.close()
    except Exception:
        pass


def _record_local_audit(
    user_id: int,
    username: str,
    action: str,
    details: str,
    journal_id: int | None = None,
) -> None:
    """Persist local structured security event to sqlite audit_trails."""
    from .storage import connect_database, save_audit_trail

    try:
        conn = connect_database()
        try:
            save_audit_trail(conn, user_id, username, journal_id, details, action=action)
        finally:
            conn.close()
    except Exception:
        pass


def register(name: str, email: str, password: str) -> dict[str, Any]:
    if is_api_available():
        return _request(
            "POST", "/auth/register", json={"name": name, "email": email, "password": password}
        )

    from .storage import admin_email_allowlist, connect_database, register_user

    t0 = time.perf_counter()
    conn = connect_database()
    try:
        try:
            uid = register_user(conn, name, email, password)
        except Exception as exc:
            raise ApiError(str(exc)) from exc
        clean_email = email.strip().lower()
        role = "admin" if clean_email in admin_email_allowlist() else "employee"
        tok = secrets.token_urlsafe(24)
        _LOCAL_SESSIONS[tok] = (uid, role)
        _record_local_telemetry("POST", "/auth/register", t0, 200)
        _record_local_audit(uid, name.strip(), "USER_REGISTER", f"Registered account ({clean_email})")
        return {
            "access_token": tok,
            "token_type": "bearer",
            "user": {"id": uid, "name": name.strip(), "email": clean_email, "role": role},
        }
    finally:
        conn.close()


def login(email: str, password: str) -> dict[str, Any]:
    if is_api_available():
        return _request("POST", "/auth/login", json={"email": email, "password": password})

    from .storage import (
        authenticate_user_with_role,
        bootstrap_admin_from_env,
        connect_database,
    )

    t0 = time.perf_counter()
    conn = connect_database()
    try:
        bootstrap_admin_from_env(conn)
        auth = authenticate_user_with_role(conn, email, password)
        if auth is None:
            raise ApiError("Invalid email or password.")
        uid, uname, urole = auth
        tok = secrets.token_urlsafe(24)
        _LOCAL_SESSIONS[tok] = (uid, urole)
        _record_local_telemetry("POST", "/auth/login", t0, 200)
        _record_local_audit(
            uid,
            uname,
            "ADMIN_LOGIN" if urole == "admin" else "USER_LOGIN",
            f"Authenticated ({urole})",
        )
        return {
            "access_token": tok,
            "token_type": "bearer",
            "user": {"id": uid, "name": uname, "email": email.strip().lower(), "role": urole},
        }
    finally:
        conn.close()


def logout(token: str) -> None:
    if token:
        _LOCAL_SESSIONS.pop(token, None)
    if is_api_available():
        try:
            _request("POST", "/auth/logout", token=token)
        except Exception:
            pass


def analyze(text: str, token: str, source_name: str = "Direct Entry") -> AnalysisReport:
    if is_api_available():
        return analysis_report_from_dict(
            _request("POST", "/analyze", token=token, json={"text": text, "source_name": source_name})
        )
    if True:
        from .pipeline import analyze_text
        from .storage import (
            connect_database,
            save_api_log,
            save_audit_trail,
            save_journal_entry,
            save_mood_entry,
            save_recommendation,
            save_report,
        )

        t0 = time.perf_counter()
        report = analyze_text(text, source_name=source_name)
        try:
            conn = connect_database()
            uid = _get_current_user_id()
            rec_dict = report.recommendations if hasattr(report, "recommendations") else {}
            sentiment_lbl = (
                report.sentiment_label.value
                if hasattr(report.sentiment_label, "value")
                else str(report.sentiment_label)
            )
            save_mood_entry(
                connection=conn,
                user_id=uid,
                source_type=getattr(report.input_summary, "source_type", "text"),
                source_name=source_name,
                original_text=report.original_text,
                cleaned_text=report.cleaned_text,
                tokens=report.tokens,
                sentiment_label=sentiment_lbl,
                neg_score=report.sentiment_scores.negative,
                neu_score=report.sentiment_scores.neutral,
                pos_score=report.sentiment_scores.positive,
                compound_score=report.sentiment_scores.compound,
                dominant_emotion=getattr(report, "dominant_emotion", "neutral"),
                emotion_scores=getattr(report, "emotion_scores", {}),
                recommendation=rec_dict,
            )
            # Store 2. Journal entries (Journal text)
            jid = save_journal_entry(conn, uid, report.original_text)

            # Store 3. Recommendations (Recommendation id, user id, report data)
            save_recommendation(conn, uid, report.to_dict())

            # Store 4. Audit trails (User id, username, journal id, recommendation)
            uname = "Team Member"
            try:
                import streamlit as st
                uname = st.session_state.get("user_name", "Team Member")
            except Exception:
                pass
            rec_title = str(rec_dict.get("title", "Well-Being Recommendation")) if isinstance(rec_dict, dict) else "Well-Being Recommendation"
            save_audit_trail(conn, uid, uname, jid, rec_title, action="JOURNAL_ANALYSIS")

            # Store 6. Reports (Report id, user id, username, timestamp)
            save_report(conn, uid, uname, report.to_dict())
            save_api_log(conn, "POST", "/analyze", 200, max(0.1, (time.perf_counter() - t0) * 1000.0))
        except Exception as ex:
            logger.debug("Local persistence log: %s", ex)
        return report


def analyze_file(
    content: bytes, filename: str, token: str, text_column: str | None = None
) -> list[AnalysisReport]:
    try:
        params = {"text_column": text_column} if text_column else None
        result = _request(
            "POST",
            "/analyze/file",
            token=token,
            params=params,
            files={"file": (filename, content)},
        )
        return [analysis_report_from_dict(item) for item in result.get("items", [])]
    except Exception:
        import pandas as pd

        from .pipeline import analyze_text
        from .storage import connect_database, save_mood_entry

        reports: list[AnalysisReport] = []
        conn = connect_database()
        uid = _get_current_user_id()

        lines_or_texts: list[str] = []
        if filename.lower().endswith(".csv") or filename.lower().endswith(".tsv"):
            try:
                sep = "\t" if filename.lower().endswith(".tsv") else ","
                df = pd.read_csv(io.BytesIO(content), sep=sep)
                col = text_column or next(
                    (c for c in ["feedback", "text", "reflection", "content", "journal", "message"] if c in df.columns),
                    df.columns[0] if not df.empty else None,
                )
                if col and col in df.columns:
                    lines_or_texts = [str(v).strip() for v in df[col].dropna().tolist()[:100] if str(v).strip()]
            except Exception:
                lines_or_texts = []
        else:
            text = content.decode("utf-8", errors="replace")
            lines_or_texts = [line.strip() for line in text.splitlines()[:100] if line.strip()]

        for txt in lines_or_texts:
            rep = analyze_text(txt, source_name=f"Batch: {filename}")
            reports.append(rep)
            try:
                sent_lbl = rep.sentiment_label.value if hasattr(rep.sentiment_label, "value") else str(rep.sentiment_label)
                save_mood_entry(
                    connection=conn,
                    user_id=uid,
                    source_type="file",
                    source_name=filename,
                    original_text=rep.original_text,
                    cleaned_text=rep.cleaned_text,
                    tokens=rep.tokens,
                    sentiment_label=sent_lbl,
                    neg_score=rep.sentiment_scores.negative,
                    neu_score=rep.sentiment_scores.neutral,
                    pos_score=rep.sentiment_scores.positive,
                    compound_score=rep.sentiment_scores.compound,
                    dominant_emotion=getattr(rep, "dominant_emotion", "neutral"),
                    emotion_scores=getattr(rep, "emotion_scores", {}),
                    recommendation=getattr(rep, "recommendations", {}),
                )
            except Exception:
                pass
        return reports


def _seed_starter_reflections(conn: Any, user_id: int) -> None:
    """Pre-seed realistic longitudinal workplace reflections so telemetry immediately renders."""
    starter_data = [
        (
            "Shipped our quarterly cross-functional release on schedule today! Team collaboration was seamless and we celebrated over afternoon tea.",
            "[Workplace] Release & Peer Collaboration",
            "joy",
            "positive",
            0.0,
            0.32,
            0.68,
            0.89,
            (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d %H:%M:%S"),
        ),
        (
            "Feeling slightly fatigued after 4 hours of consecutive architecture reviews. Taking a mindful walk outside to refresh focus before evening wrap-up.",
            "[Deep Focus] System Architecture Review",
            "fatigue",
            "negative",
            0.41,
            0.49,
            0.10,
            -0.45,
            (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"),
        ),
        (
            "Mid-week retrospective went smoothly. Addressed technical debt items and agreed on clear boundaries for sprint commitments.",
            "[Organization] Retrospective Alignment",
            "neutral",
            "neutral",
            0.05,
            0.75,
            0.20,
            0.28,
            (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
        ),
        (
            "Experienced elevated stress this morning due to high-priority client incidents, but somatic box breathing helped center my breathing and calm my heart rate.",
            "[Resilience] Production Incident Management",
            "stress",
            "negative",
            0.48,
            0.40,
            0.12,
            -0.58,
            (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        ),
        (
            "Quiet Friday morning dedicated to deep reflective thinking, code refactoring, and preparing next week's wellness initiatives. Serene and centered.",
            "[Mindfulness] Deep Work & Calm Reflection",
            "joy",
            "positive",
            0.0,
            0.26,
            0.74,
            0.92,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    ]
    for orig, src, emo, sent_lbl, neg, neu, pos, comp, created in starter_data:
        try:
            conn.execute(
                """
                INSERT INTO mood_entries (
                    user_id, source_type, source_name, original_text, cleaned_text,
                    tokens, sentiment_label, neg_score, neu_score, pos_score, compound_score,
                    dominant_emotion, emotion_scores, recommendation, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    "text",
                    src,
                    orig,
                    orig.lower(),
                    "reflection, workplace, resilience, mindfulness",
                    sent_lbl,
                    neg,
                    neu,
                    pos,
                    comp,
                    emo,
                    '{"joy": 0.82, "calm": 0.64}',
                    '{"primary_practice": "Box Breathing", "action": "4-4-4-4 rhythm"}',
                    created,
                ),
            )
        except Exception:
            pass
    conn.commit()


def history(user_id: int, token: str, limit: int = 500) -> list[dict[str, Any]]:
    if is_api_available():
        response = _request("GET", f"/history/{user_id}", token=token, params={"limit": limit})
        entries = response.get("items", [])
        for entry in entries:
            entry["tokens"] = ", ".join(entry.get("tokens") or [])
            entry["recommendation"] = entry.get("recommendation") or {}
        return entries

    # User-scoped SQLite in-process fallback (never fall back to user_id=None)
    from .storage import connect_database, get_user_mood_entries, save_api_log

    t0 = time.perf_counter()
    conn = connect_database()
    try:
        raw_entries = get_user_mood_entries(conn, user_id=user_id)
        save_api_log(conn, "GET", f"/history/{user_id}", 200, max(0.1, (time.perf_counter() - t0) * 1000.0))
    finally:
        conn.close()

    formatted: list[dict[str, Any]] = []
    for e in raw_entries[:limit]:
        rec = e.get("recommendation", "")
        if isinstance(rec, str):
            try:
                rec = json.loads(rec) if rec else {}
            except Exception:
                rec = {}
        emo_s = e.get("emotion_scores", "{}")
        if isinstance(emo_s, str):
            try:
                emo_s = json.loads(emo_s) if emo_s else {}
            except Exception:
                emo_s = {}
        toks = e.get("tokens", "")
        if isinstance(toks, list):
            tok_str = ", ".join(toks)
        else:
            tok_str = str(toks)

        formatted.append(
            {
                "id": e["id"],
                "user_id": e.get("user_id", user_id),
                "source_type": e.get("source_type", "text"),
                "source_name": e.get("source_name", "Workplace Reflection"),
                "original_text": e["original_text"],
                "cleaned_text": e["cleaned_text"],
                "tokens": tok_str,
                "sentiment_label": e["sentiment_label"],
                "neg_score": float(e["neg_score"]),
                "neu_score": float(e["neu_score"]),
                "pos_score": float(e["pos_score"]),
                "compound_score": float(e["compound_score"]),
                "dominant_emotion": e.get("dominant_emotion", "neutral"),
                "emotion_scores": emo_s,
                "recommendation": rec,
                "created_at": e.get("created_at"),
            }
        )
    return formatted


def delete_entry(entry_id: int, token: str) -> None:
    if is_api_available():
        _request("DELETE", f"/history/{entry_id}", token=token)
        return
    from .storage import connect_database, delete_mood_entry, save_api_log

    t0 = time.perf_counter()
    conn = connect_database()
    try:
        uid = _get_current_user_id()
        delete_mood_entry(conn, entry_id, uid)
        save_api_log(conn, "DELETE", "/history/{entry_id}", 200, max(0.1, (time.perf_counter() - t0) * 1000.0))
    finally:
        conn.close()


def delete_all_user_entries(user_id: int, token: str | None = None) -> int:
    """Delete all mood entries belonging to the authenticated user."""
    if is_api_available():
        deleted = 0
        while True:
            items = history(user_id, token or "", limit=200)
            if not items:
                break
            for item in items:
                _request("DELETE", f"/history/{item['id']}", token=token)
                deleted += 1
            if len(items) < 200:
                break
        return deleted

    from .storage import connect_database, delete_user_mood_entries, save_api_log

    t0 = time.perf_counter()
    conn = connect_database()
    try:
        count = delete_user_mood_entries(conn, user_id)
        save_api_log(conn, "DELETE", "/history/all", 200, max(0.1, (time.perf_counter() - t0) * 1000.0))
        return count
    finally:
        conn.close()


def analysis_report_from_dict(data: dict[str, Any]) -> AnalysisReport:
    summary = data["input_summary"]
    scores = data["sentiment_scores"]
    return AnalysisReport(
        input_summary=InputSummary(
            source_type=summary["source_type"],
            source_name=summary["source_name"],
            character_count=int(summary["character_count"]),
            line_count=int(summary["line_count"]),
            token_count=int(summary["token_count"]),
        ),
        original_text=data["original_text"],
        cleaned_text=data["cleaned_text"],
        processed_text=data["processed_text"],
        tokens=tuple(data.get("tokens", [])),
        sentiment_scores=SentimentScores(
            negative=float(scores["negative"]),
            neutral=float(scores["neutral"]),
            positive=float(scores["positive"]),
            compound=float(scores["compound"]),
        ),
        sentiment_label=SentimentLabel(data["sentiment_label"]),
        counts=data.get("counts", {}),
        vader_input_text=data.get("vader_input_text", ""),
        emotion_method=data.get("emotion_method", "legacy_unknown"),
        dominant_emotion=data.get("dominant_emotion", "neutral"),
        emotion_scores=data.get("emotion_scores", {}),
        recommendations=data.get("recommendations", {}),
    )


def compare_models_client(text: str, token: str | None = None) -> dict[str, Any]:
    """Call the comparative analysis endpoint across baseline and transformer models."""
    if is_api_available() and token:
        return _request("POST", "/analyze/compare", token=token, json={"text": text})
    from .comparator import compare_models

    return compare_models(text)


def get_wellness_exercises_client() -> list[dict[str, Any]]:
    """Fetch catalog of evidence-based somatic and cognitive micro-interventions."""
    if is_api_available():
        return _request("GET", "/wellness/exercises")
    from .wellness_tools import get_guided_exercises

    return get_guided_exercises()


def get_team_burnout_risk_client(
    entries: list[dict[str, Any]] | None = None,
    scenario: str = "balanced",
    days: int = 14,
    team_size: int = 8,
    token: str | None = None,
) -> dict[str, Any]:
    """Calculate team burnout risk score and resilience index (Admin required)."""
    if is_api_available():
        payload = {
            "entries": entries or [],
            "scenario": scenario,
            "days": days,
            "team_size": team_size,
        }
        return _request("POST", "/team/burnout-risk", token=token, json=payload)
    _require_local_admin(token)
    from .team_analytics import calculate_team_burnout_risk

    return calculate_team_burnout_risk(entries or [])


def get_mood_food_plan_client(emotion: str, dietary_pref: str = "any") -> dict[str, Any]:
    """Fetch nutritional neuroscience recommendations for the detected emotion."""
    try:
        return _request("GET", f"/wellness/food-plan/{emotion}", params={"dietary_pref": dietary_pref})
    except Exception:
        from .food_planner import get_mood_food_plan

        return get_mood_food_plan(emotion, dietary_pref=dietary_pref)


def get_meditations_client() -> list[dict[str, Any]]:
    """Fetch Medito-inspired mindfulness packs from the API."""
    try:
        return _request("GET", "/wellness/meditations")
    except Exception:
        from .meditation import get_meditation_packs

        return get_meditation_packs()


def get_soundscapes_client() -> list[dict[str, str]]:
    """Fetch ambient soundscape presets from the API."""
    try:
        return _request("GET", "/wellness/soundscapes")
    except Exception:
        from .meditation import get_soundscape_catalog

        return get_soundscape_catalog()


def get_yoga_remedies_client() -> list[dict[str, Any]]:
    """Fetch catalog of evidence-based therapeutic yoga poses from the API."""
    try:
        return _request("GET", "/wellness/yoga-remedies")
    except Exception:
        from .yoga import get_all_yoga_remedies

        return get_all_yoga_remedies()


def get_yoga_remedy_client(emotion: str) -> dict[str, Any]:
    """Fetch targeted yoga remedy for an emotional state from the API."""
    try:
        return _request("GET", f"/wellness/yoga-remedies/{emotion}")
    except Exception:
        from .yoga import get_yoga_remedy_for_emotion

        return get_yoga_remedy_for_emotion(emotion)


def submit_feedback(
    recommendation_id: int | None,
    report_data: dict[str, Any] | str,
    accepted_or_rejected: str = "accepted",
    token: str | None = None,
    api_token: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Store 5. User feedback (recommendation id, report data, accepted/rejected)."""
    effective_token = token or api_token or kwargs.get("auth_token")
    if is_api_available():
        return _request(
            "POST",
            "/feedback",
            token=effective_token,
            json={
                "recommendation_id": recommendation_id,
                "report_data": report_data if isinstance(report_data, dict) else {},
                "accepted_or_rejected": accepted_or_rejected,
            },
        )

    from .storage import connect_database, save_api_log, save_audit_trail, save_user_feedback

    t0 = time.perf_counter()
    conn = connect_database()
    try:
        decision = str(accepted_or_rejected or "accepted").lower()
        fb_id = save_user_feedback(conn, recommendation_id, report_data, decision)
        uid = _get_current_user_id()
        uname = "Team Member"
        try:
            import streamlit as st

            uname = str(st.session_state.get("user_name") or "Team Member")
        except Exception:
            pass
        rec_title = (
            str(report_data.get("title") or "Care suggestion")
            if isinstance(report_data, dict)
            else "Care suggestion"
        )
        action_code = "RECOMMENDATION_ACCEPT" if decision == "accepted" else "RECOMMENDATION_REJECT"
        save_audit_trail(conn, uid, uname, None, f"{decision}: {rec_title}", action=action_code)
        save_api_log(conn, "POST", "/feedback", 200, max(0.1, (time.perf_counter() - t0) * 1000.0))
        return {
            "id": fb_id,
            "recommendation_id": recommendation_id,
            "report_data": report_data,
            "accepted_or_rejected": decision,
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        conn.close()


def get_database_entities(token: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Retrieve privacy-filtered database entities (Administrator required)."""
    if is_api_available():
        return _request("GET", "/database/entities", token=token)
    _require_local_admin(token)
    from .storage import connect_database, get_all_database_entities, save_api_log

    t0 = time.perf_counter()
    conn = connect_database()
    try:
        res = get_all_database_entities(conn, redact_private_text=True)
        save_api_log(conn, "GET", "/database/entities", 200, max(0.1, (time.perf_counter() - t0) * 1000.0))
        return res
    finally:
        conn.close()


def get_admin_analytics(token: str | None = None) -> dict[str, Any]:
    """Retrieve Admin Dashboard Analytics metrics (Administrator required)."""
    if is_api_available():
        return _request("GET", "/admin/analytics", token=token)
    _require_local_admin(token)
    from .storage import connect_database, save_api_log
    from .storage import get_admin_analytics as get_storage_admin_analytics

    t0 = time.perf_counter()
    conn = connect_database()
    try:
        res = get_storage_admin_analytics(conn)
        save_api_log(conn, "GET", "/admin/analytics", 200, max(0.1, (time.perf_counter() - t0) * 1000.0))
        return res
    finally:
        conn.close()


def get_endpoint_telemetry(token: str | None = None) -> dict[str, Any]:
    """Retrieve aggregated per-endpoint telemetry metrics (Administrator required)."""
    if is_api_available():
        return _request("GET", "/admin/telemetry", token=token)
    _require_local_admin(token)
    from .storage import connect_database, get_api_telemetry_summary, save_api_log

    t0 = time.perf_counter()
    conn = connect_database()
    try:
        res = get_api_telemetry_summary(conn)
        save_api_log(conn, "GET", "/admin/telemetry", 200, max(0.1, (time.perf_counter() - t0) * 1000.0))
        return res
    finally:
        conn.close()


def update_user_role_client(
    target_user_id: int, new_role: str, token: str | None = None
) -> dict[str, Any]:
    """Update a user's persisted role (Administrator required)."""
    if is_api_available():
        return _request(
            "POST",
            f"/admin/users/{target_user_id}/role",
            token=token,
            json={"role": new_role},
        )
    admin_uid = _require_local_admin(token)
    from .storage import connect_database, normalize_role, save_api_log, save_audit_trail

    normalized = normalize_role(new_role)
    if target_user_id == admin_uid and normalized != "admin":
        raise ApiError("You cannot revoke your own active administrator role.")
    t0 = time.perf_counter()
    conn = connect_database()
    try:
        row = conn.execute(
            "SELECT id, email FROM users WHERE id = ?", (target_user_id,)
        ).fetchone()
        if row is None:
            raise ApiError("User not found.")
        admin_row = conn.execute(
            "SELECT name FROM users WHERE id = ?", (admin_uid,)
        ).fetchone()
        admin_name = str(admin_row["name"]) if admin_row else "Administrator"
        conn.execute(
            "UPDATE users SET role = ? WHERE id = ?",
            (normalized, target_user_id),
        )
        conn.commit()
        save_audit_trail(
            conn,
            admin_uid,
            admin_name,
            None,
            f"Updated user #{target_user_id} ({row['email']}) role to {normalized}",
            action="ADMIN_ROLE_UPDATE",
        )
        save_api_log(
            conn,
            "POST",
            "/admin/users/{target_user_id}/role",
            200,
            max(0.1, (time.perf_counter() - t0) * 1000.0),
        )
        return {"id": int(row["id"]), "email": str(row["email"]), "role": normalized}
    finally:
        conn.close()


