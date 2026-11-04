"""Privacy-conscious, single-process Streamlit Cloud showcase for MoodMentor."""
from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from html import escape
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

# Hosted mode uses bundled VADER and corpus-free tokenization, with no runtime downloads.
os.environ["WELLNESS_DISABLE_TRANSFORMER"] = "1"
os.environ["WELLNESS_LIGHTWEIGHT_PREPROCESSING"] = "1"
os.environ["WELLNESS_BUNDLED_VADER"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from wellness.errors import InputValidationError, NLTKResourceError
from wellness.pipeline import (
    EmotionClassifier,
    analyze_text,
    classify_emotions_with_hf_api,
    classify_emotions_with_method,
)
from wellness.preprocessing import get_preprocessing_breakdown
from wellness.api_client import (
    ApiError,
    analyze as api_analyze,
    history as api_history,
    login as api_login,
    logout as api_logout,
    register as api_register,
)
from wellness.report import (
    AnalysisReport,
    InputSummary,
    SentimentLabel,
    SentimentScores,
)

st.set_page_config(
    page_title="MoodMentor | Mindful Check-in",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#19352d; --muted:#657b71; --paper:#f4f7f3; --line:#e0e9e1; --forest:#155c49; --leaf:#c9ebad; }
    html,body,[class*="css"] { font-family:'Segoe UI',system-ui,sans-serif; color:var(--ink); }
    .stApp { background:radial-gradient(850px circle at 86% 3%,rgba(206,231,211,.48),transparent 48%),var(--paper); }
    .block-container { max-width:1320px; padding:2rem 2.5rem 4rem; }
    h1,h2,h3 { font-family:'Segoe UI',system-ui,sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
    .brand { display:flex;align-items:center;gap:12px;margin:0 0 20px; }
    .mark { display:grid;place-items:center;width:46px;height:46px;border-radius:16px;background:var(--forest);color:white;font:800 25px 'Segoe UI',system-ui,sans-serif; }
    .brand-name { font:800 1.3rem 'Segoe UI',system-ui,sans-serif;color:var(--ink); }
    .brand-caption { color:var(--muted);font-size:.8rem; }
    .hero { padding:30px 34px;border-radius:24px;color:#f3f8f2;background:linear-gradient(128deg,#103e34,#155b48 65%,#26745a);box-shadow:0 18px 44px #1848391c;margin-bottom:18px; }
    .eyebrow { color:#a9d9bb;text-transform:uppercase;font-size:.7rem;font-weight:800;letter-spacing:.15em; }
    .hero h1 { color:white;margin:12px 0 6px;font-size:clamp(2rem,4vw,3.1rem); }
    .hero p { max-width:720px;color:#d1e4d7;line-height:1.7;margin-bottom:0; }
    [data-testid="stMetric"] { padding:15px 18px;border:1px solid var(--line);border-radius:16px;background:white; }
    [data-testid="stVerticalBlockBorderWrapper"] { border:1px solid var(--line)!important;border-radius:18px!important;background:#ffffffcc; }
    [data-testid="stTextArea"] textarea { min-height:190px;border-radius:14px;line-height:1.65; }
    .stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"] button { min-height:43px;border-radius:12px;font-weight:700; }
    .stButton>button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background:var(--forest);border-color:var(--forest); }
    .stButton>button:hover,.stDownloadButton>button:hover { border-color:var(--forest);color:var(--forest); }
    button[data-baseweb="tab"][aria-selected="true"] { color:var(--forest);border-bottom-color:var(--forest); }
    :focus-visible { outline:3px solid #2d8467 !important; outline-offset:2px; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#103d33,#12382f); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#d0e1d5; }
    @media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;transition-duration:.01ms!important}}
    @media(max-width:800px){.block-container{padding:1rem}.hero{padding:24px}}
    </style>
    """,
    unsafe_allow_html=True,
)

MOOD_SCORE = {"Very low": 1, "Low": 2, "Okay": 3, "Good": 4, "Very good": 5}
CONTEXT_TAGS = ["Work or study", "Relationships", "Health", "Rest", "Family", "Money", "Personal growth"]

PROMPTS = {
    "Choose a gentle prompt (optional)": "",
    "What felt heavy, hopeful, or important today?": "What felt heavy, hopeful, or important today?",
    "What is one thing you handled well?": "What is one thing you handled well, even if it was small?",
    "What would help you feel a little more supported?": "What would help you feel a little more supported right now?",
    "What has been taking most of your energy?": "What has been taking most of your energy lately?",
}


def _clear_journal_draft() -> None:
    st.session_state.cloud_journal = ""


def _configured_api_url() -> str | None:
    """Return only an explicitly configured account API URL, never localhost fallback."""
    try:
        configured = st.secrets.get("API_URL", "")
    except Exception:
        configured = ""
    value = str(configured or os.getenv("API_URL", "")).strip().rstrip("/")
    return value or None


API_URL = _configured_api_url()


def _history_entry_to_record(entry: dict[str, Any]) -> dict[str, Any]:
    """Adapt an authenticated API history row to the showcase report format."""
    original_text = str(entry.get("original_text", ""))
    tokens_value = entry.get("tokens") or ""
    if isinstance(tokens_value, str):
        tokens = tuple(part.strip() for part in tokens_value.split(",") if part.strip())
    else:
        tokens = tuple(str(part) for part in tokens_value)
    timestamp_value = str(entry.get("created_at") or "")
    try:
        created_at = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
    except ValueError:
        created_at = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    recommendation = entry.get("recommendation") or {}
    scores = entry.get("emotion_scores") or {}
    report = AnalysisReport(
        input_summary=InputSummary(
            source_type=str(entry.get("source_type") or "text"),
            source_name=str(entry.get("source_name") or "Journal Entry"),
            character_count=len(original_text),
            line_count=len(original_text.splitlines()),
            token_count=len(tokens),
        ),
        original_text=original_text,
        cleaned_text=str(entry.get("cleaned_text") or ""),
        processed_text=str(entry.get("processed_text") or ""),
        tokens=tokens,
        sentiment_scores=SentimentScores(
            negative=float(entry.get("neg_score") or 0),
            neutral=float(entry.get("neu_score") or 0),
            positive=float(entry.get("pos_score") or 0),
            compound=float(entry.get("compound_score") or 0),
        ),
        sentiment_label=SentimentLabel(str(entry.get("sentiment_label", "neutral"))),
        counts=entry.get("counts") or {},
        vader_input_text=str(entry.get("vader_input_text") or ""),
        emotion_method=str(entry.get("emotion_method") or "unknown"),
        dominant_emotion=str(entry.get("dominant_emotion") or "neutral"),
        emotion_scores={str(k): float(v) for k, v in scores.items()},
        recommendations=recommendation,
    )
    api_id = entry.get("id")
    return {
        "id": f"account:{api_id}",
        "api_id": api_id,
        "created_at": created_at,
        "source": str(entry.get("source_name") or "Journal Entry"),
        "report": report,
    }


def _refresh_account_records() -> None:
    user = st.session_state.get("account_user") or {}
    token = st.session_state.get("account_token")
    if not user.get("id") or not token:
        st.session_state.records = []
        st.session_state.selected_record_id = None
        return
    entries = api_history(int(user["id"]), token, limit=500)
    records = [_history_entry_to_record(entry) for entry in entries]
    records.sort(key=lambda item: item["created_at"], reverse=True)
    st.session_state.records = records
    current = st.session_state.get("selected_record_id")
    ids = {item["id"] for item in records}
    st.session_state.selected_record_id = current if current in ids else (
        records[0]["id"] if records else None
    )


def _apply_auth(auth: dict[str, Any]) -> None:
    """Store a server-issued session and load that user's own saved history."""
    st.session_state.account_token = auth["access_token"]
    st.session_state.account_user = auth["user"]
    st.session_state.guest_mode = False
    st.session_state.records = []
    st.session_state.selected_record_id = None
    try:
        _refresh_account_records()
        st.session_state.account_flash = (
            f"Welcome, {auth['user'].get('name', 'back')} — your saved check-ins are ready."
        )
    except ApiError as exc:
        st.session_state.account_flash = (
            f"You are signed in, but saved history could not be loaded: {exc}"
        )


def _render_auth_gate() -> None:
    """Show a compact account screen when a visitor chooses account access."""
    account_flash = st.session_state.pop("account_flash", None)
    if account_flash:
        if "could not confirm token revocation" in account_flash:
            st.warning(account_flash)
        else:
            st.info(account_flash)
    st.markdown(
        '<div class="brand"><div class="mark">m</div><div><div class="brand-name">MoodMentor</div>'
        '<div class="brand-caption">A thoughtful space for your daily check-in</div></div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown(
            '<div class="hero"><div class="eyebrow">YOUR REFLECTIONS, READY WHEN YOU ARE</div>'
            '<h1>Make space for how you feel.</h1>'
            '<p>Sign in to revisit saved check-ins and notice your patterns over time. '
            'Your account is managed by the configured MoodMentor API.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "#### Your account, your history\n"
            "MoodMentor stores journal text and analysis in the connected account database. "
            "Use a password you do not reuse elsewhere, and avoid writing identifying details."
        )
    with right:
        if not API_URL:
            st.markdown("### Account access is not connected yet")
            st.info(
                "The hosted app needs an HTTPS MoodMentor API with a persistent database before "
                "it can create or verify accounts. Add API_URL in Streamlit Community Cloud "
                "under App settings → Secrets. Example:"
            )
            st.code('API_URL = "https://your-moodmentor-api.example.com"', language="toml")
            st.caption(
                "Do not put a password or database credential here. API_URL should point to your "
                "public HTTPS address for your FastAPI service. Keep database credentials on the API host, not in this setting."
            )
            if st.button("Continue as a guest", type="primary", width="stretch"):
                st.session_state.guest_mode = True
                st.rerun()
            return

        st.markdown("### Welcome in")
        st.caption("Sign in to continue your reflections, or create an account to begin.")
        sign_in_tab, register_tab, guest_tab = st.tabs(
            ["Sign in", "Create account", "Guest"]
        )
        with sign_in_tab:
            with st.form("cloud_login_form"):
                email = st.text_input("Email address", placeholder="you@example.com", key="cloud_login_email")
                password = st.text_input("Password", type="password", key="cloud_login_password")
                submitted = st.form_submit_button(
                    "Sign in to MoodMentor", type="primary", width="stretch"
                )
            if submitted:
                if not email.strip() or not password:
                    st.warning("Enter your email address and password to continue.")
                else:
                    try:
                        with st.spinner("Signing in securely…"):
                            _apply_auth(api_login(email.strip(), password))
                        st.rerun()
                    except ApiError as exc:
                        st.error(str(exc))
        with register_tab:
            with st.form("cloud_register_form"):
                name = st.text_input("Your name", placeholder="How should we address you?", key="cloud_register_name")
                email = st.text_input("Email address", placeholder="you@example.com", key="cloud_register_email")
                password = st.text_input(
                    "Create password", type="password", help="Use at least 8 characters.", key="cloud_register_password"
                )
                confirm = st.text_input("Confirm password", type="password", key="cloud_register_confirm")
                submitted = st.form_submit_button(
                    "Create my account", type="primary", width="stretch"
                )
            if submitted:
                if len(password) < 8:
                    st.warning("Choose a password with at least 8 characters.")
                elif password != confirm:
                    st.warning("The passwords do not match.")
                elif not name.strip() or not email.strip():
                    st.warning("Enter your name and email address.")
                else:
                    try:
                        with st.spinner("Creating your account…"):
                            _apply_auth(api_register(name.strip(), email.strip(), password))
                        st.rerun()
                    except ApiError as exc:
                        st.error(str(exc))
        with guest_tab:
            st.write(
                "You can try the journal without creating an account. Guest check-ins stay in "
                "this temporary browser session and are not added to account history."
            )
            if st.button("Continue as a guest", type="primary", width="stretch"):
                st.session_state.guest_mode = True
                st.rerun()


def _hugging_face_token() -> str | None:
    """Read the optional inference token without exposing it in the UI or logs."""
    token = os.getenv("HF_TOKEN", "").strip()
    try:
        token = str(st.secrets.get("HF_TOKEN", token) or token).strip()
    except Exception:
        pass
    return token or None


def _new_record(
    text: str,
    source: str,
    checkin_details: dict[str, Any] | None = None,
    emotion_classifier: EmotionClassifier | None = None,
) -> dict[str, Any]:
    report = analyze_text(
        text, source_name=source, emotion_classifier=emotion_classifier
    )
    details = checkin_details or {}
    return {
        "id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc),
        "source": source,
        "report": report,
        "self_reported_mood": details.get("self_reported_mood"),
        "energy": details.get("energy"),
        "sleep_hours": details.get("sleep_hours"),
        "context_tags": details.get("context_tags", []),
    }


def _history_frame(records: list[dict[str, Any]]):
    import pandas as pd

    rows = []
    for item in records:
        report: AnalysisReport = item["report"]
        rows.append(
            {
                "Date": item["created_at"].astimezone().strftime("%Y-%m-%d %H:%M"),
                "Sentiment": report.sentiment_label.value.title(),
                "VADER compound": report.sentiment_scores.compound,
                "Dominant emotion": report.dominant_emotion.title(),
                "Estimator": report.emotion_method.replace("_", " "),
                "Self-reported mood": item.get("self_reported_mood") or "Not recorded",
                "Mood score": MOOD_SCORE.get(item.get("self_reported_mood")),
                "Energy (1–5)": item.get("energy"),
                "Sleep (hours)": item.get("sleep_hours"),
                "Context tags": ", ".join(item.get("context_tags", [])),
                "Source": item["source"],
            }
        )
    return pd.DataFrame(rows)


def _make_pdf(item: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    report: AnalysisReport = item["report"]
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("MoodTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#155c49"))
    story = [
        Paragraph("MoodMentor Check-in Report", title),
        Spacer(1, 10),
        Paragraph(escape(item["created_at"].astimezone().strftime("%d %b %Y, %I:%M %p %Z")), styles["Normal"]),
        Spacer(1, 12),
    ]
    score = report.sentiment_scores
    metrics = [
        ["Sentiment", report.sentiment_label.value.title(), "VADER compound", f"{score.compound:+.3f}"],
        ["Dominant emotion", report.dominant_emotion.title(), "Estimator", report.emotion_method.replace("_", " ")],
    ]
    table = Table(metrics, colWidths=[1.4 * inch, 1.65 * inch, 1.4 * inch, 1.6 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef5ef")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#19352d")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9e7dc")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
    ]))
    story.extend([table, Spacer(1, 16)])
    checkin_details = []
    if item.get("self_reported_mood"):
        checkin_details.append(f"Self-rated mood: {item['self_reported_mood']}")
    if item.get("energy") is not None:
        checkin_details.append(f"Energy: {item['energy']}/5")
    if item.get("sleep_hours") is not None:
        checkin_details.append(f"Sleep: {item['sleep_hours']} hours")
    if item.get("context_tags"):
        checkin_details.append("Context: " + ", ".join(item["context_tags"]))
    if checkin_details:
        story.extend([
            Paragraph("Your quick check-in", styles["Heading2"]),
            Paragraph(escape(" · ".join(checkin_details)), styles["BodyText"]),
            Spacer(1, 12),
        ])
    story.append(Paragraph("Your reflection", styles["Heading2"]))
    story.append(Paragraph(escape(report.original_text).replace("\n", "<br/>"), styles["BodyText"]))
    story.extend([Spacer(1, 14), Paragraph("Supportive next steps", styles["Heading2"])])
    rec = report.recommendations
    story.append(Paragraph(escape(str(rec.get("summary", "A gentle check-in suggestion."))), styles["BodyText"]))
    for path in rec.get("action_paths", []):
        story.extend([Spacer(1, 8), Paragraph(escape(str(path.get("title", "A way forward"))), styles["Heading3"])])
        story.append(Paragraph(escape(str(path.get("description", ""))), styles["BodyText"]))
        for step in path.get("steps", []):
            story.append(Paragraph("• " + escape(str(step)), styles["BodyText"]))
    story.extend([Spacer(1, 16), Paragraph("MoodMentor is a self-reflection tool, not a clinical assessment or diagnosis.", styles["Italic"])])
    doc.build(story)
    return buffer.getvalue()


def _analyze_many(
    items: list[tuple[str, str]],
    checkin_details: dict[str, Any] | None = None,
    emotion_classifier: EmotionClassifier | None = None,
) -> None:
    valid_items = [(value, source) for value, source in items if value.strip()]
    if not valid_items:
        st.info("Add some text before starting an analysis.")
        return

    token = st.session_state.get("account_token")
    if token:
        saved = 0
        try:
            with st.spinner(
                f"Analyzing and saving {len(valid_items)} entr"
                f"{'y' if len(valid_items) == 1 else 'ies'} to your account…"
            ):
                for value, source in valid_items:
                    api_analyze(value, token, source_name=source)
                    saved += 1
                _refresh_account_records()
            st.success(
                f"Saved {saved} check-in{'s' if saved != 1 else ''} to your account history."
            )
        except ApiError as exc:
            try:
                _refresh_account_records()
            except ApiError:
                pass
            if saved:
                st.warning(f"Saved {saved} of {len(valid_items)} entries before the API stopped: {exc}")
            else:
                st.error(f"Your entry could not be saved. {exc}")
        return

    try:
        with st.spinner(f"Reflecting on {len(valid_items)} entr{'y' if len(valid_items) == 1 else 'ies'}…"):
            records = [
                _new_record(
                    value,
                    source,
                    checkin_details if source == "Journal Entry" else None,
                    emotion_classifier if source == "Journal Entry" else None,
                )
                for value, source in valid_items
            ]
        st.session_state.records = records + st.session_state.records
        if records:
            st.session_state.selected_record_id = records[0]["id"]
            st.success(
                f"Saved {len(records)} check-in{'s' if len(records) != 1 else ''} "
                "to this browser session."
            )
    except (InputValidationError, NLTKResourceError, RuntimeError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")

if "records" not in st.session_state:
    st.session_state.records = []
if "selected_record_id" not in st.session_state:
    st.session_state.selected_record_id = None
if "account_token" not in st.session_state:
    st.session_state.account_token = None
if "account_user" not in st.session_state:
    st.session_state.account_user = None
if "guest_mode" not in st.session_state:
    st.session_state.guest_mode = not bool(API_URL)

if not st.session_state.get("account_token") and not st.session_state.guest_mode:
    _render_auth_gate()
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand"><div class="mark">m</div><div><div class="brand-name" style="color:white">MoodMentor</div><div class="brand-caption" style="color:#b5d0c0">Your mindful space</div></div></div>', unsafe_allow_html=True)
    account_user = st.session_state.get("account_user")
    if account_user and st.session_state.get("account_token"):
        st.markdown("### Your account")
        st.write(f"**{account_user.get('name', 'MoodMentor member')}**")
        st.caption(account_user.get("email", ""))
        st.metric("Recent saved check-ins", len(st.session_state.records))
        if st.button("Refresh saved history", width="stretch"):
            try:
                _refresh_account_records()
                st.success("Your saved history is up to date.")
            except ApiError as exc:
                st.error(f"History could not be refreshed: {exc}")
        if st.button("Sign out", width="stretch", type="secondary"):
            signout_error = None
            try:
                api_logout(st.session_state.account_token)
            except ApiError as exc:
                signout_error = str(exc)
            st.session_state.account_token = None
            st.session_state.account_user = None
            st.session_state.records = []
            st.session_state.selected_record_id = None
            st.session_state.guest_mode = False
            st.session_state.account_flash = (
                "You have been signed out on this device."
                if signout_error is None
                else "You are signed out here. The server could not confirm token revocation; "
                "sign in again if you continue using the account."
            )
            st.session_state.cloud_journal = ""
            st.rerun()
        st.caption("Your account history is stored by the connected MoodMentor API.")
    else:
        st.markdown("### Guest session")
        st.metric("Saved check-ins", len(st.session_state.records))
        st.caption("Guest check-ins stay in this temporary session and may disappear when it ends.")
        if st.session_state.records and st.button(
            "Clear this session's history", width="stretch"
        ):
            st.session_state.records = []
            st.session_state.selected_record_id = None
            st.rerun()
        if st.button("Sign in / create account", width="stretch"):
            st.session_state.guest_mode = False
            st.rerun()

    account_flash = st.session_state.pop("account_flash", None)
    if account_flash:
        st.info(account_flash)

    if st.session_state.records:
        saved_records = st.session_state.records
        saved_ids = [item["id"] for item in saved_records]
        current_id = st.session_state.selected_record_id
        selected_index = saved_ids.index(current_id) if current_id in saved_ids else 0
        record_labels = {
            item["id"]: (
                item["created_at"].astimezone().strftime("%b %d · %I:%M %p")
                + f" · {item['report'].dominant_emotion.title()}"
            )
            for item in saved_records
        }
        st.session_state.selected_record_id = st.selectbox(
            "Revisit a check-in",
            saved_ids,
            index=selected_index,
            format_func=lambda record_id: record_labels[record_id],
        )
    st.markdown("---")
    st.caption("For privacy, avoid names, contact details, or other identifying information in a public cloud demo.")

st.markdown(
    '<div class="hero"><div class="eyebrow">A gentler way to check in</div><h1>Make space for how you feel.</h1><p>Write a few honest lines, notice the emotional signals, then choose a small next step that feels right for you.</p></div>',
    unsafe_allow_html=True,
)
account_mode = bool(
    st.session_state.get("account_token") and st.session_state.get("account_user")
)
hf_token = _hugging_face_token() if not account_mode else None
if account_mode:
    st.info(
        "Account mode: your reflection is sent to the configured MoodMentor API for analysis "
        "and saved with its results in your account history. Avoid sensitive or identifying details. "
        "Use Sign out when you finish.",
        icon="🔒",
    )
elif hf_token:
    st.info(
        "Guest mode keeps check-ins in temporary session memory. Text is processed by this hosted app. "
        "If you opt in for a journal entry, it is also sent to Hugging Face for transformer analysis. "
        "Avoid sensitive or identifying details.",
        icon="ℹ️",
    )
else:
    st.info(
        "Guest mode keeps check-ins in temporary session memory. Text is processed by this hosted app "
        "using VADER and lightweight emotion cues. Avoid sensitive or identifying details. "
        "To enable accounts and saved history, the app owner must configure API_URL in Streamlit Secrets.",
        icon="ℹ️",
    )

journal_tab, file_tab, trends_tab, evaluation_tab = st.tabs(["📝 Mood journal", "📄 Import a file", "📈 Trends & history", "🧪 Model evaluation"])
with journal_tab:
    col_editor, col_prompt = st.columns([2.1, 0.9], gap="large")
    with col_editor:
        st.markdown("### Start with how today feels")
        if account_mode:
            self_reported_mood = "Prefer not to say"
            context_tags = []
            include_wellbeing_details = False
            energy = None
            sleep_hours = None
            st.caption(
                "Account history stores your reflection and its analysis. Optional mood, sleep, "
                "energy, and context notes are available in guest mode only."
            )
        else:
            mood_options = ["Prefer not to say", *MOOD_SCORE]
            self_reported_mood = st.radio(
                "Your mood (optional)", mood_options, horizontal=True, index=0,
                help="This is your own check-in, shown separately from the language model's estimate.",
            )
            context_tags = st.multiselect(
                "What has been shaping your day? (optional)", CONTEXT_TAGS,
                help="Choose any context that feels relevant. These tags remain in this browser session.",
            )
            with st.expander("Add optional sleep and energy details"):
                include_wellbeing_details = st.checkbox("Include sleep and energy in this check-in")
                if include_wellbeing_details:
                    energy = st.slider("Energy level", 1, 5, 3, help="1 = very low · 5 = very high")
                    sleep_hours = st.slider("Hours of sleep", 0.0, 16.0, 7.0, 0.5)
                else:
                    energy = None
                    sleep_hours = None
        prompt = st.selectbox("Optional writing prompt", list(PROMPTS), key="cloud_prompt")
        if st.button("Use this prompt", key="apply_prompt", disabled=not PROMPTS[prompt]):
            st.session_state.cloud_journal = PROMPTS[prompt]
        with st.form("journal_entry_form", clear_on_submit=False):
            use_remote_emotion_model = False
            if hf_token and not account_mode:
                use_remote_emotion_model = st.checkbox(
                    "Use the GoEmotions transformer for this entry",
                    key="use_remote_emotion_model",
                    help=(
                        "Optional. If selected, this reflection is sent to Hugging Face Inference "
                        "for emotion analysis. Leave unchecked to use the in-app keyword estimator."
                    ),
                )
                st.caption(
                    "Model: SamLowe/roberta-base-go_emotions · 28 labels. "
                    "Only checked entries are sent to Hugging Face."
                )
            elif not account_mode:
                st.caption(
                    "Transformer analysis is not connected on this deployment; the lightweight estimator is used."
                )
            else:
                st.caption(
                    "Your reflection will be sent to the configured MoodMentor API and saved to your account history."
                )
            entry = st.text_area(
                "Your reflection",
                placeholder="What felt heavy, hopeful, or important today? A few sentences are enough to begin.",
                height=220,
                max_chars=6000,
                key="cloud_journal",
            )
            submitted = st.form_submit_button(
                "Analyze my mood", type="primary", width="stretch"
            )
        if submitted:
            if not entry.strip():
                st.info("Write a few words first. Your mood rating can also be recorded without sharing personal details in the text.")
            else:
                checkin_details = {
                    "self_reported_mood": (
                        None if self_reported_mood == "Prefer not to say" else self_reported_mood
                    ),
                    "context_tags": context_tags,
                    "energy": energy if include_wellbeing_details else None,
                    "sleep_hours": sleep_hours if include_wellbeing_details else None,
                }
                selected_classifier: EmotionClassifier | None = None
                if use_remote_emotion_model and hf_token:
                    def _remote_or_local_fallback(value: str):
                        try:
                            return classify_emotions_with_hf_api(
                                value, api_key=hf_token
                            )
                        except Exception:
                            st.warning(
                                "The remote transformer could not complete this request. "
                                "MoodMentor used its lightweight keyword estimator instead."
                            )
                            return classify_emotions_with_method(value)

                    selected_classifier = _remote_or_local_fallback
                _analyze_many(
                    [(entry.strip(), "Journal Entry")],
                    checkin_details,
                    selected_classifier,
                )
        if entry.strip():
            st.button(
                "Clear draft",
                key="clear_journal_draft",
                on_click=_clear_journal_draft,
            )
    with col_prompt:
        st.markdown("### A small reminder")
        st.write("There is no right way to write this. Describe a moment, a conversation, or simply how you feel right now.")
        st.caption("MoodMentor offers general reflection prompts. It cannot diagnose or replace professional support.")

with file_tab:
    st.markdown("### Bring in a text or CSV journal")
    st.caption("Accepted files: UTF-8 .txt or .csv up to 1 MB. CSV files need a text, feedback, comment, message, or content column. Up to 25 non-empty rows are analyzed per upload.")
    uploaded = st.file_uploader("Choose a journal file", type=["txt", "csv"], key="cloud_upload", max_upload_size=1)
    if uploaded is not None:
        data = uploaded.getvalue()
        if len(data) > 1_000_000:
            st.error("Please choose a file smaller than 1 MB.")
        else:
            try:
                decoded = data.decode("utf-8-sig")
                if uploaded.name.lower().endswith(".txt"):
                    file_items = [(decoded.strip(), uploaded.name)] if decoded.strip() else []
                else:
                    reader = csv.DictReader(StringIO(decoded))
                    raw_columns = reader.fieldnames or []
                    columns = [str(name or "").strip() for name in raw_columns]
                    duplicate_columns = []
                    seen_columns = set()
                    for column in columns:
                        normalized = column.casefold()
                        if normalized in seen_columns and normalized not in duplicate_columns:
                            duplicate_columns.append(normalized)
                        seen_columns.add(normalized)
                    if not columns:
                        st.error("The CSV needs a header row.")
                        file_items = []
                    elif any(not column for column in columns):
                        st.error("Every CSV column needs a non-empty header.")
                        file_items = []
                    elif duplicate_columns:
                        st.error("The CSV has repeated column names. Rename them and upload again.")
                        file_items = []
                    else:
                        reader.fieldnames = columns
                        preferred = next((c for c in columns if c.casefold() in {"text", "feedback", "comment", "message", "content"}), columns[0])
                        text_column = st.selectbox("Text column", columns, index=columns.index(preferred), key="cloud_csv_column")
                        file_items = [(str(row.get(text_column, "")).strip(), f"{uploaded.name} · row {index}") for index, row in enumerate(reader, start=2) if str(row.get(text_column, "")).strip()][:25]
                if file_items:
                    st.caption(f"Ready to analyze {len(file_items)} entr{'y' if len(file_items) == 1 else 'ies'}. Review the preview before continuing.")
                    st.dataframe(
                        [{"Source": source, "Text preview": text.replace("\n", " ")[:240]} for text, source in file_items[:5]],
                        width="stretch",
                        hide_index=True,
                    )
                    if st.button("Analyze file entries", type="primary", key="analyze_cloud_file"):
                        _analyze_many(file_items)
                elif uploaded.name.lower().endswith(".txt"):
                    st.warning("The text file is empty.")
            except UnicodeDecodeError:
                st.error("The file must use UTF-8 text encoding.")
            except csv.Error as exc:
                st.error(f"Could not read this CSV: {exc}")

    st.markdown("---")
    st.markdown("### 📚 Explore GoEmotions Archive Dataset")
    st.caption("Load a curated 20-entry slice from the held-out GoEmotions archive dataset (`data/goemotions/test.csv`) without uploading your own file.")
    cloud_archive_path = PROJECT_ROOT / "data" / "goemotions" / "test.csv"
    if not cloud_archive_path.is_file():
        cloud_archive_path = PROJECT_ROOT / "archive" / "data" / "test.tsv"
    if cloud_archive_path.is_file():
        col_arch_btn, col_arch_info = st.columns([1, 1.5])
        with col_arch_btn:
            if st.button("📥 Load 20 GoEmotions Archive Entries", key="load_archive_cloud_sample", use_container_width=True):
                try:
                    sep = "," if cloud_archive_path.suffix == ".csv" else "\t"
                    df_c = pd.read_csv(cloud_archive_path, sep=sep, nrows=20)
                    c_col = "text" if "text" in df_c.columns else df_c.columns[0]
                    items = [(str(row[c_col]).strip(), f"GoEmotions Archive · item {i}") for i, row in enumerate(df_c.to_dict(orient="records"), start=1) if str(row[c_col]).strip()]
                    _analyze_many(items)
                except Exception as ex:
                    st.error(f"Could not load archive sample: {ex}")
        with col_arch_info:
            st.write(f"Archive file: `{cloud_archive_path.name}` (5,427 held-out evaluation examples, 28 emotions)")
    else:
        st.info("GoEmotions archive splits can be generated locally via `python -m scripts.prepare_goemotions_csv`.")

with trends_tab:
    records = st.session_state.records
    if not records:
        st.info("Your check-ins will appear here for this browser session. Analyze a journal entry or import a file to begin.")
    else:
        frame = _history_frame(records)
        frame["Date"] = pd.to_datetime(frame["Date"])
        time_window = st.selectbox(
            "Trend period",
            ["All check-ins", "Last 7 days", "Last 30 days", "Last 90 days"],
            key="trend_period",
        )
        window_days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}.get(time_window)
        if window_days:
            start_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=window_days - 1)
            frame = frame[frame["Date"] >= start_date]
        if frame.empty:
            st.info(f"No check-ins found for {time_window.lower()}. Choose a wider period or add a new check-in.")
        else:
            frame = frame.sort_values("Date")
            st.markdown("### Your emotional pattern over time")
            st.caption("Text sentiment, predicted emotion, and your own ratings are shown as separate signals.")
            summary_columns = st.columns(4)
            summary_columns[0].metric("Check-ins", len(frame))
            summary_columns[1].metric("Most common signal", frame["Dominant emotion"].mode().iloc[0])
            summary_columns[2].metric(
                "Average text tone",
                f"{frame['VADER compound'].mean():+.2f}",
                help="VADER compound ranges from −1 (negative) to +1 (positive).",
            )
            mood_average = frame["Mood score"].dropna().mean()
            summary_columns[3].metric(
                "Your mood average",
                f"{mood_average:.1f}/5" if pd.notna(mood_average) else "Not added",
                help="Average of your optional self-ratings in the selected period.",
            )
            chart_col, count_col = st.columns([1.5, 1], gap="large")
            with chart_col:
                sentiment_figure = px.line(
                    frame,
                    x="Date",
                    y="VADER compound",
                    markers=True,
                    range_y=[-1, 1],
                    title="Text sentiment across check-ins",
                    labels={"VADER compound": "Text tone (−1 to +1)"},
                )
                sentiment_figure.update_traces(line={"color": "#155c49"}, marker={"size": 9})
                sentiment_figure.add_hline(y=0, line_dash="dot", line_color="#7b8b82")
                st.plotly_chart(sentiment_figure, width="stretch")
            with count_col:
                counts = frame["Dominant emotion"].value_counts().rename_axis("Emotion").reset_index(name="Check-ins")
                st.plotly_chart(
                    px.bar(counts, x="Emotion", y="Check-ins", color="Emotion", title="Detected emotion signals"),
                    width="stretch",
                )

            mood_frame = frame.dropna(subset=["Mood score"]).copy().sort_values("Date")
            if not mood_frame.empty:
                mood_col, _ = st.columns(2, gap="large")
                with mood_col:
                    st.plotly_chart(
                        px.line(
                            mood_frame,
                            x="Date",
                            y="Mood score",
                            markers=True,
                            range_y=[1, 5],
                            title="Your self-reported mood",
                            labels={"Mood score": "Mood (1–5)"},
                        ),
                        width="stretch",
                    )
                daily_mood = (
                    mood_frame.assign(Day=mood_frame["Date"].dt.date)
                    .groupby("Day", as_index=False)
                    .agg({"Mood score": "mean"})
                )
                daily_mood["Date"] = pd.to_datetime(daily_mood["Day"])
                first_day = daily_mood["Date"].min()
                first_week = first_day - pd.to_timedelta(first_day.weekday(), unit="D")
                week_start = daily_mood["Date"] - pd.to_timedelta(
                    daily_mood["Date"].dt.day_of_week, unit="D"
                )
                daily_mood["Week"] = ((week_start - first_week).dt.days // 7) + 1
                daily_mood["Weekday"] = daily_mood["Date"].dt.day_name()
                daily_mood["Day of month"] = daily_mood["Date"].dt.day
                calendar = px.scatter(
                    daily_mood,
                    x="Week",
                    y="Weekday",
                    color="Mood score",
                    color_continuous_scale="RdYlGn",
                    range_color=[1, 5],
                    hover_data={"Date": True, "Day of month": True, "Week": False, "Weekday": False},
                    title="Your mood in pixels",
                    category_orders={"Weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
                )
                calendar.update_traces(marker={"size": 18, "symbol": "square"})
                calendar.update_layout(height=300, xaxis_title="Calendar weeks", yaxis_title="")
                st.plotly_chart(calendar, width="stretch")
                st.caption("Each square is the average self-reported mood for that day. Blank days mean no check-in was logged.")
            else:
                st.info("Add an optional self-rated mood to build your personal mood calendar.")

            energy_frame = frame.dropna(subset=["Energy (1–5)"]).sort_values("Date")
            sleep_frame = frame.dropna(subset=["Sleep (hours)"]).sort_values("Date")
            if not energy_frame.empty or not sleep_frame.empty:
                energy_col, sleep_col = st.columns(2, gap="large")
                with energy_col:
                    if not energy_frame.empty:
                        st.plotly_chart(
                            px.line(
                                energy_frame,
                                x="Date",
                                y="Energy (1–5)",
                                markers=True,
                                range_y=[1, 5],
                                title="Your energy level",
                            ),
                            width="stretch",
                        )
                with sleep_col:
                    if not sleep_frame.empty:
                        st.plotly_chart(
                            px.line(
                                sleep_frame,
                                x="Date",
                                y="Sleep (hours)",
                                markers=True,
                                range_y=[0, 16],
                                title="Your sleep hours",
                            ),
                            width="stretch",
                        )
            st.dataframe(frame, width="stretch", hide_index=True)
            st.download_button(
                "Download filtered history (CSV)",
                frame.to_csv(index=False).encode("utf-8"),
                "moodmentor-session-trends.csv",
                "text/csv",
            )

with evaluation_tab:
    st.markdown("### GoEmotions model benchmark")
    st.caption(
        "The table below is an offline benchmark of BruceIC/emoticare on GoEmotions. "
        "When enabled, live transformer analysis uses SamLowe/roberta-base-go_emotions; "
        "the benchmark below is not a measured evaluation of that live model or of workplace journal text."
    )
    benchmark_path = PROJECT_ROOT / "docs" / "goemotions-bruceic-calibration.json"
    if not benchmark_path.is_file():
        st.warning("The saved benchmark report is not available in this deployment.")
    else:
        try:
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            st.write(
                f"**Model:** `{benchmark['model']}` · **Labels:** {benchmark['label_count']} "
                f"· **Test examples:** {benchmark['held_out_test_calibrated_metrics']['examples']:,}"
            )
            st.info(
                "Per-emotion thresholds were chosen using the validation split only. "
                "The held-out test split was kept separate for the final evaluation."
            )
            fixed = benchmark["held_out_test_fixed_threshold_metrics"]
            calibrated = benchmark["held_out_test_calibrated_metrics"]
            detail_setup = st.radio(
                "View detailed test scores",
                ["Validation-calibrated", "Fixed threshold (0.50)"],
                horizontal=True,
                key="benchmark_test_setup",
            )
            active = calibrated if detail_setup == "Validation-calibrated" else fixed
            st.markdown(f"#### {detail_setup} results")
            metric_groups = [
                ("Macro precision", "macro_precision", ".1%"),
                ("Macro recall", "macro_recall", ".1%"),
                ("Macro F1", "macro_f1", ".1%"),
                ("Exact-match accuracy", "exact_match_accuracy", ".1%"),
                ("Micro precision", "micro_precision", ".1%"),
                ("Micro recall", "micro_recall", ".1%"),
                ("Micro F1", "micro_f1", ".1%"),
                ("Hamming loss ↓", "hamming_loss", ".2%"),
            ]
            for start in (0, 4):
                metric_columns = st.columns(4)
                for column, (label, key, format_spec) in zip(
                    metric_columns, metric_groups[start : start + 4]
                ):
                    column.metric(label, format(active[key], format_spec))
            rows = [
                {
                    "Test setup": "Fixed threshold (0.50)",
                    "Macro F1": f"{fixed['macro_f1']:.1%}",
                    "Micro F1": f"{fixed['micro_f1']:.1%}",
                    "Exact-match accuracy": f"{fixed['exact_match_accuracy']:.1%}",
                    "Hamming loss ↓": f"{fixed['hamming_loss']:.2%}",
                },
                {
                    "Test setup": "Validation-calibrated thresholds",
                    "Macro F1": f"{calibrated['macro_f1']:.1%}",
                    "Micro F1": f"{calibrated['micro_f1']:.1%}",
                    "Exact-match accuracy": f"{calibrated['exact_match_accuracy']:.1%}",
                    "Hamming loss ↓": f"{calibrated['hamming_loss']:.2%}",
                },
            ]
            st.dataframe(rows, width="stretch", hide_index=True)
            chart_rows = [
                {"Metric": metric, "Score": scores[metric], "Threshold setup": setup}
                for setup, scores in [
                    ("Fixed 0.50", fixed),
                    ("Validation-calibrated", calibrated),
                ]
                for metric in ("macro_f1", "micro_f1", "exact_match_accuracy")
            ]
            for row in chart_rows:
                row["Metric"] = {
                    "macro_f1": "Macro F1",
                    "micro_f1": "Micro F1",
                    "exact_match_accuracy": "Exact-match accuracy",
                }[row["Metric"]]
            st.plotly_chart(
                px.bar(
                    chart_rows,
                    x="Metric",
                    y="Score",
                    color="Threshold setup",
                    barmode="group",
                    range_y=[0, 1],
                    title="Held-out test scores",
                ),
                width="stretch",
            )
            with st.expander("What do these evaluation metrics mean?"):
                st.markdown(
                    """
                    - **Precision:** Of the emotion labels the model predicted, how many were correct?
                    - **Recall:** Of the emotion labels present in the data, how many did the model find?
                    - **F1:** A balance between precision and recall. **Macro F1** gives each emotion equal weight; **micro F1** combines counts across emotions.
                    - **Exact-match accuracy:** The fraction of entries where all 28 label decisions matched exactly. This is intentionally strict for a multi-label task.
                    - **Hamming loss:** The fraction of individual label decisions that were wrong; lower is better.
                    """
                )
            st.caption(
                "The validation macro F1 at the fixed 0.50 threshold was 40.19%, compared with "
                "29.62% for the previous public baseline. The milestone target of 82% is not yet met. "
                "GoEmotions comes from Reddit and may not represent workplace journal language."
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            st.warning(f"Could not read the saved benchmark report: {exc}")

records = st.session_state.records
selected = next((item for item in records if item["id"] == st.session_state.selected_record_id), records[0] if records else None)
if selected:
    report: AnalysisReport = selected["report"]
    st.markdown("---")
    st.markdown("## 📊 Your mood analysis")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Sentiment tone", report.sentiment_label.value.title())
    metric2.metric("Dominant emotion", report.dominant_emotion.title() if report.emotion_scores else "No clear cue")
    metric3.metric("VADER compound", f"{report.sentiment_scores.compound:+.2f}", help="A polarity score from −1 (negative) to +1 (positive).")
    metric4.metric("Words reflected", len(report.original_text.split()))
    st.caption(f"Emotion estimator: {report.emotion_method.replace('_', ' ')}. Keyword-based fallback signals are not model confidence values.")
    if any(selected.get(key) for key in ("self_reported_mood", "energy", "sleep_hours", "context_tags")):
        with st.container(border=True):
            st.markdown("#### Your check-in details")
            detail_columns = st.columns(4)
            detail_columns[0].metric("Your mood", selected.get("self_reported_mood") or "Skipped")
            detail_columns[1].metric("Energy", f"{selected['energy']}/5" if selected.get("energy") is not None else "Not added")
            detail_columns[2].metric("Sleep", f"{selected['sleep_hours']} hrs" if selected.get("sleep_hours") is not None else "Not added")
            detail_columns[3].metric("Context tags", len(selected.get("context_tags", [])))
            if selected.get("context_tags"):
                st.caption(" · ".join(selected["context_tags"]))
            st.caption("These are your own check-in details; they are not inferred by the text analyzer.")

    result_col, emotion_col = st.columns([1, 1], gap="large")
    with result_col:
        st.markdown(f"### {report.recommendations.get('title', 'A way forward')}")
        st.write(report.recommendations.get("summary", "Choose one gentle next step."))
        for path in report.recommendations.get("action_paths", []):
            with st.expander(f"{path.get('title', 'A way forward')} · {path.get('duration', 'Flexible')}"):
                st.write(path.get("description", ""))
                for step in path.get("steps", []):
                    st.markdown(f"- {step}")
    with emotion_col:
        if report.emotion_scores:
            emotion_frame = [{"Emotion": key.title(), "Signal": value} for key, value in sorted(report.emotion_scores.items(), key=lambda pair: pair[1], reverse=True)]
            st.plotly_chart(px.bar(emotion_frame, x="Signal", y="Emotion", orientation="h", title="Emotion signal distribution", range_x=[0, 1]), width="stretch")
        else:
            st.info("No supported emotion cue matched this text. That does not mean the reflection is neutral.")

    download_col, details_col = st.columns([1, 2])
    with download_col:
        st.download_button("Download this check-in as PDF", _make_pdf(selected), f"moodmentor-check-in-{selected['created_at'].strftime('%Y%m%d-%H%M')}.pdf", "application/pdf", width="stretch")
    with details_col:
        with st.expander("See the language-processing details"):
            try:
                breakdown = get_preprocessing_breakdown(report.original_text)
                st.write("Cleaned text", breakdown["cleaned_text"])
                st.write("Tokenization", breakdown["tokenization_method"])
                st.write("Lemmatization", breakdown["lemmatization_method"])
                st.write("Processed tokens", ", ".join(breakdown["lemmas"]) or "No tokens")
            except (InputValidationError, NLTKResourceError) as exc:
                st.warning(str(exc))

st.markdown("---")
st.caption("MoodMentor is a learning and self-reflection tool, not a clinical assessment, diagnosis, or emergency service. If you may be in immediate danger, contact local emergency services or a trusted person now.")
