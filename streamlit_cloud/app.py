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

# This deployment is intentionally lightweight: no external model download and no DB/API.
os.environ["WELLNESS_DISABLE_TRANSFORMER"] = "1"
NLTK_DATA_DIR = Path(os.getenv("NLTK_DATA", "/tmp/moodmentor-nltk"))
os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from wellness.errors import InputValidationError, NLTKResourceError
from wellness.pipeline import analyze_text
from wellness.preprocessing import get_preprocessing_breakdown
from wellness.report import AnalysisReport

st.set_page_config(
    page_title="MoodMentor | Mindful Check-in",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#19352d; --muted:#657b71; --paper:#f4f7f3; --line:#e0e9e1; --forest:#155c49; --leaf:#c9ebad; }
    html,body,[class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp { background:radial-gradient(850px circle at 86% 3%,rgba(206,231,211,.48),transparent 48%),var(--paper); }
    .block-container { max-width:1320px; padding:2rem 2.5rem 4rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
    .brand { display:flex;align-items:center;gap:12px;margin:0 0 20px; }
    .mark { display:grid;place-items:center;width:46px;height:46px;border-radius:16px;background:var(--forest);color:white;font:800 25px Manrope,sans-serif; }
    .brand-name { font:800 1.3rem Manrope,sans-serif;color:var(--ink); }
    .brand-caption { color:var(--muted);font-size:.8rem; }
    .hero { padding:30px 34px;border-radius:24px;color:#f3f8f2;background:linear-gradient(128deg,#103e34,#155b48 65%,#26745a);box-shadow:0 18px 44px #1848391c;margin-bottom:18px; }
    .eyebrow { color:#a9d9bb;text-transform:uppercase;font-size:.7rem;font-weight:800;letter-spacing:.15em; }
    .hero h1 { color:white;margin:12px 0 6px;font-size:clamp(2rem,4vw,3.1rem); }
    .hero p { max-width:720px;color:#d1e4d7;line-height:1.7;margin-bottom:0; }
    [data-testid="stMetric"] { padding:15px 18px;border:1px solid var(--line);border-radius:16px;background:white; }
    [data-testid="stVerticalBlockBorderWrapper"] { border:1px solid var(--line)!important;border-radius:18px!important;background:#ffffffcc; }
    [data-testid="stTextArea"] textarea { min-height:190px;border-radius:14px;line-height:1.65; }
    .stButton>button,.stDownloadButton>button { min-height:43px;border-radius:12px;font-weight:700; }
    .stButton>button[kind="primary"] { background:var(--forest);border-color:var(--forest); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#103d33,#12382f); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#d0e1d5; }
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


@st.cache_resource(show_spinner="Preparing the language resources for the first check-in…")
def prepare_nltk() -> bool:
    import nltk

    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))
    packages = (
        "stopwords",
        "wordnet",
        "omw-1.4",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
        "vader_lexicon",
    )
    for package in packages:
        if not nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=True):
            raise RuntimeError(f"Could not download the required language resource: {package}")
    return True


def _new_record(
    text: str, source: str, checkin_details: dict[str, Any] | None = None
) -> dict[str, Any]:
    report = analyze_text(text, source_name=source)
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
    items: list[tuple[str, str]], checkin_details: dict[str, Any] | None = None
) -> None:
    try:
        prepare_nltk()
        with st.spinner(f"Reflecting on {len(items)} entr{'y' if len(items) == 1 else 'ies'}…"):
            records = [
                _new_record(
                    text,
                    source,
                    checkin_details if source == "Journal Entry" else None,
                )
                for text, source in items
                if text.strip()
            ]
        st.session_state.records = records + st.session_state.records
        if records:
            st.session_state.selected_record_id = records[0]["id"]
            st.success(f"Saved {len(records)} check-in{'s' if len(records) != 1 else ''} to this browser session.")
    except (InputValidationError, NLTKResourceError, RuntimeError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")


if "records" not in st.session_state:
    st.session_state.records = []
if "selected_record_id" not in st.session_state:
    st.session_state.selected_record_id = None

with st.sidebar:
    st.markdown('<div class="brand"><div class="mark">m</div><div><div class="brand-name" style="color:white">MoodMentor</div><div class="brand-caption" style="color:#b5d0c0">Your mindful space</div></div></div>', unsafe_allow_html=True)
    st.markdown("### This session")
    st.metric("Saved check-ins", len(st.session_state.records))
    st.caption("Your check-ins stay available for this session only. This hosted showcase has no account or permanent journal; the full local version supports saved history.")
    if st.button("Clear this session's history", use_container_width=True, disabled=not st.session_state.records):
        st.session_state.records = []
        st.session_state.selected_record_id = None
        st.rerun()
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
st.info("Your check-ins are processed by MoodMentor on Streamlit Cloud and kept only for this session. They are not saved to an account or permanent journal. Please avoid names or other details that identify you.", icon="ℹ️")

journal_tab, file_tab, trends_tab, evaluation_tab = st.tabs(["📝 Mood journal", "📄 Import a file", "📈 Trends & history", "🧪 Model evaluation"])
with journal_tab:
    col_editor, col_prompt = st.columns([2.1, 0.9], gap="large")
    with col_editor:
        st.markdown("### Start with how today feels")
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
        entry = st.text_area(
            "Your reflection",
            placeholder="What felt heavy, hopeful, or important today? A few sentences are enough to begin.",
            height=220,
            max_chars=6000,
            key="cloud_journal",
        )
        if st.button("Analyze my mood", type="primary", use_container_width=True, disabled=not entry.strip()):
            checkin_details = {
                "self_reported_mood": (
                    None if self_reported_mood == "Prefer not to say" else self_reported_mood
                ),
                "context_tags": context_tags,
                "energy": energy if include_wellbeing_details else None,
                "sleep_hours": sleep_hours if include_wellbeing_details else None,
            }
            _analyze_many([(entry.strip(), "Journal Entry")], checkin_details)
    with col_prompt:
        st.markdown("### A small reminder")
        st.write("There is no right way to write this. Describe a moment, a conversation, or simply how you feel right now.")
        st.caption("MoodMentor offers general reflection prompts. It cannot diagnose or replace professional support.")

with file_tab:
    st.markdown("### Bring in a text or CSV journal")
    st.caption("Accepted files: UTF-8 .txt or .csv up to 1 MB. CSV files need a text, feedback, comment, message, or content column. Up to 25 non-empty rows are analyzed per upload.")
    uploaded = st.file_uploader("Choose a journal file", type=["txt", "csv"], key="cloud_upload")
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
                    columns = [name.strip() for name in (reader.fieldnames or []) if name and name.strip()]
                    if not columns:
                        st.error("The CSV needs a header row.")
                        file_items = []
                    else:
                        preferred = next((c for c in columns if c.lower() in {"text", "feedback", "comment", "message", "content"}), columns[0])
                        text_column = st.selectbox("Text column", columns, index=columns.index(preferred), key="cloud_csv_column")
                        file_items = [(str(row.get(text_column, "")).strip(), f"{uploaded.name} · row {index}") for index, row in enumerate(reader, start=2) if str(row.get(text_column, "")).strip()][:25]
                if file_items:
                    st.caption(f"Ready to analyze {len(file_items)} entr{'y' if len(file_items) == 1 else 'ies'}.")
                    if st.button("Analyze file entries", type="primary", key="analyze_cloud_file"):
                        _analyze_many(file_items)
                elif uploaded.name.lower().endswith(".txt"):
                    st.warning("The text file is empty.")
            except UnicodeDecodeError:
                st.error("The file must use UTF-8 text encoding.")
            except csv.Error as exc:
                st.error(f"Could not read this CSV: {exc}")

with trends_tab:
    records = st.session_state.records
    if not records:
        st.info("Your check-ins will appear here for this browser session. Analyze a journal entry or import a file to begin.")
    else:
        frame = _history_frame(records)
        st.markdown("### Your emotional pattern over time")
        chart_col, count_col = st.columns([1.5, 1], gap="large")
        with chart_col:
            st.plotly_chart(
                px.line(frame.iloc[::-1], x="Date", y="VADER compound", color="Sentiment", markers=True, range_y=[-1, 1], title="Sentiment tone across check-ins"),
                use_container_width=True,
            )
        with count_col:
            counts = frame["Dominant emotion"].value_counts().rename_axis("Emotion").reset_index(name="Check-ins")
            st.plotly_chart(px.bar(counts, x="Emotion", y="Check-ins", color="Emotion", title="Dominant emotion signals"), use_container_width=True)
        mood_frame = frame.dropna(subset=["Mood score"]).copy()
        if not mood_frame.empty:
            mood_frame["Date"] = pd.to_datetime(mood_frame["Date"])
            mood_frame = mood_frame.sort_values("Date")
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
                use_container_width=True,
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
                daily_mood["Date"].dt.weekday, unit="D"
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
            calendar.update_layout(height=300, xaxis_title="Weeks since first check-in", yaxis_title="")
            st.plotly_chart(calendar, use_container_width=True)
            st.caption("Each square is the average self-reported mood for that day. Blank days mean no check-in was logged.")
        else:
            st.info("Add an optional self-rated mood to journal check-ins to build your personal mood calendar.")
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button("Download session trends (CSV)", frame.to_csv(index=False).encode("utf-8"), "moodmentor-session-trends.csv", "text/csv")

with evaluation_tab:
    st.markdown("### GoEmotions model benchmark")
    st.caption(
        "Offline evaluation of a public 28-label emotion model against the GoEmotions dataset. "
        "These benchmark scores are separate from this hosted showcase's lightweight keyword estimator."
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
            st.dataframe(rows, use_container_width=True, hide_index=True)
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
                use_container_width=True,
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
            st.plotly_chart(px.bar(emotion_frame, x="Signal", y="Emotion", orientation="h", title="Emotion signal distribution", range_x=[0, 1]), use_container_width=True)
        else:
            st.info("No supported emotion cue matched this text. That does not mean the reflection is neutral.")

    download_col, details_col = st.columns([1, 2])
    with download_col:
        st.download_button("Download this check-in as PDF", _make_pdf(selected), f"moodmentor-check-in-{selected['created_at'].strftime('%Y%m%d-%H%M')}.pdf", "application/pdf", use_container_width=True)
    with details_col:
        with st.expander("See the language-processing details"):
            try:
                breakdown = get_preprocessing_breakdown(report.original_text)
                st.write("Cleaned text", breakdown["cleaned_text"])
                st.write("Lemmatized tokens", ", ".join(breakdown["lemmas"]) or "No tokens")
            except (InputValidationError, NLTKResourceError) as exc:
                st.warning(str(exc))

st.markdown("---")
st.caption("MoodMentor is a learning and self-reflection tool, not a clinical assessment, diagnosis, or emergency service. If you may be in immediate danger, contact local emergency services or a trusted person now.")