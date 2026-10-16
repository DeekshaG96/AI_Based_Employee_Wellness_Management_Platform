"""Streamlit frontend for the MoodMentor Employee Wellness Management Platform.

Infosys Springboard Virtual Internship - Milestone 1:
Text Ingestion, Preprocessing & Baseline Sentiment Analysis.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import streamlit as st

from wellness.errors import WellnessError
from wellness.pipeline import analyze_file, analyze_text
from wellness.preprocessing import get_preprocessing_breakdown
from wellness.storage import (
    authenticate_user,
    connect_database,
    delete_mood_entry,
    get_user_mood_entries,
    register_user,
    save_mood_entry,
)

CUSTOM_CSS = """
<style>
/* Modern Glassmorphic Dark Theme Styling */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.sentiment-positive {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 12px;
    padding: 18px 24px;
    color: #34d399;
}
.sentiment-neutral {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.1));
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 12px;
    padding: 18px 24px;
    color: #fbbf24;
}
.sentiment-negative {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 12px;
    padding: 18px 24px;
    color: #f87171;
}
.token-badge {
    display: inline-block;
    background: rgba(59, 130, 246, 0.18);
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.35);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 3px 4px;
    letter-spacing: 0.02em;
}
.pipeline-step {
    background: rgba(255, 255, 255, 0.03);
    border-left: 3px solid #3b82f6;
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.92rem;
}
</style>
"""


def _analyze_upload(uploaded_file: Any, text_column: str | None):
    """Uploaded file helper maintaining backward compatibility with tests."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".txt", ".csv"}:
        raise WellnessError("Upload a .txt or .csv file.")
    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(uploaded_file.getvalue())
    try:
        return analyze_file(temporary_path, text_column if suffix == ".csv" else None)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    st.set_page_config(
        page_title="MoodMentor | Employee Wellness Platform",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    for key, value in (("page", "landing"), ("db", False), ("user", None)):
        st.session_state.setdefault(key, value)

    if st.session_state.page == "landing":
        _landing_page()
    elif st.session_state.page == "login":
        _login_page()
    elif st.session_state.page == "register":
        _registration_page()
    else:
        _analysis_page()


def _landing_page() -> None:
    st.title("🌱 MoodMentor: Employee Wellness Platform")
    st.caption("Infosys Springboard Virtual Internship • Milestone 1 (Batch-4)")

    st.markdown(
        """
        <div class="metric-card">
            <h3>Personalized Well-Being Through AI-Powered Emotional Analysis</h3>
            <p style="color: #94a3b8; font-size: 1.05rem;">
                Written communication contains deep emotional signals. <b>MoodMentor</b> provides
                an intelligent Natural Language Processing pipeline that ingests employee journal entries,
                messages, and feedback, applies rigorous NLTK text preprocessing, computes baseline
                VADER sentiment intensity, and logs persistent wellness records.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h4>📥 Ingestion & Preprocessing</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Accepts direct text, UTF-8 <code>.txt</code> notes, and multi-row <code>.csv</code> feedback.
                    Cleans URLs, symbols, tokenizes, removes stop-words, and performs WordNet lemmatization.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h4>🧠 VADER Sentiment Engine</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Calculates compound sentiment (-1.0 to +1.0) alongside positive, neutral,
                    and negative distributions with intensity calibration.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h4>💾 Persistence & Analytics</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Stores all user sessions and parsed mood entries in an SQLite relational database
                    with salted PBKDF2 authentication and history inspection.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    left, right = st.columns(2)
    if left.button("🔐 Sign In", type="primary", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()
    if right.button("📝 Create New Account", use_container_width=True):
        st.session_state.page = "register"
        st.rerun()


def _login_page() -> None:
    st.title("🔐 Sign In to MoodMentor")
    st.caption("Access the employee wellness analysis workspace")

    with st.form("login_form"):
        email = st.text_input("Work Email", placeholder="user@company.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

    if submitted:
        try:
            connection = connect_database()
            user = authenticate_user(connection, email, password)
            connection.close()
            if user is None:
                st.error("Invalid email or password.")
            else:
                st.session_state.user = user
                st.session_state.db = True
                st.session_state.page = "analysis"
                st.rerun()
        except sqlite3.Error as exc:
            st.error(f"Database error: {exc}")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Welcome", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()
    with col2:
        if st.button("Need an account? Register →", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()


def _registration_page() -> None:
    st.title("📝 Create Your Account")
    st.caption("Register for the AI-Powered Employee Wellness Platform")

    with st.form("registration_form"):
        name = st.text_input("Full Name", placeholder="Jane Doe")
        email = st.text_input("Work Email", placeholder="jane@company.com")
        password = st.text_input(
            "Password (min 8 characters)", type="password", placeholder="••••••••"
        )
        confirmation = st.text_input(
            "Confirm Password", type="password", placeholder="••••••••"
        )
        submitted = st.form_submit_button("Register Account", type="primary", use_container_width=True)

    if submitted:
        if password != confirmation:
            st.error("Passwords do not match.")
        else:
            connection = connect_database()
            try:
                register_user(connection, name, email, password)
                st.success("Registration successful! You can now sign in.")
                st.session_state.page = "login"
                st.rerun()
            except WellnessError as exc:
                st.error(str(exc))
            finally:
                connection.close()

    if st.button("Already have an account? Sign In", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()


def _render_sentiment_card(report: Any) -> None:
    scores = report.sentiment_scores
    label = report.sentiment_label.value
    css_class = (
        "sentiment-positive"
        if label == "positive"
        else "sentiment-negative"
        if label == "negative"
        else "sentiment-neutral"
    )
    emoji = "🌟" if label == "positive" else "⚠️" if label == "negative" else "⚖️"

    st.markdown(
        f"""
        <div class="{css_class}">
            <h3 style="margin: 0; padding: 0;">{emoji} Detected Sentiment: <b>{label.upper()}</b></h3>
            <p style="margin-top: 6px; margin-bottom: 0; font-size: 1.05rem;">
                Compound Polarity: <b>{scores.compound:+.3f}</b> &nbsp;|&nbsp;
                Confidence: <b>{abs(scores.compound) * 100:.1f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Compound Score", f"{scores.compound:+.3f}")
    col2.metric("Positive Polarity", f"{scores.positive * 100:.1f}%")
    col3.metric("Neutral Polarity", f"{scores.neutral * 100:.1f}%")
    col4.metric("Negative Polarity", f"{scores.negative * 100:.1f}%")


def _render_preprocessing_breakdown(text: str, report: Any) -> None:
    st.subheader("🔬 Text Preprocessing Pipeline (M1)")
    breakdown = get_preprocessing_breakdown(text)

    with st.expander("🔍 View Step-by-Step NLP Preprocessing Stages", expanded=True):
        st.markdown(
            f"""
            <div class="pipeline-step">
                <b>Step 1: Raw Ingested Text</b><br/>
                <span style="color: #cbd5e1;">{breakdown['raw_text']}</span>
            </div>
            <div class="pipeline-step">
                <b>Step 2: Cleaned Text (Lowercased, URLs & Symbols Stripped)</b><br/>
                <span style="color: #93c5fd;">{breakdown['cleaned_text']}</span>
            </div>
            <div class="pipeline-step">
                <b>Step 3: Tokenization & Stop-word Filtering</b><br/>
                <span style="color: #a7f3d0;">
                    Raw Tokens ({len(breakdown['raw_tokens'])}) → Filtered Words ({len(breakdown['filtered_tokens'])})
                </span>
            </div>
            <div class="pipeline-step">
                <b>Step 4: WordNet Lemmatization & POS Tagging (Final Keywords)</b><br/>
            </div>
            """,
            unsafe_allow_html=True,
        )

        badge_html = "".join(f'<span class="token-badge">{token}</span>' for token in report.tokens)
        st.markdown(badge_html or "<em>No alphabetic tokens extracted.</em>", unsafe_allow_html=True)


def _analysis_page() -> None:
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user[1]}")
        st.caption("Active Session")

        st.divider()
        st.markdown("#### ⚙️ Database Status")
        if not st.session_state.db:
            if st.button("🔌 Connect SQLite Database", type="primary", use_container_width=True):
                try:
                    conn = connect_database()
                    conn.close()
                    st.session_state.db = True
                    st.success("Connected to wellness.db")
                    st.rerun()
                except sqlite3.Error as e:
                    st.error(f"Error: {e}")
        else:
            st.success("🟢 SQLite Connected (`wellness.db`)")

        st.divider()
        st.markdown("#### 📌 Milestone 1 Quick Info")
        st.markdown(
            "- **Pipeline:** NLTK + WordNet\n"
            "- **Classifier:** VADER Lexicon\n"
            "- **Persistence:** SQLite Table `mood_entries`\n"
            "- **Formats:** Direct Text, `.txt`, `.csv`"
        )

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.db = False
            st.session_state.page = "landing"
            st.rerun()

    # Main Heading
    st.title("🌱 MoodMentor: Emotional Tone & Wellness Analyzer")
    st.caption("Infosys Springboard Virtual Internship • Milestone 1")

    if not st.session_state.db:
        st.warning("⚠️ Please connect to the database in the sidebar before analyzing feedback.")
        return

    tab1, tab2, tab3 = st.tabs([
        "📝 Direct Feedback Analysis",
        "📁 File Ingestion (.txt / .csv)",
        "📊 Stored Records & Analytics",
    ])

    user_id = st.session_state.user[0]

    # TAB 1: DIRECT TEXT
    with tab1:
        st.subheader("1. Ingest Direct Text Feedback")
        st.write("Enter an employee journal entry, 1-on-1 meeting note, chat snippet, or pulse feedback:")

        # Quick preset buttons for convenience
        st.markdown("###### Quick Test Samples:")
        c1, c2, c3 = st.columns(3)
        sample_positive = "I feel very supported and valued by my team. The project milestone went smoothly and leadership is encouraging!"
        sample_negative = "The workload is completely overwhelming and exhausting. We have impossible deadlines and no work-life balance."
        sample_neutral = "We held our daily sprint sync at 10 AM today. The team reviewed the pull requests and updated the milestone status."

        if "selected_sample" not in st.session_state:
            st.session_state.selected_sample = sample_positive

        if c1.button("🌟 Supportive (Positive)", use_container_width=True):
            st.session_state.selected_sample = sample_positive
            st.rerun()
        if c2.button("⚠️ Exhausted (Negative)", use_container_width=True):
            st.session_state.selected_sample = sample_negative
            st.rerun()
        if c3.button("⚖️ Sprint Sync (Neutral)", use_container_width=True):
            st.session_state.selected_sample = sample_neutral
            st.rerun()

        feedback_input = st.text_area(
            "Feedback Content",
            value=st.session_state.selected_sample,
            height=140,
            placeholder="Type or paste employee feedback here...",
        )

        if st.button("⚡ Run Sentiment Analysis", type="primary", key="btn_direct_analyze"):
            try:
                with st.spinner("Executing NLP Preprocessing & VADER Scoring..."):
                    report = analyze_text(feedback_input, source_name="Direct Input")

                    # Persist to SQLite
                    conn = connect_database()
                    entry_id = save_mood_entry(
                        conn,
                        user_id=user_id,
                        source_type="text",
                        source_name="Direct Input",
                        original_text=report.original_text,
                        cleaned_text=report.cleaned_text,
                        tokens=report.tokens,
                        sentiment_label=report.sentiment_label.value,
                        neg_score=report.sentiment_scores.negative,
                        neu_score=report.sentiment_scores.neutral,
                        pos_score=report.sentiment_scores.positive,
                        compound_score=report.sentiment_scores.compound,
                    )
                    conn.close()

                st.success(f"✅ Feedback analyzed and saved to database (Entry #{entry_id})!")
                st.divider()

                _render_sentiment_card(report)
                st.write("")
                _render_preprocessing_breakdown(feedback_input, report)

                st.write("")
                col_dl, col_raw = st.columns([1, 1])
                with col_dl:
                    st.download_button(
                        "📥 Download JSON Analysis Report",
                        data=json.dumps(report.to_dict(), indent=2),
                        file_name=f"moodmentor-report-entry-{entry_id}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with col_raw, st.expander("📄 View Human-Readable Report Text"):
                    st.text(report.to_human_readable())

            except WellnessError as exc:
                st.error(f"Validation Error: {exc}")
            except (sqlite3.Error, OSError) as exc:
                st.error(f"Analysis Error: {exc}")

    # TAB 2: FILE INGESTION
    with tab2:
        st.subheader("2. File Ingestion & Workplace Dataset")
        st.write("Upload feedback files (.txt / .csv) or test with the curated workplace wellness benchmark dataset:")

        # Quick Benchmark Dataset Loader
        benchmark_path = Path("data/employee_wellness_dataset.csv")
        if benchmark_path.exists():
            st.markdown("###### Curated Workplace Dataset (Kaggle Benchmark Standard):")
            col_b1, col_b2 = st.columns([2, 1])
            with col_b1:
                st.caption("60 realistic employee pulse survey entries across Engineering, Marketing, HR, Product, and Sales.")
            with col_b2:
                load_bench = st.button("📊 Load Benchmark Dataset", use_container_width=True)
                if load_bench:
                    st.session_state.active_file_mode = "benchmark"
                    st.rerun()

        st.divider()
        st.markdown("###### Upload Custom File:")
        uploaded = st.file_uploader(
            "Upload file (.txt or .csv)",
            type=["txt", "csv"],
            help="UTF-8 encoded text files or CSV files containing feedback columns",
        )

        custom_col = st.text_input(
            "CSV Text Column Name (Optional override)",
            placeholder="feedback (default auto-detects: text, feedback, comment, message, content)",
        )

        # Check active file source: either uploaded or benchmark
        active_source = None
        if uploaded is not None:
            active_source = ("uploaded", uploaded, uploaded.name)
            st.info(f"📄 Selected uploaded file: **{uploaded.name}** ({uploaded.size} bytes)")
            if uploaded.name.lower().endswith(".csv"):
                try:
                    df_preview = pd.read_csv(uploaded, nrows=5)
                    st.markdown("###### Preview of Uploaded CSV (First 5 rows):")
                    st.dataframe(df_preview, use_container_width=True)
                    uploaded.seek(0)
                except (pd.errors.ParserError, ValueError, UnicodeDecodeError):
                    st.warning("Could not generate tabular preview for this CSV.")
        elif st.session_state.get("active_file_mode") == "benchmark" and benchmark_path.exists():
            active_source = ("benchmark", benchmark_path, "employee_wellness_dataset.csv")
            st.success("🟢 Loaded: `data/employee_wellness_dataset.csv` (60 Workplace Reviews)")
            try:
                df_preview = pd.read_csv(benchmark_path)
                st.dataframe(df_preview.head(10), use_container_width=True)
            except (pd.errors.ParserError, ValueError, OSError):
                st.warning("Could not generate tabular preview for this dataset.")



        if st.button("🚀 Analyze File Content", type="primary", key="btn_file_analyze"):
            if active_source is None:
                st.error("Please upload a file or click 'Load Benchmark Dataset' above.")
            else:
                try:
                    with st.spinner("Processing file, cleaning text, and computing sentiment..."):
                        src_type, file_obj, display_name = active_source
                        if src_type == "uploaded":
                            report = _analyze_upload(file_obj, custom_col or None)
                        else:
                            report = analyze_file(file_obj, text_column="feedback")

                        # Persist to SQLite
                        conn = connect_database()
                        entry_id = save_mood_entry(
                            conn,
                            user_id=user_id,
                            source_type=report.input_summary.source_type,
                            source_name=display_name,
                            original_text=report.original_text,
                            cleaned_text=report.cleaned_text,
                            tokens=report.tokens,
                            sentiment_label=report.sentiment_label.value,
                            neg_score=report.sentiment_scores.negative,
                            neu_score=report.sentiment_scores.neutral,
                            pos_score=report.sentiment_scores.positive,
                            compound_score=report.sentiment_scores.compound,
                        )
                        conn.close()

                    st.success(f"✅ Analyzed and recorded in SQLite database (Entry #{entry_id})!")
                    st.divider()

                    _render_sentiment_card(report)
                    st.write("")
                    _render_preprocessing_breakdown(report.original_text, report)

                    st.download_button(
                        "📥 Download JSON Analysis Report",
                        data=json.dumps(report.to_dict(), indent=2),
                        file_name=f"moodmentor-file-entry-{entry_id}.json",
                        mime="application/json",
                    )
                except WellnessError as exc:
                    st.error(f"File Ingestion Error: {exc}")
                except (sqlite3.Error, OSError) as exc:
                    st.error(f"Processing Error: {exc}")

    # TAB 3: STORED RECORDS & DATABASE VIEW
    with tab3:
        st.subheader("3. Stored Mood Entries & Sentiment History")
        st.write("All analyzed feedback records saved persistently in the SQLite database:")

        conn = connect_database()
        entries = get_user_mood_entries(conn, user_id=user_id)
        conn.close()

        if not entries:
            st.info("ℹ️ No mood entries stored yet. Use Tab 1 or Tab 2 to analyze feedback!")
        else:
            df = pd.DataFrame(entries)

            # Analytics KPIs
            total_records = len(df)
            pos_count = len(df[df["sentiment_label"] == "positive"])
            neu_count = len(df[df["sentiment_label"] == "neutral"])
            neg_count = len(df[df["sentiment_label"] == "negative"])
            avg_compound = df["compound_score"].mean()

            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Total Records", total_records)
            kpi2.metric("Positive Entries", f"{pos_count} ({(pos_count/total_records)*100:.1f}%)")
            kpi3.metric("Neutral Entries", f"{neu_count} ({(neu_count/total_records)*100:.1f}%)")
            kpi4.metric("Negative Entries", f"{neg_count} ({(neg_count/total_records)*100:.1f}%)")
            kpi5.metric("Avg Compound", f"{avg_compound:+.2f}")

            st.divider()

            # Filtering options
            filter_col1, filter_col2 = st.columns([1, 2])
            with filter_col1:
                filter_sentiment = st.selectbox(
                    "Filter by Sentiment",
                    options=["All", "positive", "neutral", "negative"],
                )
            with filter_col2:
                search_query = st.text_input("Search Feedback Text or Keywords", placeholder="Filter records...")

            filtered_df = df.copy()
            if filter_sentiment != "All":
                filtered_df = filtered_df[filtered_df["sentiment_label"] == filter_sentiment]
            if search_query.strip():
                query = search_query.lower()
                filtered_df = filtered_df[
                    filtered_df["original_text"].str.lower().str.contains(query)
                    | filtered_df["tokens"].str.lower().str.contains(query)
                ]

            st.markdown(f"**Showing {len(filtered_df)} of {total_records} records:**")

            # Clean display table
            display_columns = [
                "id",
                "created_at",
                "source_type",
                "source_name",
                "sentiment_label",
                "compound_score",
                "tokens",
                "original_text",
            ]
            st.dataframe(
                filtered_df[display_columns],
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "created_at": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                    "source_type": st.column_config.TextColumn("Source", width="small"),
                    "source_name": st.column_config.TextColumn("Filename/Source"),
                    "sentiment_label": st.column_config.TextColumn("Sentiment"),
                    "compound_score": st.column_config.NumberColumn("Compound", format="%.3f"),
                    "tokens": st.column_config.TextColumn("Lemmatized Keywords"),
                    "original_text": st.column_config.TextColumn("Original Excerpt", width="large"),
                },
                use_container_width=True,
                hide_index=True,
            )

            # Record Detail Viewer & Delete
            st.write("---")
            st.markdown("##### Inspect Record Details")
            selected_id = st.selectbox(
                "Select Record ID to Inspect:",
                options=filtered_df["id"].tolist(),
                format_func=lambda x: f"Record #{x} - {df.loc[df['id'] == x, 'sentiment_label'].values[0].upper()} ({df.loc[df['id'] == x, 'created_at'].values[0]})",
            )

            if selected_id:
                record = df[df["id"] == selected_id].iloc[0]
                rec_col1, rec_col2 = st.columns([3, 1])
                with rec_col1:
                    st.write(f"**Original Text:** {record['original_text']}")
                    st.write(f"**Cleaned Text:** {record['cleaned_text']}")
                    st.write(f"**Lemmatized Tokens:** `{record['tokens']}`")
                    st.write(
                        f"**Scores:** Negative: `{record['neg_score']:.3f}` | "
                        f"Neutral: `{record['neu_score']:.3f}` | "
                        f"Positive: `{record['pos_score']:.3f}` | "
                        f"Compound: `{record['compound_score']:.3f}`"
                    )
                with rec_col2:
                    if st.button("🗑️ Delete Record", key=f"del_{selected_id}", type="secondary", use_container_width=True):
                        conn = connect_database()
                        delete_mood_entry(conn, selected_id, user_id)
                        conn.close()
                        st.success(f"Record #{selected_id} deleted.")
                        st.rerun()

            # CSV download of records
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Filtered History to CSV",
                data=csv_data,
                file_name="moodmentor_sentiment_history.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
