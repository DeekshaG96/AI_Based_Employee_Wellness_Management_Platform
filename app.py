"""Streamlit frontend for the Milestone 1 wellness analysis pipeline."""

import json
import sqlite3
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from wellness.errors import WellnessError
from wellness.pipeline import analyze_file, analyze_text
from wellness.storage import authenticate_user, connect_database, register_user


def _analyze_upload(uploaded_file, text_column: str | None):
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
    st.set_page_config(page_title="Employee Wellness Platform", page_icon=":seedling:")
    for key, value in (("page", "landing"), ("db", None), ("user", None)):
        st.session_state.setdefault(key, value)

    if st.session_state.page == "landing":
        _landing_page()
        return
    if st.session_state.page == "login":
        _login_page()
        return
    if st.session_state.page == "register":
        _registration_page()
        return
    _analysis_page()


def _landing_page() -> None:
    st.title("Employee Wellness Platform")
    st.subheader("Understand how your team is feeling.")
    st.write(
        "Securely preprocess employee feedback and turn it into clear, actionable "
        "sentiment insights."
    )
    left, right = st.columns(2)
    if left.button("Login", type="primary", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()
    if right.button("Create account", use_container_width=True):
        st.session_state.page = "register"
        st.rerun()


def _login_page() -> None:
    st.title("Welcome back")
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")
    if submitted:
        try:
            connection = connect_database()
            user = authenticate_user(connection, email, password)
            connection.close()
            if user is None:
                st.error("Invalid email or password.")
            else:
                st.session_state.user = user
                st.session_state.page = "analysis"
                st.rerun()
        except sqlite3.Error as exc:
            st.error(f"Database error: {exc}")
    if st.button("Need an account? Register"):
        st.session_state.page = "register"
        st.rerun()


def _registration_page() -> None:
    st.title("Create your account")
    with st.form("registration"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirmation = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register", type="primary")
    if submitted:
        if password != confirmation:
            st.error("Passwords do not match.")
        else:
            connection = connect_database()
            try:
                register_user(connection, name, email, password)
                st.success("Registration successful. You can now log in.")
                st.session_state.page = "login"
            except WellnessError as exc:
                st.error(str(exc))
            finally:
                connection.close()
    if st.button("Already have an account? Login"):
        st.session_state.page = "login"
        st.rerun()


def _analysis_page() -> None:
    st.title("Employee Wellness Analyzer")
    st.write(f"Signed in as **{st.session_state.user[1]}**.")
    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.db = None
        st.session_state.page = "landing"
        st.rerun()

    st.subheader("1. Connect database")
    if st.button("Connect Database", type="primary"):
        try:
            connection = connect_database()
            connection.close()
            st.session_state.db = True
            st.success("Database connected successfully.")
        except sqlite3.Error as exc:
            st.session_state.db = False
            st.error(f"Database connection failed: {exc}")
    if not st.session_state.db:
        st.info("Connect the database before submitting feedback.")
        return

    st.subheader("2. Submit feedback")
    input_mode = st.radio("Input type", ("Direct text", "Upload file"), horizontal=True)
    report = None
    if input_mode == "Direct text":
        text = st.text_area("Feedback text", height=180, placeholder="Enter employee feedback...")
        if st.button("Analyze text", type="primary"):
            try:
                report = analyze_text(text)
            except WellnessError as exc:
                st.error(str(exc))
    else:
        uploaded_file = st.file_uploader("Upload UTF-8 TXT or CSV", type=["txt", "csv"])
        text_column = st.text_input("CSV text column (optional)", placeholder="feedback")
        if st.button("Analyze file", type="primary"):
            if uploaded_file is None:
                st.error("Choose a .txt or .csv file before analyzing.")
            else:
                try:
                    report = _analyze_upload(uploaded_file, text_column or None)
                except WellnessError as exc:
                    st.error(str(exc))

    if report is not None:
        st.subheader("3. Sentiment analysis")
        st.subheader("Sentiment")
        score = report.sentiment_scores
        st.metric("Classification", report.sentiment_label.value.title())
        st.json(
            {
                "negative": score.negative,
                "neutral": score.neutral,
                "positive": score.positive,
                "compound": score.compound,
            }
        )
        st.subheader("Processed report")
        st.write(report.to_human_readable())
        st.download_button(
            "Download JSON report",
            data=json.dumps(report.to_dict(), indent=2),
            file_name="wellness-report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
