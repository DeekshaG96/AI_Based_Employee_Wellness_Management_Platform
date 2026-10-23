"""Small authenticated HTTP client used by the Streamlit frontend."""

from __future__ import annotations

import os
from typing import Any

import requests

from .report import AnalysisReport, InputSummary, SentimentLabel, SentimentScores


class ApiError(RuntimeError):
    """An API request failed with a user-readable explanation."""


def api_url() -> str:
    try:
        import streamlit as st

        configured = st.secrets.get("API_URL")
    except Exception:
        configured = None
    return str(configured or os.getenv("API_URL", "http://127.0.0.1:8000")).rstrip("/")


def _request(
    method: str,
    path: str,
    token: str | None = None,
    **kwargs: Any,
) -> Any:
    headers = dict(kwargs.pop("headers", {}))
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.request(
            method,
            f"{api_url()}{path}",
            headers=headers,
            timeout=(5, 180),
            **kwargs,
        )
    except requests.RequestException as exc:
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


def register(name: str, email: str, password: str) -> dict[str, Any]:
    return _request(
        "POST", "/auth/register", json={"name": name, "email": email, "password": password}
    )


def login(email: str, password: str) -> dict[str, Any]:
    return _request("POST", "/auth/login", json={"email": email, "password": password})


def logout(token: str) -> None:
    _request("POST", "/auth/logout", token=token)


def analyze(text: str, token: str, source_name: str = "Direct Entry") -> AnalysisReport:
    return analysis_report_from_dict(
        _request("POST", "/analyze", token=token, json={"text": text, "source_name": source_name})
    )


def analyze_file(
    content: bytes, filename: str, token: str, text_column: str | None = None
) -> list[AnalysisReport]:
    params = {"text_column": text_column} if text_column else None
    result = _request(
        "POST",
        "/analyze/file",
        token=token,
        params=params,
        files={"file": (filename, content)},
    )
    return [analysis_report_from_dict(item) for item in result.get("items", [])]


def history(user_id: int, token: str, limit: int = 500) -> list[dict[str, Any]]:
    response = _request("GET", f"/history/{user_id}", token=token, params={"limit": limit})
    entries = response.get("items", [])
    for entry in entries:
        entry["tokens"] = ", ".join(entry.get("tokens") or [])
        entry["recommendation"] = entry.get("recommendation") or {}
    return entries


def delete_entry(entry_id: int, token: str) -> None:
    _request("DELETE", f"/history/{entry_id}", token=token)


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
