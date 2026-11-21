"""MoodMentor employee wellness text analysis and emotion telemetry API."""

from pathlib import Path
import os
import nltk

_local_nltk = Path(__file__).resolve().parent.parent / "nltk_data"
# Filter out unreadable AppData paths to prevent NLTK pathsec permission errors in restricted environments
_safe_paths = [str(_local_nltk)] + [
    p for p in nltk.data.path
    if "AppData" not in p and Path(p).is_dir()
]
nltk.data.path = _safe_paths

from .food_planner import get_mood_food_plan, get_supported_mood_categories
from .meditation import (
    get_meditation_packs,
    get_soundscape_catalog,
    render_medito_soundscape_widget,
)
from .pipeline import (
    EMOTIONS,
    analyze_csv_records,
    analyze_file,
    analyze_text,
    classify_emotions,
    generate_recommendations,
)
from .preprocessing import get_preprocessing_breakdown, sanitize_text_for_vader
from .report import AnalysisReport, SentimentLabel
from .storage import (
    authenticate_user,
    connect_database,
    delete_mood_entry,
    get_user_mood_entries,
    register_user,
    save_mood_entry,
)

__all__ = [
    "EMOTIONS",
    "AnalysisReport",
    "SentimentLabel",
    "analyze_csv_records",
    "analyze_file",
    "analyze_text",
    "authenticate_user",
    "classify_emotions",
    "connect_database",
    "delete_mood_entry",
    "generate_recommendations",
    "get_meditation_packs",
    "get_mood_food_plan",
    "get_preprocessing_breakdown",
    "get_soundscape_catalog",
    "get_supported_mood_categories",
    "get_user_mood_entries",
    "register_user",
    "render_medito_soundscape_widget",
    "sanitize_text_for_vader",
    "save_mood_entry",
]
