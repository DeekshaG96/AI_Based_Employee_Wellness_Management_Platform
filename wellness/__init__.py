"""MoodMentor employee wellness text analysis and emotion telemetry API."""

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
