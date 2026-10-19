import csv
from pathlib import Path

import pytest

from wellness.errors import InputValidationError
from wellness.pipeline import (
    analyze_csv_records,
    analyze_file,
    analyze_text,
    classify_emotions,
    generate_recommendations,
)
from wellness.preprocessing import (
    clean_text,
    get_preprocessing_breakdown,
    preprocess,
    sanitize_text_for_vader,
)
from wellness.report import SentimentLabel


def test_clean_text_removes_urls_punctuation_and_numbers():
    assert clean_text("Great work! Visit https://example.com in 2024.") == "great work visit in"


def test_sanitize_for_vader_preserves_caps_and_punctuation():
    raw = "The product is FANTASTIC!!! Check https://example.com or email me at user@test.com <p>now</p>."
    sanitized = sanitize_text_for_vader(raw)
    assert "FANTASTIC" in sanitized
    assert "!!!" in sanitized
    assert "https://" not in sanitized
    assert "user@test.com" not in sanitized
    assert "<p>" not in sanitized


def test_contractions_expansion():
    sanitized = sanitize_text_for_vader("I don't think we can't make it.")
    assert "not" in sanitized


def test_preprocessing_filters_stopwords_but_preserves_negations():
    _, lemmas = preprocess("We are not happy with this slow process.")
    assert "not" in lemmas
    assert "happy" in lemmas
    assert "are" not in lemmas


def test_direct_text_positive_sentiment():
    report = analyze_text("I absolutely love this supportive and excellent workplace.")
    assert report.sentiment_label is SentimentLabel.POSITIVE
    assert report.sentiment_scores.compound > 0.05
    assert report.dominant_emotion in ("joy", "neutral", "surprise")
    assert report.to_dict()["sentiment_label"] == "positive"


def test_strict_vader_threshold_boundaries():
    # Boundary logic tests
    # > 0.05 is positive
    rep_pos = analyze_text("This is an outstanding and brilliant success.")
    assert rep_pos.sentiment_scores.compound > 0.05
    assert rep_pos.sentiment_label is SentimentLabel.POSITIVE

    # < -0.05 is negative
    rep_neg = analyze_text("This is terrible, disastrous, and painful.")
    assert rep_neg.sentiment_scores.compound < -0.05
    assert rep_neg.sentiment_label is SentimentLabel.NEGATIVE

    # Neutral between -0.05 and 0.05
    rep_neu = analyze_text("The meeting started at 10 AM in the room.")
    assert -0.05 <= rep_neu.sentiment_scores.compound <= 0.05
    assert rep_neu.sentiment_label is SentimentLabel.NEUTRAL


def test_fine_grained_emotion_classification():
    scores, dominant = classify_emotions("I feel deeply depressed and hopeless.")
    assert isinstance(scores, dict)
    assert "sadness" in scores
    assert dominant in ("sadness", "fear")


def test_recommendation_engine():
    recs = generate_recommendations(
        dominant_emotion="fear",
        sentiment_label=SentimentLabel.NEGATIVE,
        compound=-0.65,
        text="I am terrified and anxious about my job security.",
    )
    assert recs["category"] == "Stress Management"
    assert len(recs["action_items"]) > 0

    recs_pos = generate_recommendations(
        dominant_emotion="joy",
        sentiment_label=SentimentLabel.POSITIVE,
        compound=0.8,
        text="I am thrilled and proud of our win!",
    )
    assert recs_pos["category"] == "Thriving & Growth"


def test_crisis_screening():
    recs = generate_recommendations(
        dominant_emotion="sadness",
        sentiment_label=SentimentLabel.NEGATIVE,
        compound=-0.9,
        text="I want to die and kill myself.",
    )
    assert recs["is_crisis"] is True
    assert "988" in " ".join(recs["action_items"])


def test_txt_file_ingestion(tmp_path: Path):
    path = tmp_path / "feedback.txt"
    path.write_text("The team is calm and productive.", encoding="utf-8")
    report = analyze_file(path)
    assert report.input_summary.source_type == "txt"
    assert report.original_text.startswith("The team")


def test_csv_file_ingestion_with_text_column(tmp_path: Path):
    path = tmp_path / "feedback.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "feedback"])
        writer.writeheader()
        writer.writerow({"id": "1", "feedback": "Helpful manager"})
        writer.writerow({"id": "2", "feedback": "Great benefits"})
    report = analyze_file(path)
    assert report.input_summary.source_type == "csv"
    assert "Helpful manager" in report.original_text


def test_analyze_csv_records(tmp_path: Path):
    path = tmp_path / "multi_row.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "comment"])
        writer.writeheader()
        writer.writerow({"id": "1", "comment": "Great morning meeting"})
        writer.writerow({"id": "2", "comment": "Feeling fatigued and drained"})
    records = analyze_csv_records(path, "comment")
    assert len(records) == 2
    assert records[0].sentiment_label is SentimentLabel.POSITIVE
    assert records[1].sentiment_label is SentimentLabel.NEGATIVE


@pytest.mark.parametrize(
    "content, suffix",
    [("", ".txt"), ("text\n", ".csv")],
)
def test_empty_files_are_rejected(tmp_path: Path, content: str, suffix: str):
    path = tmp_path / f"empty{suffix}"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InputValidationError):
        analyze_file(path)


def test_invalid_text_and_file_are_rejected(tmp_path: Path):
    with pytest.raises(InputValidationError):
        analyze_text("  ")
    with pytest.raises(InputValidationError):
        analyze_file(tmp_path / "missing.txt")


def test_csv_without_text_column_is_rejected(tmp_path: Path):
    path = tmp_path / "invalid.csv"
    path.write_text("id,name\n1,Ada\n", encoding="utf-8")
    with pytest.raises(InputValidationError, match="text column"):
        analyze_file(path)


def test_preprocessing_breakdown():
    breakdown = get_preprocessing_breakdown("The employees were working happily.")
    assert "cleaned_text" in breakdown
    assert "vader_input" in breakdown
    assert "raw_tokens" in breakdown
    assert "filtered_tokens" in breakdown
    assert "lemmas" in breakdown
    assert "the" not in breakdown["filtered_tokens"]
