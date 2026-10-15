import csv
from types import SimpleNamespace

import pytest

from app import _analyze_upload
from wellness.errors import InputValidationError
from wellness.pipeline import analyze_file, analyze_text
from wellness.preprocessing import clean_text
from wellness.report import SentimentLabel


def test_clean_text_removes_urls_punctuation_and_numbers():
    assert clean_text("Great work! Visit https://example.com in 2024.") == "great work visit in"


def test_direct_text_positive_sentiment():
    report = analyze_text("I absolutely love this supportive and excellent workplace.")
    assert report.sentiment_label is SentimentLabel.POSITIVE
    assert report.sentiment_scores.compound > 0
    assert "supportive" in report.tokens
    assert report.to_dict()["sentiment_label"] == "positive"


def test_preprocessing_filters_stopwords_and_lemmatizes():
    report = analyze_text("The employees were running and happily helping teams.")
    assert "the" not in report.tokens
    assert "run" in report.tokens
    assert report.processed_text


def test_txt_file_ingestion(tmp_path):
    path = tmp_path / "feedback.txt"
    path.write_text("The team is calm and productive.", encoding="utf-8")
    report = analyze_file(path)
    assert report.input_summary.source_type == "txt"
    assert report.original_text.startswith("The team")


def test_csv_file_ingestion_with_text_column(tmp_path):
    path = tmp_path / "feedback.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "feedback"])
        writer.writeheader()
        writer.writerow({"id": "1", "feedback": "Helpful manager"})
        writer.writerow({"id": "2", "feedback": "Great benefits"})
    report = analyze_file(path)
    assert report.input_summary.source_type == "csv"
    assert "Helpful manager" in report.original_text


@pytest.mark.parametrize(
    "content, suffix",
    [("", ".txt"), ("text\n", ".csv")],
)
def test_empty_files_are_rejected(tmp_path, content, suffix):
    path = tmp_path / f"empty{suffix}"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InputValidationError):
        analyze_file(path)


def test_invalid_text_and_file_are_rejected(tmp_path):
    with pytest.raises(InputValidationError):
        analyze_text("  ")
    with pytest.raises(InputValidationError):
        analyze_file(tmp_path / "missing.txt")


def test_csv_without_text_column_is_rejected(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("id,name\n1,Ada\n", encoding="utf-8")
    with pytest.raises(InputValidationError, match="text column"):
        analyze_file(path)


def test_streamlit_upload_adapter_uses_backend_pipeline():
    uploaded = SimpleNamespace(name="feedback.txt", getvalue=lambda: b"Helpful team")
    report = _analyze_upload(uploaded, None)
    assert report.input_summary.source_type == "txt"
    assert report.sentiment_scores.compound > 0
