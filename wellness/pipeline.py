"""Public analysis API combining ingestion, preprocessing, and VADER."""

from pathlib import Path

from nltk.sentiment import SentimentIntensityAnalyzer

from .ingestion import read_input_file, validate_text
from .preprocessing import preprocess, validate_nltk_resources
from .report import AnalysisReport, InputSummary, SentimentLabel, SentimentScores


def analyze_text(text: str, source_name: str = "direct text") -> AnalysisReport:
    """Analyze direct text and return a typed report."""
    original = validate_text(text)
    validate_nltk_resources()
    cleaned, tokens = preprocess(original)
    scores = SentimentIntensityAnalyzer().polarity_scores(original)
    compound = scores["compound"]
    label = (
        SentimentLabel.POSITIVE
        if compound >= 0.05
        else SentimentLabel.NEGATIVE
        if compound <= -0.05
        else SentimentLabel.NEUTRAL
    )
    lines = len(original.splitlines())
    return AnalysisReport(
        input_summary=InputSummary("text", source_name, len(original), lines, len(tokens)),
        original_text=original,
        cleaned_text=cleaned,
        processed_text=" ".join(tokens),
        tokens=tokens,
        sentiment_scores=SentimentScores(
            scores["neg"], scores["neu"], scores["pos"], compound
        ),
        sentiment_label=label,
        counts={
            "characters": len(original),
            "words_before_filtering": len(cleaned.split()),
            "tokens_after_lemmatization": len(tokens),
            "lines": lines,
        },
    )


def analyze_file(path: str | Path, text_column: str | None = None) -> AnalysisReport:
    """Analyze a supported TXT or CSV file."""
    text, source_name = read_input_file(path, text_column)
    report = analyze_text(text, source_name)
    return AnalysisReport(
        input_summary=InputSummary(
            "txt" if str(path).lower().endswith(".txt") else "csv",
            report.input_summary.source_name,
            report.input_summary.character_count,
            report.input_summary.line_count,
            report.input_summary.token_count,
        ),
        original_text=report.original_text,
        cleaned_text=report.cleaned_text,
        processed_text=report.processed_text,
        tokens=report.tokens,
        sentiment_scores=report.sentiment_scores,
        sentiment_label=report.sentiment_label,
        counts=report.counts,
    )
