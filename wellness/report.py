"""Typed report models for wellness text analysis."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class SentimentLabel(str, Enum):
    """The three VADER sentiment classes used by the project."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SentimentScores:
    negative: float
    neutral: float
    positive: float
    compound: float


@dataclass(frozen=True)
class InputSummary:
    source_type: str
    source_name: str
    character_count: int
    line_count: int
    token_count: int


@dataclass(frozen=True)
class AnalysisReport:
    input_summary: InputSummary
    original_text: str
    cleaned_text: str
    processed_text: str
    tokens: tuple[str, ...]
    sentiment_scores: SentimentScores
    sentiment_label: SentimentLabel
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable machine-readable report."""
        return {
            **asdict(self),
            "input_summary": asdict(self.input_summary),
            "tokens": list(self.tokens),
            "sentiment_scores": asdict(self.sentiment_scores),
            "sentiment_label": self.sentiment_label.value,
        }

    def to_human_readable(self) -> str:
        """Return a concise human-readable report."""
        scores = self.sentiment_scores
        return "\n".join(
            [
                "Employee Wellness Text Analysis",
                f"Source: {self.input_summary.source_name} ({self.input_summary.source_type})",
                (
                    f"Input: {self.input_summary.character_count} characters, "
                    f"{self.input_summary.line_count} lines"
                ),
                f"Processed text: {self.processed_text or '(no tokens)'}",
                f"Tokens: {', '.join(self.tokens) or '(none)'}",
                f"Sentiment: {self.sentiment_label.value}",
                (
                    f"Scores: negative={scores.negative:.3f}, neutral={scores.neutral:.3f}, "
                    f"positive={scores.positive:.3f}, compound={scores.compound:.3f}"
                ),
                "Counts: " + ", ".join(f"{key}={value}" for key, value in self.counts.items()),
            ]
        )
