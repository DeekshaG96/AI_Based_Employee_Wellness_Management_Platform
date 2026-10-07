"""Unit tests for VADER sentiment analysis engine."""

import pytest
from src.sentiment import SentimentAnalyzer

@pytest.fixture
def analyzer():
    return SentimentAnalyzer()

def test_positive_sentiment(analyzer):
    text = "Our team sprint went amazingly well! Highly motivated and happy."
    res = analyzer.analyze(text)
    assert res["is_valid"] is True
    assert res["sentiment"] == "Positive"
    assert res["compound"] >= 0.05
    assert res["positive"] > 0

def test_negative_sentiment(analyzer):
    text = "I am completely overwhelmed, exhausted, and feeling severe burnout."
    res = analyzer.analyze(text)
    assert res["is_valid"] is True
    assert res["sentiment"] == "Negative"
    assert res["compound"] <= -0.05
    assert res["negative"] > 0
    assert "Burnout" in res["emotion_tone"]

def test_neutral_sentiment(analyzer):
    text = "The status meeting was scheduled at 3 PM today."
    res = analyzer.analyze(text)
    assert res["is_valid"] is True
    assert res["sentiment"] == "Neutral"
    assert -0.05 < res["compound"] < 0.05

def test_empty_sentiment_input(analyzer):
    res = analyzer.analyze("   ")
    assert res["is_valid"] is False
    assert res["sentiment"] == "Neutral"
    assert res["compound"] == 0.0
