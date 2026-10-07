"""Unit tests for text preprocessing pipeline."""

import pytest
from src.preprocessing import TextPreprocessor

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_clean_text(preprocessor):
    raw = "Check this URL https://example.com and email test@infosys.com! Hello World..."
    cleaned = preprocessor.clean_text(raw)
    assert "https" not in cleaned
    assert "@" not in cleaned
    assert "hello world" in cleaned

def test_tokenize(preprocessor):
    tokens = preprocessor.tokenize("wellness balance health")
    assert tokens == ["wellness", "balance", "health"]

def test_stopword_filtering(preprocessor):
    tokens = ["this", "is", "a", "good", "team", "not", "stressed"]
    filtered = preprocessor.filter_stopwords(tokens, preserve_negations=True)
    assert "this" not in filtered
    assert "is" not in filtered
    assert "good" in filtered
    assert "not" in filtered  # Negation preserved

def test_lemmatization(preprocessor):
    tokens = ["running", "meetings", "improved", "working"]
    lemmas = preprocessor.lemmatize(tokens)
    assert "run" in lemmas
    assert "meeting" in lemmas

def test_empty_input_handling(preprocessor):
    res = preprocessor.process("   \n\t  ")
    assert res["is_valid"] is False
    assert "empty" in res["error"].lower()
