"""NLTK text preprocessing pipeline for MoodMentor.

Provides two dedicated paths:
1. `sanitize_text_for_vader(text)` — Light sanitization strictly preserving
   capitalization, punctuation, intensifiers, and negations for VADER.
2. `preprocess(text)` / `get_preprocessing_breakdown(text)` — Full sequential
   NLP breakdown (Normalization -> Tokenization -> Negation-Preserved Stopwords
   -> POS Tagging -> WordNet Lemmatization) for transparency and keyword analytics.
"""

from __future__ import annotations

import re
from typing import Any

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .errors import NLTKResourceError

_NON_LETTER = re.compile(r"[^a-zA-Z\s]")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_HTML = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"[ \t]+")

# Common contractions expansion mapping
_CONTRACTIONS: dict[str, str] = {
    r"\bcan\'t\b": "cannot",
    r"\bwon\'t\b": "will not",
    r"\bain\'t\b": "is not",
    r"\bn\'t\b": " not",
    r"\b\'re\b": " are",
    r"\b\'s\b": " is",
    r"\b\'d\b": " would",
    r"\b\'ll\b": " will",
    r"\b\'t\b": " not",
    r"\b\'ve\b": " have",
    r"\b\'m\b": " am",
}

# Negation words crucial for sentiment polarity that must NEVER be stripped as stopwords
_NEGATIONS: frozenset[str] = frozenset(
    {"no", "not", "nor", "never", "n't", "nothing", "none", "neither", "cannot"}
)

_REQUIRED_RESOURCES = (
    (("corpora/stopwords",), "stopwords"),
    (("corpora/wordnet", "corpora/wordnet.zip"), "wordnet"),
    (("sentiment/vader_lexicon", "sentiment/vader_lexicon.zip"), "vader_lexicon"),
)


def validate_nltk_resources() -> None:
    """Raise a clear setup error if required NLTK data is unavailable."""
    import nltk

    missing: list[str] = []
    for paths, name in _REQUIRED_RESOURCES:
        if not any(_resource_exists(nltk, path) for path in paths):
            missing.append(name)
    if not any(
        _resource_exists(nltk, path)
        for path in ("tokenizers/punkt", "tokenizers/punkt_tab")
    ):
        missing.append("punkt")
    if not any(
        _resource_exists(nltk, path)
        for path in (
            "taggers/averaged_perceptron_tagger",
            "taggers/averaged_perceptron_tagger_eng",
        )
    ):
        missing.append("averaged_perceptron_tagger")
    if missing:
        joined = ", ".join(missing)
        raise NLTKResourceError(
            f"Missing NLTK data: {joined}. Run "
            "`python -m wellness.cli --download-nltk-data` and try again."
        )


def _resource_exists(nltk_module: object, path: str) -> bool:
    try:
        nltk_module.data.find(path)  # type: ignore[attr-defined]
    except LookupError:
        return False
    return True


def expand_contractions(text: str) -> str:
    """Expand standard English contractions to preserve polarity."""
    for pattern, replacement in _CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def sanitize_text_for_vader(text: str) -> str:
    """Sanitize text for VADER without altering punctuation, caps, or sentiment structure.

    VADER is a rule-based model relying on capitalizations (e.g. GREAT),
    punctuation (e.g. !!!), and negation words. We strip only external noise
    (URLs, HTML tags, emails) and expand contractions.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    cleaned = _URL.sub(" ", text)
    cleaned = _EMAIL.sub(" ", cleaned)
    cleaned = _HTML.sub(" ", cleaned)
    cleaned = expand_contractions(cleaned)
    # Collapse excess horizontal whitespace while keeping punctuation and casing
    cleaned = _MULTI_WS.sub(" ", cleaned).strip()
    return cleaned


def clean_text(text: str) -> str:
    """Cleaning step for lexical display: strip noise, non-letters, and normalize case."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    sanitized = sanitize_text_for_vader(text).lower()
    return re.sub(r"\s+", " ", _NON_LETTER.sub(" ", sanitized)).strip()


def _wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return "n"


def preprocess(text: str) -> tuple[str, tuple[str, ...]]:
    """Run full sequential NLP pipeline for keyword analytics with negations preserved."""
    validate_nltk_resources()
    cleaned = clean_text(text)
    raw_tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words("english")) - _NEGATIONS
    filtered = [token for token in raw_tokens if token.isalpha() and token not in stop_words]
    tagged = pos_tag(filtered)
    lemmatizer = WordNetLemmatizer()
    lemmas = tuple(lemmatizer.lemmatize(token, _wordnet_pos(tag)) for token, tag in tagged)
    return cleaned, lemmas


def get_preprocessing_breakdown(text: str) -> dict[str, Any]:
    """Return all intermediate stages of the preprocessing pipeline for visual UI display."""
    validate_nltk_resources()
    vader_input = sanitize_text_for_vader(text)
    cleaned = clean_text(text)
    raw_tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words("english")) - _NEGATIONS
    filtered = [token for token in raw_tokens if token.isalpha() and token not in stop_words]
    tagged = pos_tag(filtered)
    lemmatizer = WordNetLemmatizer()
    lemmas = tuple(lemmatizer.lemmatize(token, _wordnet_pos(tag)) for token, tag in tagged)
    return {
        "raw_text": text,
        "vader_input": vader_input,
        "cleaned_text": cleaned,
        "raw_tokens": raw_tokens,
        "filtered_tokens": filtered,
        "pos_tags": tagged,
        "lemmas": lemmas,
        "processed_text": " ".join(lemmas),
    }
