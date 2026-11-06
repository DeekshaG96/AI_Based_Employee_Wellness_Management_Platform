"""spaCy and NLTK text preprocessing pipeline for MoodMentor.

Provides two dedicated paths:
1. `sanitize_text_for_vader(text)` — Light sanitization strictly preserving
   capitalization, punctuation, intensifiers, and negations for VADER.
2. `preprocess(text)` / `get_preprocessing_breakdown(text)` — Full sequential
   NLP breakdown (Normalization -> Tokenization -> Negation-Preserved Stopwords
   -> spaCy/WordNet locally, or corpus-free token filtering in the hosted showcase).
"""

from __future__ import annotations

import os
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

# Small dependency-free list used only by the hosted showcase. The full local app
# continues to use the complete NLTK stopword corpus and spaCy/WordNet pipeline.
_HOSTED_STOP_WORDS: frozenset[str] = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "while", "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them", "their", "this", "that", "these", "those", "what", "which", "who", "when", "where", "why", "how", "as", "at", "by", "for", "from", "in", "into", "of", "on", "to", "with", "about", "between", "through", "before", "after", "above", "below", "up", "down", "out", "off", "over", "under", "again", "then", "once", "here", "there", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "can", "could", "would", "should", "will", "may", "might", "must", "also", "just", "very"]
)


def _lightweight_preprocessing_enabled() -> bool:
    return os.getenv("WELLNESS_LIGHTWEIGHT_PREPROCESSING") == "1"

_REQUIRED_RESOURCES = (
    (("corpora/stopwords",), "stopwords"),
    (("corpora/wordnet", "corpora/wordnet.zip"), "wordnet"),
    (("sentiment/vader_lexicon", "sentiment/vader_lexicon.zip"), "vader_lexicon"),
)

_SPACY_NLP: Any = None
_SPACY_CHECKED = False


def _get_spacy_pipeline() -> Any:
    """Load spaCy's English pipeline once, returning None when it is not installed."""
    global _SPACY_NLP, _SPACY_CHECKED
    if _SPACY_CHECKED:
        return _SPACY_NLP
    _SPACY_CHECKED = True
    try:
        import spacy

        _SPACY_NLP = spacy.load(
            os.getenv("SPACY_MODEL", "en_core_web_sm"), disable=["parser", "ner"]
        )
    except (ImportError, OSError):
        _SPACY_NLP = None
    return _SPACY_NLP


def validate_nltk_resources() -> None:
    """Validate only resources that cannot be replaced by a safe local fallback."""
    if _lightweight_preprocessing_enabled():
        return

    import nltk

    if os.getenv("WELLNESS_BUNDLED_VADER") != "1" and not any(
        _resource_exists(nltk, path) for path in _REQUIRED_RESOURCES[2][0]
    ):
        raise NLTKResourceError(
            "Could not load the VADER language resource. Run "
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


def _tokenize_and_lemmatize(
    cleaned: str,
) -> tuple[list[str], list[str], list[tuple[str, str]], tuple[str, ...], list[str]]:
    """Use corpus-free filtering in hosted mode, else spaCy or the NLTK fallback."""
    notes: list[str] = []
    if _lightweight_preprocessing_enabled():
        raw_tokens = re.findall(r"[a-z]+", cleaned.lower())
        stop_words = _HOSTED_STOP_WORDS - _NEGATIONS
        filtered = [token for token in raw_tokens if token not in stop_words]
        tagged = [(token, "TOKEN") for token in filtered]
        # Keep surface forms here; without a corpus, do not claim lemmatization.
        return raw_tokens, filtered, tagged, tuple(filtered), notes

    try:
        stop_words = set(stopwords.words("english")) - _NEGATIONS
    except LookupError:
        stop_words = set(_HOSTED_STOP_WORDS) - _NEGATIONS
        notes.append(
            "NLTK stopwords were unavailable; the built-in English stopword list was used."
        )
    nlp = _get_spacy_pipeline()
    if nlp is not None:
        document = nlp(cleaned)
        raw_tokens = [token.text.lower() for token in document if not token.is_space]
        kept = [
            token for token in document if token.is_alpha and token.text.lower() not in stop_words
        ]
        filtered = [token.text.lower() for token in kept]
        tagged = [(token.text.lower(), token.tag_ or token.pos_) for token in kept]
        lemmas = tuple((token.lemma_ or token.text).lower().strip() for token in kept)
        return raw_tokens, filtered, tagged, lemmas, notes

    try:
        raw_tokens = word_tokenize(cleaned)
    except LookupError:
        raw_tokens = re.findall(r"[a-z]+", cleaned.lower())
        notes.append(
            "NLTK tokenization data were unavailable; regex tokenization was used."
        )
    filtered = [token for token in raw_tokens if token.isalpha() and token not in stop_words]
    try:
        tagged = pos_tag(filtered)
    except LookupError:
        tagged = [(token, "TOKEN") for token in filtered]
        notes.append(
            "NLTK part-of-speech data were unavailable; surface token labels were used."
        )
    try:
        lemmatizer = WordNetLemmatizer()
        lemmas = tuple(lemmatizer.lemmatize(token, _wordnet_pos(tag)) for token, tag in tagged)
    except LookupError:
        lemmas = tuple(filtered)
        notes.append(
            "WordNet data were unavailable; surface forms were retained instead of claiming lemmatization."
        )
    return raw_tokens, filtered, tagged, lemmas, notes


def preprocess(text: str) -> tuple[str, tuple[str, ...]]:
    """Run full sequential NLP pipeline for keyword analytics with negations preserved."""
    validate_nltk_resources()
    cleaned = clean_text(text)
    _raw_tokens, _filtered, _tagged, lemmas, _notes = _tokenize_and_lemmatize(cleaned)
    return cleaned, lemmas


def get_preprocessing_breakdown(text: str) -> dict[str, Any]:
    """Return all intermediate stages of the preprocessing pipeline for visual UI display."""
    validate_nltk_resources()
    vader_input = sanitize_text_for_vader(text)
    cleaned = clean_text(text)
    raw_tokens, filtered, tagged, lemmas, notes = _tokenize_and_lemmatize(cleaned)
    return {
        "raw_text": text,
        "vader_input": vader_input,
        "cleaned_text": cleaned,
        "raw_tokens": raw_tokens,
        "filtered_tokens": filtered,
        "pos_tags": tagged,
        "lemmas": lemmas,
        "processed_text": " ".join(lemmas),
        "tokenization_method": (
            "Lightweight word tokenization"
            if _lightweight_preprocessing_enabled()
            else "spaCy or NLTK tokenization"
        ),
        "lemmatization_method": (
            "Surface-form tokens (corpus-free hosted mode)"
            if _lightweight_preprocessing_enabled()
            else (
                "Surface-form tokens (WordNet unavailable)"
                if any("WordNet data" in note for note in notes)
                else "spaCy or WordNet lemmatization"
            )
        ),
        "resource_notes": notes,
    }
