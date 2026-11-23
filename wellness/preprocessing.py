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

    # Register local nltk_data directories if present
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in [os.path.join(repo_root, "nltk_data"), os.path.expanduser("~/nltk_data")]:
        if os.path.exists(d) and d not in nltk.data.path:
            nltk.data.path.insert(0, d)

    if os.getenv("WELLNESS_BUNDLED_VADER") == "1":
        return

    # Check if VADER lexicon already exists
    if any(_resource_exists(nltk, path) for path in _REQUIRED_RESOURCES[2][0]):
        return

    # Auto-download on-the-fly for hosted environments (e.g. Streamlit Cloud)
    try:
        nltk.download("vader_lexicon", quiet=True)
        if any(_resource_exists(nltk, path) for path in _REQUIRED_RESOURCES[2][0]):
            return
    except Exception:
        pass

    # Seamless fallback to bundled vaderSentiment if installed
    try:
        import vaderSentiment  # noqa: F401

        os.environ["WELLNESS_BUNDLED_VADER"] = "1"
        return
    except ImportError:
        pass

    # Fallback to zero-dependency bundled heuristic analyzer so analysis never crashes
    os.environ["WELLNESS_BUNDLED_VADER"] = "1"


def _resource_exists(nltk_module: object, path: str) -> bool:
    try:
        nltk_module.data.find(path)  # type: ignore[attr-defined]
    except (LookupError, PermissionError, OSError):
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


_HEURISTIC_POS: dict[str, str] = {
    # Adjectives (JJ)
    "bad": "JJ", "good": "JJ", "great": "JJ", "happy": "JJ", "sad": "JJ", "tired": "JJ",
    "exhausted": "JJ", "calm": "JJ", "peaceful": "JJ", "radiant": "JJ", "anxious": "JJ",
    "stressed": "JJ", "overwhelmed": "JJ", "angry": "JJ", "terrible": "JJ", "awful": "JJ",
    "amazing": "JJ", "wonderful": "JJ", "healthy": "JJ", "hard": "JJ", "easy": "JJ",
    # Verbs (VB)
    "sleep": "VB", "work": "VB", "feel": "VB", "do": "VB", "make": "VB", "go": "VB",
    "take": "VB", "see": "VB", "come": "VB", "think": "VB", "look": "VB", "want": "VB",
    "give": "VB", "use": "VB", "find": "VB", "tell": "VB", "ask": "VB", "try": "VB",
    # Modals / Negations (MD / RB)
    "cannot": "MD", "not": "RB", "never": "RB", "no": "DT", "neither": "DT",
    # Nouns (NN)
    "today": "NN", "time": "NN", "day": "NN", "week": "NN", "job": "NN", "workload": "NN",
    "boss": "NN", "team": "NN", "project": "NN", "deadline": "NN", "meeting": "NN",
    "stress": "NN", "burnout": "NN", "energy": "NN", "mind": "NN", "body": "NN", "heart": "NN",
}


def _heuristic_pos_tag(token: str) -> str:
    """Return a linguistically grounded POS tag fallback when NLTK corpora are unavailable."""
    lower = token.lower()
    if lower in _HEURISTIC_POS:
        return _HEURISTIC_POS[lower]
    if lower.endswith("ly"):
        return "RB"
    if lower.endswith("ing"):
        return "VBG"
    if lower.endswith("ed"):
        return "VBD"
    if lower.endswith(("ful", "ous", "ive", "able")):
        return "JJ"
    return "NN"


def _heuristic_lemmatize(token: str) -> str:
    """Return a simple morphological stem fallback when WordNet is unavailable."""
    lower = token.lower()
    if lower.endswith("ing") and len(lower) > 5:
        return lower[:-3]
    if lower.endswith("ed") and len(lower) > 4:
        return lower[:-2]
    if lower.endswith("s") and len(lower) > 3 and not lower.endswith("ss") and lower not in _NEGATIONS:
        return lower[:-1]
    return lower


def _tokenize_and_lemmatize(
    cleaned: str,
) -> tuple[list[str], list[str], list[tuple[str, str]], tuple[str, ...], list[str]]:
    """Use corpus-free filtering in hosted mode, else spaCy or the NLTK fallback."""
    notes: list[str] = []
    if _lightweight_preprocessing_enabled():
        raw_tokens = re.findall(r"[a-z]+", cleaned.lower())
        stop_words = _HOSTED_STOP_WORDS - _NEGATIONS
        filtered = [token for token in raw_tokens if token not in stop_words]
        tagged = [(token, _heuristic_pos_tag(token)) for token in filtered]
        lemmas = tuple(_heuristic_lemmatize(token) for token in filtered)
        return raw_tokens, filtered, tagged, lemmas, notes

    try:
        stop_words = set(stopwords.words("english")) - _NEGATIONS
    except (LookupError, PermissionError, OSError):
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
    except (LookupError, PermissionError, OSError):
        raw_tokens = re.findall(r"[a-z]+", cleaned.lower())
        notes.append(
            "NLTK tokenization data were unavailable; regex tokenization was used."
        )
    filtered = [token for token in raw_tokens if token.isalpha() and token not in stop_words]
    try:
        tagged = pos_tag(filtered)
    except (LookupError, PermissionError, OSError):
        tagged = [(token, _heuristic_pos_tag(token)) for token in filtered]
        notes.append(
            "NLTK part-of-speech data were unavailable; grounded heuristic POS labels were used."
        )
    try:
        lemmatizer = WordNetLemmatizer()
        lemmas = tuple(lemmatizer.lemmatize(token, _wordnet_pos(tag)) for token, tag in tagged)
    except (LookupError, PermissionError, OSError):
        lemmas = tuple(_heuristic_lemmatize(token) for token in filtered)
        notes.append(
            "WordNet data were unavailable; grounded morphological stem rules were used."
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
    stopwords_removed = [t for t in raw_tokens if t not in filtered]
    negations_preserved = [t for t in filtered if t in _NEGATIONS]
    content_tokens = [t for t in filtered if t not in _NEGATIONS]
    return {
        "raw_text": text,
        "vader_input": vader_input,
        "cleaned_text": cleaned,
        "raw_tokens": raw_tokens,
        "filtered_tokens": filtered,
        "stopwords_removed": stopwords_removed,
        "negations_preserved": negations_preserved,
        "content_tokens": content_tokens,
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
