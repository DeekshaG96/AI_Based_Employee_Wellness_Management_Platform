"""NLTK text preprocessing in the documented Milestone 1 order."""

import re

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .errors import NLTKResourceError

_NON_LETTER = re.compile(r"[^a-zA-Z\s]")
_URL = re.compile(r"https?://\S+|www\.\S+")
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


def clean_text(text: str) -> str:
    """Cleaning step: remove URLs, punctuation/numbers, and normalize case."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.sub(r"\s+", " ", _NON_LETTER.sub(" ", _URL.sub(" ", text.lower()))).strip()


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
    """Run cleaning, tokenization, stop-word filtering, and lemmatization."""
    validate_nltk_resources()
    cleaned = clean_text(text)
    raw_tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words("english"))
    filtered = [token for token in raw_tokens if token.isalpha() and token not in stop_words]
    tagged = pos_tag(filtered)
    lemmatizer = WordNetLemmatizer()
    lemmas = tuple(lemmatizer.lemmatize(token, _wordnet_pos(tag)) for token, tag in tagged)
    return cleaned, lemmas


def get_preprocessing_breakdown(text: str) -> dict[str, object]:
    """Return all intermediate stages of the preprocessing pipeline for visual display."""
    validate_nltk_resources()
    cleaned = clean_text(text)
    raw_tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words("english"))
    filtered = [token for token in raw_tokens if token.isalpha() and token not in stop_words]
    tagged = pos_tag(filtered)
    lemmatizer = WordNetLemmatizer()
    lemmas = tuple(lemmatizer.lemmatize(token, _wordnet_pos(tag)) for token, tag in tagged)
    return {
        "raw_text": text,
        "cleaned_text": cleaned,
        "raw_tokens": raw_tokens,
        "filtered_tokens": filtered,
        "pos_tags": tagged,
        "lemmas": lemmas,
        "processed_text": " ".join(lemmas),
    }

