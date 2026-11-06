"""Model comparison utility for Milestone 2 evaluation.

Provides side-by-side analysis comparing lexical rule-based baselines (VADER + keyword matching)
against the fine-tuned DistilBERT multi-label transformer model on the same input text.
"""

from __future__ import annotations

import re
from typing import Any

from nltk.sentiment import SentimentIntensityAnalyzer

from .pipeline import (
    _EMOTION_KEYWORDS,
    _GOEMOTIONS_GROUPS,
    _dominant_goemotions_category,
    _get_emotion_pipeline,
    classify_emotions_with_method,
    sanitize_text_for_vader,
)
from .preprocessing import sanitize_text_for_vader


def run_lexical_baseline(text: str) -> dict[str, Any]:
    """Run pure lexical rule-based sentiment and emotion classification."""
    vader_input = sanitize_text_for_vader(text)
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(vader_input)
    compound = float(vader_scores["compound"])

    if compound >= 0.05:
        polarity = "positive"
    elif compound <= -0.05:
        polarity = "negative"
    else:
        polarity = "neutral"

    lower = text.lower()
    match_counts = {
        emotion: sum(
            len(re.findall(rf"(?<!\w){re.escape(kw)}(?!\w)", lower))
            for kw in kws
        )
        for emotion, kws in _EMOTION_KEYWORDS.items()
    }
    total = sum(match_counts.values())
    if total > 0:
        emotion_scores = {k: round(v / total, 4) for k, v in match_counts.items() if v > 0}
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    else:
        emotion_scores = {}
        dominant_emotion = "unclassified"

    return {
        "method": "lexical_baseline",
        "polarity": polarity,
        "compound_score": round(compound, 4),
        "vader_scores": {k: round(float(v), 4) for k, v in vader_scores.items()},
        "emotion_scores": emotion_scores,
        "dominant_emotion": dominant_emotion,
        "matched_keyword_counts": {k: v for k, v in match_counts.items() if v > 0},
    }


def run_transformer_analysis(text: str) -> dict[str, Any]:
    """Run fine-tuned DistilBERT transformer multi-label classification."""
    scores, dominant, method = classify_emotions_with_method(text)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_emotions = [{"label": k, "score": round(v, 4)} for k, v in sorted_scores[:6]]

    return {
        "method": method,
        "dominant_emotion": dominant,
        "top_emotions": top_emotions,
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
        "label_count": len(scores),
    }


def compare_models(text: str) -> dict[str, Any]:
    """Produce structured comparative analysis across baseline and transformer models."""
    lexical = run_lexical_baseline(text)
    transformer = run_transformer_analysis(text)

    # Calculate analytical delta
    trans_dominant = transformer["dominant_emotion"]
    lex_dominant = lexical["dominant_emotion"]
    dominant_agreement = (trans_dominant == lex_dominant) if lex_dominant != "unclassified" else False

    # Detect subtle emotions found by transformer (> 0.20) but missing from lexical cues
    trans_scores = transformer["all_scores"]
    lex_matched_emotions = set(lexical["emotion_scores"].keys())
    subtle_emotions = [
        {"label": k, "score": round(v, 4)}
        for k, v in trans_scores.items()
        if v >= 0.20 and k not in lex_matched_emotions and k != "neutral"
    ]
    subtle_emotions.sort(key=lambda x: x["score"], reverse=True)

    # Detect mixed emotions (at least 2 non-neutral emotions with score >= 0.25)
    salient_emotions = [k for k, v in trans_scores.items() if v >= 0.25 and k != "neutral"]
    is_mixed = len(salient_emotions) >= 2

    # Contextual insight synthesis
    if is_mixed:
        insight = (
            f"Mixed emotional state detected with high co-occurrence ({', '.join(salient_emotions[:3])}). "
            "The multi-label transformer resolves these distinct simultaneous signals with independent "
            "sigmoid probabilities, whereas lexical bag-of-words rules either collapse them into a single score "
            "or cancel them out."
        )
    elif not dominant_agreement and lex_dominant != "unclassified":
        insight = (
            f"Model divergence: Lexical keyword rules highlighted '{lex_dominant}', but the deep transformer "
            f"classified the dominant state as '{trans_dominant}'. This highlights the transformer's ability to "
            "process syntactic context, negations, and semantic subtext beyond dictionary keyword lookups."
        )
    elif lex_dominant == "unclassified":
        insight = (
            f"Lexical baseline returned 'unclassified' due to lack of exact dictionary keywords. "
            f"The fine-tuned DistilBERT transformer successfully generalized to capture '{trans_dominant}' "
            f"({transformer['top_emotions'][0]['score']:.2f} confidence) through contextual embeddings."
        )
    else:
        insight = (
            f"Both models agree on '{trans_dominant}'. The transformer provides fine-grained calibrated "
            f"probabilistic confidence across 28 GoEmotions dimensions."
        )

    return {
        "text": text,
        "lexical_baseline": lexical,
        "transformer_model": transformer,
        "comparative_analysis": {
            "dominant_agreement": dominant_agreement,
            "is_mixed_emotion": is_mixed,
            "salient_emotions": salient_emotions,
            "subtle_emotions_detected": subtle_emotions,
            "insight": insight,
        },
    }

