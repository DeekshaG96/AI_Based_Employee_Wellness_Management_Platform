"""Public analysis API combining ingestion, preprocessing, VADER, and emotion telemetry."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

from nltk.sentiment import SentimentIntensityAnalyzer

from .errors import InputValidationError
from .ingestion import read_input_file, validate_text
from .preprocessing import (
    preprocess,
    sanitize_text_for_vader,
    validate_nltk_resources,
)
from .report import AnalysisReport, InputSummary, SentimentLabel, SentimentScores

# The transformer's seven classes; the lexical fallback can also flag fatigue.
EMOTIONS: tuple[str, ...] = ("joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral")

_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "joy": [
        "happy",
        "great",
        "delighted",
        "love",
        "awesome",
        "proud",
        "excited",
        "wonderful",
        "grateful",
        "fantastic",
        "good",
        "satisfied",
        "enjoy",
        "pleased",
    ],
    "sadness": [
        "sad",
        "depressed",
        "unhappy",
        "lonely",
        "down",
        "hopeless",
        "grief",
        "miserable",
        "hurt",
        "crying",
        "lost",
        "burnout",
        "unmotivated",
    ],
    "fatigue": ["tired", "fatigued", "exhausted", "worn out", "sleepy", "drained"],
    "anger": [
        "angry",
        "mad",
        "furious",
        "annoyed",
        "irritated",
        "rage",
        "frustrated",
        "outraged",
        "hate",
        "resentful",
        "hostile",
        "toxic",
    ],
    "fear": [
        "scared",
        "afraid",
        "terrified",
        "anxious",
        "nervous",
        "dread",
        "panic",
        "worried",
        "stressed",
        "alarmed",
        "overwhelmed",
        "insecure",
        "pressure",
    ],
    "surprise": [
        "surprised",
        "shocked",
        "amazed",
        "astonished",
        "unexpected",
        "stunned",
        "wow",
    ],
    "disgust": [
        "disgusted",
        "gross",
        "nasty",
        "revolted",
        "sickening",
        "awful",
        "loathe",
    ],
    "neutral": [
        "okay",
        "fine",
        "normal",
        "routine",
        "regular",
        "standard",
        "average",
        "moderate",
        "daily",
        "status",
    ],
}

_CRISIS_KEYWORDS: tuple[str, ...] = (
    "kill myself",
    "suicide",
    "end my life",
    "want to die",
    "self harm",
    "hurt myself",
    "no reason to live",
)

_EMOTION_CLF: Any = None
_EMOTION_CLF_LOADED: bool = False


def _get_emotion_pipeline() -> Any:
    """Retrieve or lazily initialize the configured Hugging Face emotion classifier."""
    global _EMOTION_CLF, _EMOTION_CLF_LOADED
    if _EMOTION_CLF_LOADED:
        return _EMOTION_CLF

    # Keep the free hosted demo within its memory limit; local Compose can use the transformer.
    if os.getenv("WELLNESS_DISABLE_TRANSFORMER") == "1":
        _EMOTION_CLF = None
        _EMOTION_CLF_LOADED = True
        return None

    # During pytest runs or when offline testing is requested, use high-speed fallback
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("WELLNESS_FAST_TEST"):
        _EMOTION_CLF = None
        _EMOTION_CLF_LOADED = True
        return None

    try:
        from transformers import pipeline

        _EMOTION_CLF = pipeline(
            "text-classification",
            model=os.getenv("MODEL_NAME", "j-hartmann/emotion-english-distilroberta-base"),
            top_k=None,
            device=-1,
        )
    except Exception:
        _EMOTION_CLF = None

    _EMOTION_CLF_LOADED = True
    return _EMOTION_CLF


def classify_emotions_with_method(text: str) -> tuple[dict[str, float], str, str]:
    """Classify emotions and identify whether model scores or fallback heuristics were used."""
    clf = _get_emotion_pipeline()
    if clf is not None:
        try:
            preds = clf(text[:512])[0]
            scores = {item["label"].lower(): float(item["score"]) for item in preds}
            for emo in EMOTIONS:
                scores.setdefault(emo, 0.0)
            dominant = max(scores, key=scores.get)
            return scores, dominant, "transformer"
        except Exception:
            pass

    # Keyword scores are a lightweight cue, not a calibrated emotion probability.
    # Match whole words and phrases so "happy" does not spuriously match "unhappy".
    lower = text.lower()
    match_counts = {
        emotion: sum(
            len(re.findall(rf"(?<!\w){re.escape(keyword)}(?!\w)", lower))
            for keyword in keywords
        )
        for emotion, keywords in _EMOTION_KEYWORDS.items()
    }
    total_matches = sum(match_counts.values())
    if total_matches == 0:
        # Avoid inventing an equal-probability distribution when the text has no
        # supported emotion cue. Neutral sentiment is not implied by this result.
        return {}, "unclassified", "no_emotion_signal"

    normalized = {
        emotion: round(count / total_matches, 4)
        for emotion, count in match_counts.items()
        if count > 0
    }
    dominant = max(normalized, key=normalized.get)
    return normalized, dominant, "lexical_fallback"


def classify_emotions(text: str) -> tuple[dict[str, float], str]:
    """Backward-compatible emotion classifier result without estimator metadata."""
    scores, dominant, _method = classify_emotions_with_method(text)
    return scores, dominant


def _action_paths_for_emotion(emotion: str, *, is_crisis: bool = False) -> list[dict[str, Any]]:
    """Offer a few optional, low-pressure ways forward for this check-in."""
    if is_crisis:
        return [
            {
                "title": "Get to a safer place",
                "duration": "Right now",
                "description": "If you may be in immediate danger, focus on getting in-person help.",
                "steps": ["Call your local emergency number or go to the nearest emergency department."],
            },
            {
                "title": "Bring in someone you trust",
                "duration": "A few minutes",
                "description": "You do not have to handle this moment alone.",
                "steps": ["Call or text someone you trust.", "Tell them plainly that you need them to stay with you or help you get support."],
            },
            {
                "title": "Contact crisis support",
                "duration": "When you can",
                "description": "A trained person can help you work through the next moment.",
                "steps": ["Contact a crisis support service available where you live, or a licensed mental health professional."],
            },
        ]

    choices: dict[str, list[dict[str, Any]]] = {
        "fear": [
            {
                "title": "Ground in the room",
                "duration": "2 minutes",
                "description": "Give your attention a steady, concrete place to land.",
                "steps": ["Name five things you see, four you feel, and three you hear.", "Let your shoulders soften and take one slow exhale."],
            },
            {
                "title": "Make one thing smaller",
                "duration": "5–10 minutes",
                "description": "Turn a large worry into one next action you can influence.",
                "steps": ["Write down what is worrying you.", "Circle one part you can act on today and give it ten minutes."],
            },
            {
                "title": "Ask for clarity or company",
                "duration": "A few minutes",
                "description": "Choose practical help or a supportive check-in.",
                "steps": ["Message someone you trust: ‘Could you listen for a few minutes?’", "If this is about a deadline, ask the person involved what matters most first."],
            },
        ],
        "sadness": [
            {
                "title": "Offer your body a reset",
                "duration": "5 minutes",
                "description": "Keep the first step gentle and easy to complete.",
                "steps": ["Drink some water or get a little fresh air.", "Stretch, sit somewhere comfortable, or rest without needing to earn it."],
            },
            {
                "title": "Choose a tiny win",
                "duration": "5–15 minutes",
                "description": "Reduce the pressure by making one task smaller.",
                "steps": ["Pick one task that would make today slightly easier.", "Do only the first small part, then decide whether to continue."],
            },
            {
                "title": "Reach for connection",
                "duration": "A few minutes",
                "description": "A simple check-in can make a hard day feel less solitary.",
                "steps": ["Send a trusted person a short message or ask to spend time together.", "If this feeling keeps returning, consider talking with a qualified professional."],
            },
        ],
        "burnout": [
            {
                "title": "Create a real pause",
                "duration": "5–10 minutes",
                "description": "Let your attention and body come off the clock briefly.",
                "steps": ["Step away from the task or screen if possible.", "Choose water, a stretch, fresh air, or quiet rest—whichever asks the least of you."],
            },
            {
                "title": "Reduce the load",
                "duration": "5 minutes",
                "description": "Look for one pressure you can make smaller or share.",
                "steps": ["List what feels urgent and what can wait.", "Ask for a priority check or renegotiate one deadline if that is available to you."],
            },
            {
                "title": "Tell someone how depleted you feel",
                "duration": "A few minutes",
                "description": "You deserve support before you reach an empty tank.",
                "steps": ["Contact a trusted person and be specific about the help you need.", "If exhaustion is ongoing, consider talking with a qualified professional."],
            },
        ],
        "anger": [
            {
                "title": "Pause before responding",
                "duration": "10 minutes",
                "description": "Create enough space to choose your response deliberately.",
                "steps": ["Step away from the message or conversation if you can.", "Take a short walk or breathe out slowly before replying."],
            },
            {
                "title": "Separate facts from friction",
                "duration": "5 minutes",
                "description": "Clarify what happened and what you need next.",
                "steps": ["Write the observable facts in one column and your assumptions in another.", "Name one boundary or change that would help."],
            },
            {
                "title": "Prepare a clear conversation",
                "duration": "10 minutes",
                "description": "Use calm, specific language when you feel ready.",
                "steps": ["Try: ‘When ___ happened, I felt ___. Could we ___?’", "Choose a time to talk when you are both able to focus."],
            },
        ],
        "disgust": [
            {
                "title": "Step back from the trigger",
                "duration": "2–5 minutes",
                "description": "A little distance can help you decide what you need.",
                "steps": ["Move away from the situation or pause the conversation if you can.", "Notice what boundary or value feels affected."],
            },
            {
                "title": "Name the specific concern",
                "duration": "5 minutes",
                "description": "Turn a strong reaction into a clear, workable observation.",
                "steps": ["Write down what happened without judging yourself for your reaction.", "Identify one change or boundary that would make the situation feel more acceptable."],
            },
            {
                "title": "Choose a respectful next move",
                "duration": "Flexible",
                "description": "Decide whether to address it, set a limit, or get support.",
                "steps": ["Consider what outcome you want before responding.", "If it feels safe, explain the specific behavior and request a specific change."],
            },
        ],
        "surprise": [
            {
                "title": "Give the news a moment to land",
                "duration": "2 minutes",
                "description": "You do not have to decide what it means immediately.",
                "steps": ["Take a slow breath and notice your first reaction.", "Write one sentence about what you know so far."],
            },
            {
                "title": "Separate what is known from what is next",
                "duration": "5 minutes",
                "description": "A short list can make uncertainty easier to navigate.",
                "steps": ["Write down what is confirmed and what is still unclear.", "Choose one question to ask or one detail to verify."],
            },
            {
                "title": "Talk it through",
                "duration": "A few minutes",
                "description": "A trusted perspective may help you process a sudden change.",
                "steps": ["Tell someone what surprised you.", "Ask them to listen or help you think through the next step."],
            },
        ],
        "joy": [
            {
                "title": "Savor what went well",
                "duration": "2 minutes",
                "description": "Notice what made this moment meaningful so you can remember it.",
                "steps": ["Write down what happened and what you contributed.", "Notice one detail you would like to remember."],
            },
            {
                "title": "Carry the momentum forward",
                "duration": "10 minutes",
                "description": "Use your energy on something that matters to you.",
                "steps": ["Choose one meaningful task or idea.", "Set a short focus block and stop when it ends."],
            },
            {
                "title": "Share appreciation",
                "duration": "A few minutes",
                "description": "Let someone who helped know that it mattered.",
                "steps": ["Send a specific thank-you to someone who contributed.", "If you prefer, keep the note as a private reflection."],
            },
        ],
        "fatigue": [
            {
                "title": "Take a small reset",
                "duration": "5 minutes",
                "description": "Give yourself a brief pause without needing to earn it.",
                "steps": ["Get some water or fresh air if that would help.", "Stretch, sit quietly, or rest for a few minutes."],
            },
            {
                "title": "Make the next task lighter",
                "duration": "5 minutes",
                "description": "A smaller next step can protect your energy.",
                "steps": ["Choose one task that matters most.", "Shorten it, split it into a first step, or let another task wait."],
            },
            {
                "title": "Ask for a little support",
                "duration": "A few minutes",
                "description": "You can name what you need without having a perfect plan.",
                "steps": ["Tell someone you trust that you are feeling tired.", "Ask for one specific kind of help or a chance to pause."],
            },
        ],
        "neutral": [
            {
                "title": "Check what you need",
                "duration": "2 minutes",
                "description": "A quick check-in can reveal the most useful next move.",
                "steps": ["Ask yourself: do I need rest, food, movement, focus, or connection?", "Choose one small response to that need."],
            },
            {
                "title": "Pick one priority",
                "duration": "5 minutes",
                "description": "Give your attention a clear and manageable target.",
                "steps": ["Write down everything competing for your attention.", "Choose one priority and define its first step."],
            },
            {
                "title": "Refresh your routine",
                "duration": "5–10 minutes",
                "description": "A small change of pace can help you reset.",
                "steps": ["Stand, stretch, walk, or look away from your screen.", "Return when you feel ready, without rushing."],
            },
        ],
    }
    key = "fear" if emotion in {"anxiety", "fear"} else emotion
    return choices.get(key, choices["neutral"])


def generate_recommendations(
    dominant_emotion: str,
    sentiment_label: SentimentLabel,
    compound: float,
    text: str,
) -> dict[str, Any]:
    """Generate empathetic, actionable well-being strategies tailored to emotion and polarity."""
    lower = text.lower()
    is_crisis = any(phrase in lower for phrase in _CRISIS_KEYWORDS)

    if is_crisis:
        return {
            "title": "Immediate Support & Emergency Assistance",
            "category": "Crisis Intervention",
            "urgency": "High",
            "summary": (
                "Your message may point to an immediate safety concern. This text analysis cannot "
                "assess risk or provide crisis care. Please contact a person who can support you now."
            ),
            "action_items": [
                "If you may act on these thoughts or are in immediate danger, call your local emergency number or go to the nearest emergency department.",
                "Call or text 988 (Suicide & Crisis Lifeline - 24/7, free, confidential) or text HOME to 741741 (Crisis Text Line).",
                "Contact a local crisis support service or a trusted person and ask them to stay with you.",
                "Reach out to a licensed mental health professional or your organization's employee assistance program.",
            ],
            "action_paths": _action_paths_for_emotion(dominant_emotion, is_crisis=True),
            "is_crisis": True,
        }

    if dominant_emotion in ("fear", "anxiety") or compound < -0.3:
        return {
            "title": "Cognitive Decompression & Stress Triage",
            "category": "Stress Management",
            "urgency": "Medium",
            "summary": (
                "Elevated anxiety and stress indicators detected. Prioritize nervous system "
                "regulation and workload compartmentalization."
            ),
            "action_items": [
                "Practice the 4-7-8 diaphragmatic breathing cycle for 3 minutes.",
                "Execute the '2-Minute Brain Dump' to separate urgent tasks from background noise.",
                "Schedule a 15-minute screen-free decompression walk.",
                "Sync with your mentor or manager to calibrate pending deadlines.",
            ],
            "action_paths": _action_paths_for_emotion(dominant_emotion),
            "is_crisis": False,
        }

    has_fatigue_cue = any(
        re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", lower)
        for keyword in _EMOTION_KEYWORDS["fatigue"]
    )
    if has_fatigue_cue:
        return {
            "title": "Balance Energy with Recovery",
            "category": "Rest & Recovery",
            "urgency": "Low",
            "summary": (
                "Your entry mentions feeling tired. Low energy can exist alongside a positive "
                "mood, so choose a supportive pace instead of pushing harder."
            ),
            "action_items": [
                "Take a brief pause and notice whether water, food, fresh air, or quiet would help.",
                "Choose one priority and let a less urgent task wait if you can.",
                "Tell someone you trust how you are feeling and ask for one small form of support.",
            ],
            "action_paths": _action_paths_for_emotion("fatigue"),
            "is_crisis": False,
        }

    if dominant_emotion in ("sadness", "burnout") or sentiment_label is SentimentLabel.NEGATIVE:
        return {
            "title": "Energy Recovery & Empathetic Reconnection",
            "category": "Emotional Well-Being",
            "urgency": "Medium",
            "summary": (
                "Signals of fatigue and low valence detected. Gentle boundary-setting and "
                "restorative micro-breaks are strongly advised."
            ),
            "action_items": [
                "Take a deliberate 10-minute micro-break away from work stations.",
                "Hydrate and engage in light muscle stretching or sunlight exposure.",
                "Reach out to an empathetic colleague or trusted friend for a casual check-in.",
                "Celebrate small wins accomplished today rather than focusing on uncompleted tasks.",
            ],
            "action_paths": _action_paths_for_emotion(dominant_emotion),
            "is_crisis": False,
        }

    if dominant_emotion == "anger":
        return {
            "title": "Cool-Down Interval & Perspective Realignment",
            "category": "Emotional Regulation",
            "urgency": "Medium",
            "summary": (
                "Frustration markers identified. Avoid reactive communications until baseline "
                "composure is restored."
            ),
            "action_items": [
                "Implement a 10-minute pause before sending challenging emails or Slack responses.",
                "Identify root friction points and write them down objectively without blame.",
                "Channel energy into physical movement or focused independent problem-solving.",
            ],
            "action_paths": _action_paths_for_emotion(dominant_emotion),
            "is_crisis": False,
        }

    if dominant_emotion == "joy" or sentiment_label is SentimentLabel.POSITIVE:
        return {
            "title": "Positive Reinforcement & Energy Amplification",
            "category": "Thriving & Growth",
            "urgency": "Low",
            "summary": (
                "Excellent morale and positive momentum detected! Harness this state to build "
                "resilience and uplift teammates."
            ),
            "action_items": [
                "Log this positive milestone in your personal achievement journal.",
                "Send a note of appreciation or shout-out to a collaborative peer.",
                "Leverage high creative energy on complex tasks or strategic planning.",
            ],
            "action_paths": _action_paths_for_emotion(dominant_emotion),
            "is_crisis": False,
        }

    # Default neutral state
    return {
        "title": "Mindful Cadence & Focus Calibration",
        "category": "Productive Balance",
        "urgency": "Low",
        "summary": (
            "Steady, balanced emotional baseline observed. Maintain consistent pacing and ergonomic habits."
        ),
        "action_items": [
            "Maintain regular hydration and eye-rest intervals (20-20-20 rule).",
            "Plan priorities for tomorrow's morning cycle before wrapping up today.",
            "Take a moment of mindfulness to ground your focus.",
        ],
        "action_paths": _action_paths_for_emotion(dominant_emotion),
        "is_crisis": False,
    }


def analyze_text(text: str, source_name: str = "direct text") -> AnalysisReport:
    """Analyze text and return a typed AnalysisReport with strict thresholds and emotion telemetry."""
    original = validate_text(text)
    validate_nltk_resources()

    # Preprocessing: VADER receives sanitized text preserving caps, punctuation, and negations
    vader_input = sanitize_text_for_vader(original)
    cleaned, tokens = preprocess(original)

    # VADER Polarity Scoring
    scores = SentimentIntensityAnalyzer().polarity_scores(vader_input)
    compound = scores["compound"]

    # Strict Polarity Thresholds (Evaluation Fault #2 Resolution)
    # Positive: compound > +0.05
    # Negative: compound < -0.05
    # Neutral: -0.05 <= compound <= +0.05
    if compound > 0.05:
        label = SentimentLabel.POSITIVE
    elif compound < -0.05:
        label = SentimentLabel.NEGATIVE
    else:
        label = SentimentLabel.NEUTRAL

    # Fine-grained Transformer Emotion Classification
    emotion_scores, dominant_emotion, emotion_method = classify_emotions_with_method(original)

    # Personalized Well-Being Recommendations
    recs = generate_recommendations(dominant_emotion, label, compound, original)

    lines = len(original.splitlines())
    return AnalysisReport(
        input_summary=InputSummary("text", source_name, len(original), lines, len(tokens)),
        original_text=original,
        cleaned_text=cleaned,
        processed_text=" ".join(tokens),
        tokens=tokens,
        sentiment_scores=SentimentScores(scores["neg"], scores["neu"], scores["pos"], compound),
        sentiment_label=label,
        counts={
            "characters": len(original),
            "words_before_filtering": len(cleaned.split()),
            "tokens_after_lemmatization": len(tokens),
            "lines": lines,
        },
        vader_input_text=vader_input,
        emotion_method=emotion_method,
        dominant_emotion=dominant_emotion,
        emotion_scores=emotion_scores,
        recommendations=recs,
    )


def analyze_file(path: str | Path, text_column: str | None = None) -> AnalysisReport:
    """Analyze a supported TXT or single-entry/aggregated CSV file."""
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
        vader_input_text=report.vader_input_text,
        emotion_method=report.emotion_method,
        dominant_emotion=report.dominant_emotion,
        emotion_scores=report.emotion_scores,
        recommendations=report.recommendations,
    )


def analyze_csv_records(path: str | Path, text_column: str | None = None) -> list[AnalysisReport]:
    """Parse each row in a CSV file and analyze entries individually for trend tracking."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise InputValidationError(f"File not found: {file_path}")

    reports: list[AnalysisReport] = []
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise InputValidationError("CSV must contain a header row.")

        columns = [c.strip() for c in reader.fieldnames if c]
        selected = (
            text_column.strip()
            if text_column
            else next(
                (
                    c
                    for c in columns
                    if c.lower() in ("text", "feedback", "comment", "message", "content")
                ),
                None,
            )
        )
        if not selected:
            raise InputValidationError("Could not detect text feedback column in CSV.")

        for idx, row in enumerate(reader, start=1):
            val = (row.get(selected) or "").strip()
            if val:
                rep = analyze_text(val, f"{file_path.name} (Row #{idx})")
                reports.append(rep)

    return reports
