"""Public analysis API combining ingestion, preprocessing, VADER, and emotion telemetry."""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Callable
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

# Broad recommendation categories; GoEmotions also supplies fine-grained labels.
EMOTIONS: tuple[str, ...] = ("joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral")

_GOEMOTIONS_GROUPS: dict[str, tuple[str, ...]] = {
    "anger": ("anger", "annoyance", "disapproval"),
    "disgust": ("disgust",),
    "fear": ("fear", "nervousness"),
    "joy": (
        "joy", "amusement", "approval", "excitement", "gratitude", "love",
        "optimism", "relief", "pride", "admiration", "desire", "caring",
    ),
    "sadness": ("sadness", "disappointment", "embarrassment", "grief", "remorse"),
    "surprise": ("surprise", "realization", "confusion", "curiosity"),
}
_GOEMOTIONS_FINE_LABELS = {
    label for labels in _GOEMOTIONS_GROUPS.values() for label in labels
} - set(EMOTIONS)

EmotionClassifier = Callable[[str], tuple[dict[str, float], str, str]]


def _dominant_goemotions_category(scores: dict[str, float]) -> str:
    """Map independent GoEmotions scores to a broad recommendation category."""
    grouped = {
        category: max((scores.get(label, 0.0) for label in labels), default=0.0)
        for category, labels in _GOEMOTIONS_GROUPS.items()
    }
    grouped["neutral"] = scores.get("neutral", 0.0)
    return max(grouped, key=grouped.get)


def classify_emotions_with_hf_api(
    text: str,
    *,
    api_key: str,
    model_name: str = "SamLowe/roberta-base-go_emotions",
) -> tuple[dict[str, float], str, str]:
    """Classify one explicitly opted-in reflection through Hugging Face Inference."""
    if not api_key.strip():
        raise ValueError("A Hugging Face Inference token is required.")

    from huggingface_hub import InferenceClient

    client = InferenceClient(provider="hf-inference", api_key=api_key, timeout=25)
    predictions = client.text_classification(
        text[:512],
        model=model_name,
        top_k=None,
        function_to_apply="sigmoid",
    )
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]

    scores: dict[str, float] = {}
    known_labels = _GOEMOTIONS_FINE_LABELS | set(EMOTIONS)
    for item in predictions or []:
        label = item.get("label") if isinstance(item, dict) else getattr(item, "label", None)
        score = item.get("score") if isinstance(item, dict) else getattr(item, "score", None)
        normalized_label = str(label or "").strip().lower()
        if normalized_label not in known_labels:
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if 0.0 <= numeric_score <= 1.0:
            scores[normalized_label] = numeric_score

    if not scores:
        raise ValueError("The Hugging Face model returned no recognized GoEmotions labels.")
    return scores, _dominant_goemotions_category(scores), "transformer_remote_api"

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

        configured_model = os.getenv("MODEL_NAME")
        local_model = Path(__file__).resolve().parents[1] / "models" / "goemotions-distilbert"
        if configured_model:
            model_name = configured_model
        elif (local_model / "config.json").is_file():
            model_name = str(local_model)
        else:
            model_name = "BruceIC/emoticare"
        _EMOTION_CLF = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=-1,
        )
        if model_name.rstrip("/").lower() in {
            "bhadresh-savani/bert-base-go-emotion",
            "bruceic/emoticare",
        }:
            # These published GoEmotions checkpoints have 28 independent labels.
            # Set the task type when a model config omits it so Transformers uses sigmoid.
            _EMOTION_CLF.model.config.problem_type = "multi_label_classification"
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
            scores = {item["label"].strip().lower(): float(item["score"]) for item in preds}
            if not scores:
                raise ValueError("The emotion model returned no labels.")

            # GoEmotions is multi-label: preserve all 28 independent scores for
            # the report, but map its strongest broad family for recommendations.
            if set(scores).intersection(_GOEMOTIONS_FINE_LABELS):
                return scores, _dominant_goemotions_category(scores), "transformer_multilabel"

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
                "title": "Ground in the room (5-4-3-2-1 Sensory Reset)",
                "duration": "2 minutes",
                "description": "Give your attention a steady, concrete place to land by anchoring to sensory cues.",
                "steps": [
                    "Name 5 things you can visually see in your immediate room.",
                    "Notice 4 physical textures you can touch (desk, clothing, chair, floor).",
                    "Identify 3 distinct ambient sounds around you.",
                    "Drop your shoulders, unclamp your jaw, and take 1 slow, audible exhale.",
                ],
            },
            {
                "title": "The Physiological Sigh (Autonomic Reset)",
                "duration": "60 seconds",
                "description": "Rapidly triggers parasympathetic braking and lowers rapid heart rate.",
                "steps": [
                    "Take two deep inhales through your nose: one long breath, followed by a quick top-off sip of air.",
                    "Release an extended, slow, audible sigh through your mouth.",
                    "Repeat for 3 to 5 cycles to pop open collapsed alveoli and lower arterial CO2.",
                ],
            },
            {
                "title": "Cognitive Defusion (Leaves on a Stream)",
                "duration": "3 minutes",
                "description": "Notice anxious work thoughts without getting hooked or dragged into them.",
                "steps": [
                    "Visualize a quiet, gently flowing stream with broad green leaves floating past.",
                    "Whenever an anxious deadline or fear arises, gently place it on a leaf and let it drift around the bend.",
                    "Notice that you are the riverbank, not the floating leaf.",
                ],
            },
            {
                "title": "Make one thing smaller (10-Minute Focus Block)",
                "duration": "10 minutes",
                "description": "Turn an overwhelming worry into one tiny, concrete micro-action you control.",
                "steps": [
                    "Write down the single task or obstacle triggering the most anxiety.",
                    "Carve out the easiest first step (e.g., write the first bullet or open the document).",
                    "Set a 10-minute timer and focus only on that sub-step, ignoring everything else.",
                ],
            },
            {
                "title": "Mammalian Dive Reflex & Temperature Reset",
                "duration": "2 minutes",
                "description": "Cold temperature stimulates the vagus nerve and downregulates acute sympathetic arousal.",
                "steps": [
                    "Go to the washroom and splash cool water over your forehead, temples, and wrists.",
                    "Alternatively, press a cool damp cloth against your cheekbones for 30 seconds.",
                    "Notice how your heart rate naturally slows as your body registers the cooling cue.",
                ],
            },
            {
                "title": "Ask for clarity or collaborative triage",
                "duration": "5 minutes",
                "description": "Choose practical calibration over isolated cognitive looping.",
                "steps": [
                    "Message a trusted team lead or peer: 'Could we sync for 5 mins to prioritize pending deliverables?'",
                    "Clearly ask what the single most critical deliverable is for today.",
                ],
            },
        ],
        "sadness": [
            {
                "title": "Offer your body a gentle physical reset",
                "duration": "5 minutes",
                "description": "Keep the first step gentle, warming, and easy to complete.",
                "steps": [
                    "Sip a cup of warm water or chamomile tea.",
                    "Stretch softly, sit somewhere comfortable, or rest without needing to earn it.",
                ],
            },
            {
                "title": "Choose a micro-win (5-Minute Threshold Rule)",
                "duration": "5 minutes",
                "description": "Reduce internal friction by shrinking expectations down to a single tiny milestone.",
                "steps": [
                    "Pick one task that would make today slightly easier.",
                    "Agree with yourself to work on it for only 5 minutes; stop if you need to, or continue if momentum builds.",
                ],
            },
            {
                "title": "Self-Compassion Pause (Hand-on-Heart Anchor)",
                "duration": "2 minutes",
                "description": "Activates somatic oxytocin release to soothe feelings of inadequacy or loneliness.",
                "steps": [
                    "Place your warm right hand flat over the center of your chest or heart.",
                    "Take 3 slow, soothing breaths feeling the warmth of your palm against your chest.",
                    "Silently remind yourself: 'This is a difficult moment. I am meeting myself with gentleness.'",
                ],
            },
            {
                "title": "Sensory Uplift & Outdoor Photons",
                "duration": "5–10 minutes",
                "description": "Natural ambient light and fresh airflow gently re-engage the nervous system.",
                "steps": [
                    "Step outdoors or stand by an open window with natural sunlight.",
                    "Look toward the natural horizon or greenery, allowing your eye muscles to soften.",
                ],
            },
            {
                "title": "Reach for connection without pressure",
                "duration": "A few minutes",
                "description": "A simple check-in can make a heavy day feel less solitary.",
                "steps": [
                    "Send a trusted friend or peer a short greeting or emoji.",
                    "You do not need to explain everything; simply maintaining low-pressure contact helps.",
                ],
            },
            {
                "title": "Expressive 3-Minute Stream-of-Consciousness",
                "duration": "3 minutes",
                "description": "Discharge heavy emotional weight from working memory onto paper.",
                "steps": [
                    "Write continuously without editing, filtering, or fixing spelling.",
                    "Once finished, close the notepad and breathe deeply.",
                ],
            },
        ],
        "burnout": [
            {
                "title": "Create a true sensory pause",
                "duration": "5–10 minutes",
                "description": "Let your attention and nervous system come completely off the clock.",
                "steps": [
                    "Step away from your workstation, monitor, and smartphone.",
                    "Choose quiet rest, a gentle stretch, or drinking water—whichever asks the least effort.",
                ],
            },
            {
                "title": "Strict Scope Triage (The 3-Bucket Rule)",
                "duration": "5 minutes",
                "description": "Separate what is genuinely non-negotiable from low-yield background noise.",
                "steps": [
                    "Write down tasks into 3 buckets: Must-Do Today, Can Wait Until Tomorrow, Can Be Dropped.",
                    "Firmly defer the bottom two buckets without guilt.",
                ],
            },
            {
                "title": "Vagal Sub-Occipital Neck Release",
                "duration": "3 minutes",
                "description": "Releases tech-neck muscular tension and promotes vagal parasympathetic flow.",
                "steps": [
                    "Interlace your fingers behind your head at the base of your skull.",
                    "Keep your head facing forward while resting your weight back into your hands.",
                    "Hold gently for 60 seconds while taking smooth belly breaths.",
                ],
            },
            {
                "title": "Establish an Uncompromising Evening Cutoff",
                "duration": "2 minutes",
                "description": "Guards circadian recovery by drawing a clear boundary between work and rest.",
                "steps": [
                    "Pick a hard time to close your work laptop tonight (e.g., 6:30 PM).",
                    "Put an end-of-day block on your calendar so coworkers see your availability boundary.",
                ],
            },
            {
                "title": "Communicate capacity limits proactively",
                "duration": "5 minutes",
                "description": "Honest communication protects long-term productivity and prevents severe burnout.",
                "steps": [
                    "Frame workload to your manager: 'To maintain high quality on Project A, I need to pause Project B.'",
                    "Offer solutions rather than absorbing endless extra tasks silently.",
                ],
            },
        ],
        "anger": [
            {
                "title": "The 10-Minute Buffer (Pause Before Responding)",
                "duration": "10 minutes",
                "description": "Prevents reactive regret by allowing high cortisol and adrenaline spikes to settle.",
                "steps": [
                    "Do not hit send or respond to the triggering email or Slack message immediately.",
                    "Step away from your desk, walk down the hall or outside, and take slow belly breaths.",
                ],
            },
            {
                "title": "Bilateral Tapping / Somatic Discharge",
                "duration": "3 minutes",
                "description": "Cross-body tapping integrates emotional processing and discharges autonomic agitation.",
                "steps": [
                    "Cross your arms over your chest, placing hands on opposite shoulders (Butterfly Hug).",
                    "Tap left, then right, in a slow rhythmic 1-second cadence while exhaling deeply.",
                    "Repeat for 20 taps until your breathing rhythm slows.",
                ],
            },
            {
                "title": "Separate objective facts from emotional friction",
                "duration": "5 minutes",
                "description": "Clarify what actually occurred versus what your mind is projecting.",
                "steps": [
                    "Write down only what a video camera would record (the exact facts).",
                    "In a second column, identify the underlying boundary or value that feels crossed.",
                ],
            },
            {
                "title": "Nonviolent Communication Script (Observation & Need)",
                "duration": "5 minutes",
                "description": "Converts intense frustration into constructive, actionable boundaries.",
                "steps": [
                    "Draft your response: 'When [Fact] occurred, it made it difficult to [Need]. Can we align on [Request]?'",
                    "Review it with a calm lens before delivering.",
                ],
            },
            {
                "title": "Cold Water Forehead & Wrist Immersion",
                "duration": "2 minutes",
                "description": "Triggers rapid vasoconstriction and dampens sympathetic fight-or-flight energy.",
                "steps": [
                    "Run cold water over both wrists for 30 seconds.",
                    "Lightly press cool water against your temples and back of neck.",
                ],
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
                "title": "Name the specific boundary concern",
                "duration": "5 minutes",
                "description": "Turn a strong aversion into a clear, workable ethical or quality observation.",
                "steps": ["Write down what happened without judging yourself for your reaction.", "Identify one change or boundary that would restore your equilibrium."],
            },
            {
                "title": "Choose a constructive next move",
                "duration": "Flexible",
                "description": "Decide whether to address it directly, set a boundary, or consult a mentor.",
                "steps": ["Consider what constructive outcome you want before responding.", "If safe, explain the specific observation and request an explicit adjustment."],
            },
        ],
        "surprise": [
            {
                "title": "Give the news a moment to land",
                "duration": "2 minutes",
                "description": "You do not have to decide what unexpected information means immediately.",
                "steps": ["Take a slow breath and notice your first physical reaction.", "Write one sentence about what you know for sure so far."],
            },
            {
                "title": "Separate what is known from what is speculative",
                "duration": "5 minutes",
                "description": "A structured list makes rapid change and unexpected updates manageable.",
                "steps": ["Write down what is verified and what remains an assumption.", "Pick one concrete question to clarify with your team."],
            },
            {
                "title": "Talk it through with a calm colleague",
                "duration": "A few minutes",
                "description": "A trusted perspective helps process a sudden change without panic.",
                "steps": ["Tell someone what surprised you.", "Ask them to help you sanity check the most logical next step."],
            },
        ],
        "joy": [
            {
                "title": "Neuroplastic Savoring (20-Second Hold)",
                "duration": "2 minutes",
                "description": "Holding a positive feeling in conscious awareness physically strengthens resilient neural circuits.",
                "steps": [
                    "Recall the specific detail of the win, compliment, or breakthrough.",
                    "Keep your attention fully on the warm somatic feeling in your chest for 20 continuous seconds.",
                    "Let the sense of competence and gratitude soak deeply into memory.",
                ],
            },
            {
                "title": "Carry the momentum into a focused flow block",
                "duration": "15 minutes",
                "description": "High positive affect expands cognitive flexibility and creative problem-solving.",
                "steps": [
                    "Pick one challenging, high-impact task you have been putting off.",
                    "Ride the wave of positive momentum to initiate a focused work sprint.",
                ],
            },
            {
                "title": "Send a genuine note of appreciation",
                "duration": "3 minutes",
                "description": "Social contagion: sharing gratitude multiplies positive affect across your team.",
                "steps": [
                    "Identify one colleague or mentor who helped make this moment possible.",
                    "Send a short, specific message acknowledging their unique contribution.",
                ],
            },
            {
                "title": "Capture a Milestone Anchor in Your Journal",
                "duration": "3 minutes",
                "description": "Documenting achievements creates a permanent resilience reservoir for stressful weeks.",
                "steps": [
                    "Write down: What happened, what skill I used, and why it matters.",
                    "Tag it as a milestone to celebrate at your next reflection check-in.",
                ],
            },
            {
                "title": "Pay It Forward (Micro-Mentorship)",
                "duration": "5 minutes",
                "description": "Support someone else while your energy and outlook are abundant.",
                "steps": [
                    "Reach out to an onboarding peer or junior teammate with an encouraging note.",
                    "Offer to review a pull request or help unblock a task.",
                ],
            },
        ],
        "fatigue": [
            {
                "title": "Take a 5-Minute Autonomic Reset",
                "duration": "5 minutes",
                "description": "Give yourself a brief pause without needing to earn or justify it.",
                "steps": [
                    "Get a glass of cool water and step away from illuminated screens.",
                    "Stretch, sit quietly with closed eyes, or rest your neck against your chair.",
                ],
            },
            {
                "title": "The 20-20-20 Visual & Circadian Eye Refresh",
                "duration": "2 minutes",
                "description": "Releases ciliary muscle spasms caused by fixed-depth screen gazing.",
                "steps": [
                    "Shift your gaze to an object at least 20 feet away for 20 seconds.",
                    "Blink slowly 10 times to naturally lubricate the cornea and reduce eye fatigue.",
                ],
            },
            {
                "title": "Cellular Hydration & Electrolyte Recharge",
                "duration": "2 minutes",
                "description": "Mild 1-2% dehydration directly mimics and exacerbates cognitive fatigue and headaches.",
                "steps": [
                    "Drink 300ml of room-temperature water with a squeeze of fresh lemon or mineral salt.",
                    "Notice if your energy lift begins within 5 minutes.",
                ],
            },
            {
                "title": "Make the next task lighter (Micro-Chunking)",
                "duration": "5 minutes",
                "description": "Lower the barrier to entry by reducing scope to an effortless starting step.",
                "steps": [
                    "Identify the most critical task on your plate right now.",
                    "Break off a 5-minute piece (e.g., draft the outline only) and give yourself permission to stop thereafter.",
                ],
            },
            {
                "title": "Brief 10-Minute Outdoor Daylight Walk",
                "duration": "10 minutes",
                "description": "Natural sunlight hits retinal ganglion cells, signaling the brain to clear adenosine.",
                "steps": [
                    "Step outside for a brisk loop around your building or block.",
                    "Leave headphones behind to allow involuntary attention to wander and recharge.",
                ],
            },
            {
                "title": "Negotiate timelines or ask for a hand",
                "duration": "A few minutes",
                "description": "Naming tired states early prevents errors and deep exhaustion.",
                "steps": [
                    "Let your team know: 'Energy is a bit low this afternoon; focusing on async reviews.'",
                    "Reschedule optional meetings to protect your recovery.",
                ],
            },
        ],
        "neutral": [
            {
                "title": "Check what you need (Somatic Alignment Check)",
                "duration": "2 minutes",
                "description": "A quick bodily check-in reveals the most effective next intervention.",
                "steps": [
                    "Pause and ask: 'Do I need hydration, physical movement, deep focus, or quiet?'",
                    "Take one decisive, small step to honor that specific need.",
                ],
            },
            {
                "title": "Pick one high-impact priority (MIT Focus)",
                "duration": "5 minutes",
                "description": "Channel emotional calm into pristine, unhurried deep work.",
                "steps": [
                    "Identify your single Most Important Task (MIT) for the remainder of today.",
                    "Close extraneous browser tabs, turn notifications on do-not-disturb, and begin.",
                ],
            },
            {
                "title": "Refresh your routine & workspace ergonomics",
                "duration": "5 minutes",
                "description": "Physical organization directly declutters working memory and reduces subconscious friction.",
                "steps": [
                    "Clear loose mugs, paper, and desktop clutter from your field of view.",
                    "Adjust your chair height so elbows and knees are at 90-degree angles.",
                ],
            },
            {
                "title": "3-Minute Browser & Cognitive Tab Cleanse",
                "duration": "3 minutes",
                "description": "Close open loops that silently drain executive attention.",
                "steps": [
                    "Bookmark and close all background browser tabs you aren't using right now.",
                    "Jot any nagging reminders into a quick capture list so your brain doesn't have to hold them.",
                ],
            },
            {
                "title": "Set an Equanimous Daily Intention",
                "duration": "2 minutes",
                "description": "Anchors your focus in calm capability rather than reactive rushing.",
                "steps": [
                    "Choose an anchor word for the afternoon: 'Patience', 'Clarity', 'Steady', or 'Ease'.",
                    "Return to this word whenever urgent requests ping your notifications.",
                ],
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
                "In the United States and its territories, call or text 988 for the 988 Lifeline. Elsewhere, find local support at https://findahelpline.com/.",
                "In the United States, text CONNECT to 741741 to reach Crisis Text Line.",
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
                "Some wording here can be associated with stress or worry. This text-based cue is not an assessment; "
                "if it fits how you feel, choose one small support step."
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
                "The wording in this entry leans negative, but text scores can miss context and mixed feelings. "
                "If this matches how you feel, a small pause or reaching out may help."
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
                "Some frustration-related words appeared. If that fits, a pause may give you room "
                "to decide how you want to respond."
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
                "Some positive words appeared. If they reflect your experience, take a moment "
                "to notice what felt meaningful to you."
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
            "No strong positive or negative text signal appeared. That does not tell us your mood; "
            "choose whichever next step fits you."
        ),
        "action_items": [
            "Maintain regular hydration and eye-rest intervals (20-20-20 rule).",
            "Plan priorities for tomorrow's morning cycle before wrapping up today.",
            "Take a moment of mindfulness to ground your focus.",
        ],
        "action_paths": _action_paths_for_emotion(dominant_emotion),
        "is_crisis": False,
    }


class _FallbackVaderAnalyzer:
    """Zero-dependency heuristic sentiment scorer used if both NLTK VADER and vaderSentiment are absent."""

    _POS_WORDS = frozenset({
        "good", "great", "excellent", "amazing", "wonderful", "happy", "joy", "peaceful",
        "calm", "radiant", "loved", "grateful", "energized", "strong", "best", "positive",
        "success", "glad", "delighted", "inspired", "healthy", "thriving", "safe"
    })
    _NEG_WORDS = frozenset({
        "bad", "terrible", "awful", "horrible", "sad", "depressed", "anxious", "stress",
        "stressed", "tired", "exhausted", "fatigue", "overwhelmed", "angry", "hate", "pain",
        "hurt", "worst", "negative", "failure", "hopeless", "frustrated", "burnout", "burnt"
    })

    @property
    def lexicon(self) -> dict[str, float]:
        lex = {w: 2.0 for w in self._POS_WORDS}
        lex.update({w: -2.0 for w in self._NEG_WORDS})
        lex["bad"] = -2.5
        lex["terrible"] = -3.0
        lex["good"] = 1.9
        lex["great"] = 3.1
        lex["sad"] = -2.1
        lex["happy"] = 2.7
        return lex

    def polarity_scores(self, text: str) -> dict[str, float]:
        tokens = re.findall(r"[a-z']+", text.lower())
        if not tokens:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        pos_count = 0
        neg_count = 0
        negation_tokens = {"not", "no", "never", "n't", "cannot", "hardly", "barely"}

        for i, t in enumerate(tokens):
            is_negated = any(tokens[max(0, i - 2):i] and tok in negation_tokens for tok in tokens[max(0, i - 2):i])
            if t in self._POS_WORDS:
                if is_negated:
                    neg_count += 1
                else:
                    pos_count += 1
            elif t in self._NEG_WORDS:
                if is_negated:
                    pos_count += 0.5
                else:
                    neg_count += 1

        total = pos_count + neg_count
        if total == 0:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        diff = pos_count - neg_count
        compound = max(-1.0, min(1.0, diff / (abs(diff) + 1.0)))
        pos_ratio = round(pos_count / len(tokens), 3)
        neg_ratio = round(neg_count / len(tokens), 3)
        neu_ratio = round(max(0.0, 1.0 - (pos_ratio + neg_ratio)), 3)
        return {"neg": neg_ratio, "neu": neu_ratio, "pos": pos_ratio, "compound": round(compound, 4)}


_VADER_INSTANCE: Any = None


def _vader_analyzer() -> Any:
    """Use bundled VADER data in hosted mode; preserve the NLTK implementation locally."""
    global _VADER_INSTANCE
    if _VADER_INSTANCE is not None:
        return _VADER_INSTANCE
    if os.getenv("WELLNESS_BUNDLED_VADER") == "1":
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as BundledVader

            _VADER_INSTANCE = BundledVader()
            return _VADER_INSTANCE
        except ImportError:
            pass
    try:
        instance = SentimentIntensityAnalyzer()
        # Verify that lexicon data is actually present and functional
        instance.polarity_scores("health check")
        _VADER_INSTANCE = instance
        return _VADER_INSTANCE
    except Exception:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as BundledVader

            _VADER_INSTANCE = BundledVader()
            return _VADER_INSTANCE
        except ImportError:
            _VADER_INSTANCE = _FallbackVaderAnalyzer()
            return _VADER_INSTANCE


def warm_up_emotion_pipeline() -> None:
    """Eagerly initialize and warm up the transformer classifier in memory."""
    try:
        clf = _get_emotion_pipeline()
        if clf is not None:
            clf("System pipeline warm-up ping.")
    except Exception:
        pass


def analyze_text(
    text: str,
    source_name: str = "direct text",
    *,
    emotion_classifier: EmotionClassifier | None = None,
) -> AnalysisReport:
    """Analyze text and return a typed AnalysisReport with strict thresholds and emotion telemetry."""
    original = validate_text(text)
    validate_nltk_resources()

    # Preprocessing: VADER receives sanitized text preserving caps, punctuation, and negations
    vader_input = sanitize_text_for_vader(original)
    cleaned, tokens = preprocess(original)

    # VADER Polarity Scoring
    scores = _vader_analyzer().polarity_scores(vader_input)
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
    classifier = emotion_classifier or classify_emotions_with_method
    emotion_scores, dominant_emotion, emotion_method = classifier(original)

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
