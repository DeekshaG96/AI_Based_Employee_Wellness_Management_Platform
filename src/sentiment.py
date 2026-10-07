"""
VADER Sentiment Analysis Engine for Employee Wellness Management.
Fulfills M1 Deliverables:
- VADER integration
- Positive/Negative/Neutral classification
- Compound score computation & interpretation
- Intensity analysis
"""

from typing import Dict, Any, List
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    """Performs sentiment and emotional tone analysis using VADER."""

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        try:
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Calculates VADER polarity scores and derives sentiment class and intensity.
        Handles empty or invalid input gracefully.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "is_valid": False,
                "error": "Cannot compute sentiment for empty or invalid text.",
                "compound": 0.0,
                "positive": 0.0,
                "neutral": 0.0,
                "negative": 0.0,
                "sentiment": "Neutral",
                "intensity": "None",
                "emotion_tone": "Indeterminate"
            }

        scores = self.sia.polarity_scores(text)
        compound = round(scores["compound"], 4)
        pos = round(scores["pos"], 4)
        neu = round(scores["neu"], 4)
        neg = round(scores["neg"], 4)

        if compound >= self.pos_threshold:
            sentiment = "Positive"
        elif compound <= self.neg_threshold:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Intensity classification
        abs_compound = abs(compound)
        if abs_compound >= 0.6:
            intensity = "High"
        elif abs_compound >= 0.2:
            intensity = "Moderate"
        else:
            intensity = "Low / Neutral"

        # Emotion tone heuristic for employee wellbeing context
        lower = text.lower()
        if any(w in lower for w in ['burnout', 'exhausted', 'overwhelmed', 'fatigue', 'drained']):
            emotion_tone = "Burnout / Exhaustion"
        elif any(w in lower for w in ['stress', 'anxiety', 'worried', 'panic', 'pressure']):
            emotion_tone = "High Stress / Anxiety"
        elif any(w in lower for w in ['frustrat', 'annoy', 'angry', 'unfair', 'confused']):
            emotion_tone = "Frustration / Friction"
        elif any(w in lower for w in ['great', 'happy', 'fantastic', 'love', 'energized', 'upbeat', 'rewarding']):
            emotion_tone = "High Morale / Joy"
        elif any(w in lower for w in ['good', 'support', 'appreciate', 'helpful', 'satisfied', 'calm']):
            emotion_tone = "Satisfaction / Calm"
        else:
            emotion_tone = "Neutral / Routine"

        return {
            "is_valid": True,
            "error": None,
            "compound": compound,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "sentiment": sentiment,
            "intensity": intensity,
            "emotion_tone": emotion_tone
        }

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyzes a collection of text entries."""
        return [self.analyze(t) for t in texts]
