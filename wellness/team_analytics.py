"""Team-level wellness analytics, burnout risk assessment, and resilience indexing.

Aggregates individual wellness signals into enterprise-level health telemetry
while preserving individual employee privacy.
"""

from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta
from typing import Any


def calculate_team_burnout_risk(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate an aggregated Burnout Risk Score (0–100) and Resilience Index.

    Args:
        entries: List of check-in entries with 'sentiment_compound' (float),
                 'dominant_emotion' (str), and optional 'timestamp'.

    Returns:
        Structured team health index with risk level and management recommendations.
    """
    if not entries:
        return {
            "sample_size": 0,
            "burnout_risk_score": 0.0,
            "risk_level": "Insufficient Data",
            "team_resilience_index": 50.0,
            "sentiment_mean": 0.0,
            "sentiment_stddev": 0.0,
            "distress_ratio": 0.0,
            "flourishing_ratio": 0.0,
            "recommendations": ["Gather at least 3 reflection check-ins to compute team trends."],
        }

    compounds = [float(e.get("sentiment_compound", 0.0)) for e in entries]
    emotions = [str(e.get("dominant_emotion", "neutral")).lower() for e in entries]
    n = len(compounds)

    mean_compound = statistics.mean(compounds)
    stddev_compound = statistics.stdev(compounds) if n > 1 else 0.0

    # Categorize emotion families
    distress_labels = {"anger", "annoyance", "disapproval", "disgust", "fear", "nervousness", "sadness", "grief", "disappointment"}
    flourishing_labels = {"joy", "admiration", "optimism", "caring", "excitement", "gratitude", "love", "pride", "relief"}

    distress_count = sum(1 for emo in emotions if emo in distress_labels)
    flourishing_count = sum(1 for emo in emotions if emo in flourishing_labels)

    distress_ratio = distress_count / n
    flourishing_ratio = flourishing_count / n

    # Compute Component Scores (0 to 100)
    # Component 1: Negative Polarity Pressure (more negative = higher risk)
    # Map compound [-1.0, 1.0] to [100, 0]
    polarity_risk = max(0.0, min(100.0, (1.0 - mean_compound) * 50.0))

    # Component 2: Distress-to-Flourishing Emotion Ratio
    emotion_risk = max(0.0, min(100.0, distress_ratio * 100.0))

    # Component 3: Volatility Penalty (erratic swings indicate team friction)
    volatility_risk = max(0.0, min(100.0, stddev_compound * 100.0))

    # Weighted Composite Burnout Risk Score
    composite_burnout_score = round(
        (polarity_risk * 0.40) + (emotion_risk * 0.40) + (volatility_risk * 0.20),
        1,
    )

    # Team Resilience Index (opposite of burnout with buffer)
    resilience_index = round(max(0.0, min(100.0, 100.0 - composite_burnout_score + (flourishing_ratio * 20.0))), 1)

    # Classify Risk Level
    if composite_burnout_score < 30.0:
        risk_level = "Low (Thriving)"
        recs = [
            "Team emotional baseline is positive and stable.",
            "Celebrate recent team wins and sustain balanced workload pacing.",
            "Encourage proactive peer recognition to maintain momentum.",
        ]
    elif composite_burnout_score < 55.0:
        risk_level = "Moderate (Manageable)"
        recs = [
            "Moderate work stress signals detected across several reflections.",
            "Conduct a mid-sprint workload check-in during your next 1-on-1s.",
            "Encourage no-meeting focus blocks to prevent context switching fatigue.",
        ]
    elif composite_burnout_score < 75.0:
        risk_level = "Elevated (At Risk)"
        recs = [
            "Sustained distress signals detected. Review deliverable deadlines and meeting volume.",
            "Actively encourage team members to use designated offline breaks.",
            "Offer access to wellness EAP (Employee Assistance Programs) and supportive resources.",
        ]
    else:
        risk_level = "Severe (Urgent Intervention)"
        recs = [
            "Critical burnout indicators: widespread negative sentiment and acute distress markers.",
            "Schedule an immediate workload reassessment with leadership.",
            "Implement mandatory decompression time and offer professional wellness counseling.",
        ]

    return {
        "sample_size": n,
        "burnout_risk_score": composite_burnout_score,
        "risk_level": risk_level,
        "team_resilience_index": resilience_index,
        "sentiment_mean": round(mean_compound, 3),
        "sentiment_stddev": round(stddev_compound, 3),
        "distress_ratio": round(distress_ratio, 3),
        "flourishing_ratio": round(flourishing_ratio, 3),
        "recommendations": recs,
    }


def generate_synthetic_team_data(days: int = 14, team_size: int = 8, scenario: str = "balanced") -> list[dict[str, Any]]:
    """Generate realistic, privacy-compliant mock team reflection data for demonstrations."""
    random.seed(42)
    now = datetime.now()
    records: list[dict[str, Any]] = []

    scenario_configs = {
        "thriving": {
            "compounds": [0.4, 0.6, 0.8, 0.7, 0.5, 0.9, 0.3],
            "emotions": ["joy", "optimism", "admiration", "gratitude", "excitement", "caring", "neutral"],
        },
        "burnout": {
            "compounds": [-0.6, -0.4, -0.7, -0.8, -0.2, -0.5, 0.1],
            "emotions": ["sadness", "anger", "fear", "annoyance", "disappointment", "nervousness", "neutral"],
        },
        "balanced": {
            "compounds": [0.5, -0.2, 0.3, -0.4, 0.6, 0.1, -0.1, 0.7, -0.3],
            "emotions": ["joy", "nervousness", "optimism", "annoyance", "admiration", "neutral", "fear", "caring"],
        },
    }

    config = scenario_configs.get(scenario, scenario_configs["balanced"])

    for day in range(days):
        date_stamp = (now - timedelta(days=days - day - 1)).strftime("%Y-%m-%d")
        for member_idx in range(team_size):
            compound = random.choice(config["compounds"]) + random.uniform(-0.1, 0.1)
            compound = max(-1.0, min(1.0, compound))
            emotion = random.choice(config["emotions"])

            records.append({
                "date": date_stamp,
                "member_alias": f"Employee {member_idx + 1:02d}",
                "sentiment_compound": round(compound, 3),
                "dominant_emotion": emotion,
            })

    return records

