"""Unit tests for the Therapeutic Yoga Remedies catalog and emotional mapping."""

import pytest
from wellness.yoga import (
    get_all_yoga_remedies,
    get_yoga_categories,
    get_yoga_remedy_for_emotion,
)


def test_get_all_yoga_remedies():
    remedies = get_all_yoga_remedies()
    assert len(remedies) >= 8
    for r in remedies:
        assert "id" in r
        assert "sanskrit_name" in r
        assert "english_name" in r
        assert "somatic_mechanism" in r
        assert "steps_floor" in r
        assert len(r["steps_floor"]) > 0
        assert "office_variation" in r
        assert isinstance(r["desk_friendly"], bool)


def test_get_yoga_categories():
    categories = get_yoga_categories()
    assert len(categories) >= 3
    assert "Stress & Anxiety Relief" in categories
    assert "Deep Recovery & Burnout" in categories


def test_get_yoga_remedy_for_emotion_mappings():
    # Anxiety / Fear -> Balasana (Child's Pose)
    fear_remedy = get_yoga_remedy_for_emotion("fear")
    assert fear_remedy["id"] == "balasana"

    # Burnout / Fatigue -> Viparita Karani
    fatigue_remedy = get_yoga_remedy_for_emotion("fatigue")
    assert fatigue_remedy["id"] == "viparita_karani"

    # Anger / Frustration -> Cat-Cow
    anger_remedy = get_yoga_remedy_for_emotion("anger")
    assert anger_remedy["id"] == "marjaryasana_bitilasana"

    # Unknown fallback
    default_remedy = get_yoga_remedy_for_emotion("unknown_state")
    assert default_remedy["id"] == "balasana"
