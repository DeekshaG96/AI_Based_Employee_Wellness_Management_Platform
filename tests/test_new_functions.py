"""Unit tests for newly added functions: Model Comparator, Wellness Tools, and Team Analytics."""

import pytest
from fastapi.testclient import TestClient

from wellness.comparator import compare_models, run_lexical_baseline, run_transformer_analysis
from wellness.team_analytics import calculate_team_burnout_risk, generate_synthetic_team_data
from wellness.wellness_tools import (
    get_cognitive_reframes,
    get_guided_exercises,
    render_box_breathing_widget,
)
from wellness_api.main import app


def test_lexical_baseline_identifies_polarity_and_keywords():
    result = run_lexical_baseline("I am ecstatic and joyful about this fantastic promotion!")
    assert result["polarity"] == "positive"
    assert result["compound_score"] > 0.5
    assert result["dominant_emotion"] in ("joy", "unclassified")
    assert "vader_scores" in result


def test_transformer_analysis_returns_probabilities():
    result = run_transformer_analysis("I feel nervous about the upcoming milestone evaluation.")
    assert "dominant_emotion" in result
    assert "top_emotions" in result
    assert len(result["top_emotions"]) > 0
    assert "all_scores" in result


def test_compare_models_synthesizes_insight():
    text = "I love collaborating with my team, but this extreme workload is exhausting and stressful."
    comparison = compare_models(text)

    assert "lexical_baseline" in comparison
    assert "transformer_model" in comparison
    assert "comparative_analysis" in comparison

    analysis = comparison["comparative_analysis"]
    assert "is_mixed_emotion" in analysis
    assert "insight" in analysis
    assert len(analysis["insight"]) > 10


def test_guided_exercises_catalog():
    exercises = get_guided_exercises()
    assert len(exercises) >= 4
    exercise_ids = {e["id"] for e in exercises}
    assert "box_breathing" in exercise_ids
    assert "relaxing_breath_478" in exercise_ids
    assert "physiological_sigh" in exercise_ids
    assert "grounding_54321" in exercise_ids

    # Verify widget HTML
    widget_html = render_box_breathing_widget()
    assert "boxBreath" in widget_html
    assert "4-4-4-4" in widget_html


def test_cognitive_reframing_catalog():
    fear_reframe = get_cognitive_reframes("fear")
    assert "reframing_prompt" in fear_reframe
    assert "self_inquiry_questions" in fear_reframe
    assert len(fear_reframe["self_inquiry_questions"]) >= 3
    assert "affirmation" in fear_reframe

    # Test unknown emotion uses fallback
    unknown_reframe = get_cognitive_reframes("quantum_superposition")
    assert "reframing_prompt" in unknown_reframe
    assert "affirmation" in unknown_reframe


def test_team_burnout_risk_metrics():
    # Empty case
    empty_result = calculate_team_burnout_risk([])
    assert empty_result["sample_size"] == 0
    assert empty_result["risk_level"] == "Insufficient Data"

    # Synthetic burnout data
    burnout_data = generate_synthetic_team_data(days=7, team_size=5, scenario="burnout")
    burnout_result = calculate_team_burnout_risk(burnout_data)
    assert burnout_result["sample_size"] == 35
    assert burnout_result["burnout_risk_score"] > 40.0
    assert burnout_result["distress_ratio"] > 0.3

    # Synthetic thriving data
    thriving_data = generate_synthetic_team_data(days=7, team_size=5, scenario="thriving")
    thriving_result = calculate_team_burnout_risk(thriving_data)
    assert thriving_result["burnout_risk_score"] < burnout_result["burnout_risk_score"]
    assert thriving_result["team_resilience_index"] > burnout_result["team_resilience_index"]


def test_api_new_endpoints():
    client = TestClient(app)

    # 1. Compare models endpoint
    res_comp = client.post("/analyze/compare", json={"text": "I feel grateful for the help."})
    assert res_comp.status_code == 200
    data_comp = res_comp.json()
    assert "lexical_baseline" in data_comp
    assert "transformer_model" in data_comp
    assert "comparative_analysis" in data_comp

    # 2. Exercises endpoint
    res_ex = client.get("/wellness/exercises")
    assert res_ex.status_code == 200
    assert len(res_ex.json()) >= 4

    # 3. Reframing endpoint
    res_ref = client.get("/wellness/reframes/anger")
    assert res_ref.status_code == 200
    assert "reframing_prompt" in res_ref.json()

    # 4. Team Burnout Risk endpoint
    res_team = client.post("/team/burnout-risk", json={"scenario": "thriving", "days": 7, "team_size": 4})
    assert res_team.status_code == 200
    assert "burnout_risk_score" in res_team.json()
    assert "team_resilience_index" in res_team.json()


def test_f1_benchmarks_exceed_80_and_89_percent():
    import json
    from pathlib import Path

    # Verify Polarity Benchmark: Macro F1 >= 80%, Positive F1 >= 89%
    pol_path = Path("docs/semeval-binary-polarity-89plus.json")
    assert pol_path.exists()
    with open(pol_path, encoding="utf-8") as f:
        pol = json.load(f)
    assert pol["macro_f1_score"] >= 0.80, "Polarity Macro F1 must be >= 80%"
    assert pol["positive_class_metrics"]["f1_score"] >= 0.89, "Positive F1 must be >= 89%"
    assert pol["overall_accuracy"] >= 0.85

    # Verify GoEmotions Core Classes Benchmark: Top-3 Macro F1 >= 80%, Top-4 Macro F1 >= 80%
    core_path = Path("docs/goemotions-core-emotions-80plus.json")
    assert core_path.exists()
    with open(core_path, encoding="utf-8") as f:
        core = json.load(f)
    assert core["top_3_macro_f1"] >= 0.80, "GoEmotions Top-3 Macro F1 must be >= 80%"
    assert core["top_4_macro_f1"] >= 0.80, "GoEmotions Top-4 Macro F1 must be >= 80%"
    assert core["core_emotion_breakdown"]["gratitude"]["f1_score"] >= 0.89, "Gratitude F1 must be >= 89%"


def test_mood_food_planner_recommendations():
    from wellness.food_planner import get_mood_food_plan, get_supported_mood_categories

    # 1. Stress / Fear profile
    plan_fear = get_mood_food_plan("fear", dietary_pref="any")
    assert plan_fear["mood_category"] == "stress_anxiety"
    assert "Magnesium" in " ".join(plan_fear["key_nutrients"])
    assert len(plan_fear["superfoods"]) >= 4
    assert "breakfast" in plan_fear["curated_meals"]
    assert "lunch" in plan_fear["curated_meals"]
    assert "snack" in plan_fear["curated_meals"]
    assert "dinner" in plan_fear["curated_meals"]
    assert len(plan_fear["foods_to_avoid"]) >= 2
    assert len(plan_fear["shopping_essentials"]) >= 3

    # 2. Sadness / Low Mood profile
    plan_sad = get_mood_food_plan("sadness", dietary_pref="vegetarian")
    assert plan_sad["mood_category"] == "low_mood_sadness"
    assert "Tryptophan" in " ".join(plan_sad["key_nutrients"])

    # 3. Vegan substitution check
    plan_vegan = get_mood_food_plan("fear", dietary_pref="vegan")
    assert "salmon" not in plan_vegan["curated_meals"]["dinner"].lower()
    assert "tofu" in plan_vegan["curated_meals"]["dinner"].lower() or "tempeh" in plan_vegan["curated_meals"]["dinner"].lower()

    # 4. Joy / Flourishing profile
    plan_joy = get_mood_food_plan("joy")
    assert plan_joy["mood_category"] == "positive_flourishing"

    # 5. Supported categories listing
    categories = get_supported_mood_categories()
    assert len(categories) >= 5


def test_api_mood_food_endpoints():
    client = TestClient(app)

    # 1. GET /wellness/food-plan/{emotion}
    res_get = client.get("/wellness/food-plan/fear?dietary_pref=vegan")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["emotion_detected"] == "fear"
    assert data_get["dietary_preference"] == "vegan"
    assert "superfoods" in data_get
    assert "curated_meals" in data_get

    # 2. POST /wellness/food-plan
    res_post = client.post("/wellness/food-plan", json={"emotion": "sadness", "dietary_preference": "vegetarian"})
    assert res_post.status_code == 200
    data_post = res_post.json()
    assert data_post["emotion_detected"] == "sadness"
    assert data_post["dietary_preference"] == "vegetarian"

    # 3. GET /wellness/food-plan-categories
    res_cats = client.get("/wellness/food-plan-categories")
    assert res_cats.status_code == 200
    assert len(res_cats.json()) >= 5



