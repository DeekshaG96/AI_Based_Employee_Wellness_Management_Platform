"""API contract tests run when the optional service dependencies are installed."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from wellness.report import AnalysisReport, InputSummary, SentimentLabel, SentimentScores
from wellness_api import database
from wellness_api.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    database.configure_database(f"sqlite:///{tmp_path / 'api-test.db'}")
    database.create_tables()
    with TestClient(app) as test_client:
        yield test_client


def register(client: TestClient, email: str = "ada@example.com") -> tuple[str, int]:
    response = client.post(
        "/auth/register",
        json={"name": "Ada Lovelace", "email": email, "password": "strongpass123"},
    )
    assert response.status_code == 201, response.text
    data = response.json()
    return data["access_token"], data["user"]["id"]


def test_health_checks_database(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "connected"}


def test_authenticate_analyze_and_history_are_user_scoped(client: TestClient, monkeypatch):
    token, user_id = register(client)
    auth = {"Authorization": f"Bearer {token}"}

    def fake_analyze(text: str, source_name: str) -> AnalysisReport:
        return AnalysisReport(
            input_summary=InputSummary("text", source_name, len(text), 1, 2),
            original_text=text,
            cleaned_text=text.lower(),
            processed_text="feel good",
            tokens=("feel", "good"),
            sentiment_scores=SentimentScores(0.0, 0.2, 0.8, 0.8),
            sentiment_label=SentimentLabel.POSITIVE,
            counts={"characters": len(text), "lines": 1},
            vader_input_text=text,
            dominant_emotion="joy",
            emotion_scores={"joy": 0.9, "neutral": 0.1},
            recommendations={
                "title": "Keep the momentum",
                "category": "Well-Being",
                "urgency": "Low",
                "summary": "Notice what is working.",
                "action_items": ["Record a small win."],
                "is_crisis": False,
            },
        )

    monkeypatch.setattr("wellness_api.main.analyze_text", fake_analyze)
    analyzed = client.post("/analyze", headers=auth, json={"text": "I feel good"})
    assert analyzed.status_code == 200, analyzed.text
    assert analyzed.json()["dominant_emotion"] == "joy"
    assert analyzed.json()["emotion_method"] == "lexical_fallback"
    assert analyzed.json()["recommendations"]["title"] == "Keep the momentum"

    history = client.get(f"/history/{user_id}", headers=auth)
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["original_text"] == "I feel good"

    other_token, _other_id = register(client, "grace@example.com")
    forbidden = client.get(
        f"/history/{user_id}", headers={"Authorization": f"Bearer {other_token}"}
    )
    assert forbidden.status_code == 403

    deleted = client.delete(
        f"/history/{history.json()['items'][0]['id']}", headers={"Authorization": f"Bearer {token}"}
    )
    assert deleted.status_code == 200
    assert client.get(f"/history/{user_id}", headers=auth).json()["total"] == 0


def test_logout_revokes_token(client: TestClient):
    token, _user_id = register(client)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/auth/me", headers=headers).status_code == 200
    assert client.post("/auth/logout", headers=headers).status_code == 200
    assert client.get("/auth/me", headers=headers).status_code == 401
