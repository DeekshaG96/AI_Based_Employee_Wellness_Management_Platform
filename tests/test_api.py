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


def test_rbac_admin_endpoints_reject_anonymous_and_employees(client: TestClient, monkeypatch):
    # Unauthenticated access must fail across sensitive endpoints
    assert client.get("/admin/analytics").status_code == 401
    assert client.get("/admin/overview").status_code == 401
    assert client.get("/database/entities").status_code == 401
    assert client.post("/team/burnout-risk", json={}).status_code == 401
    assert client.post("/recommend", json={"emotion": "joy"}).status_code == 401
    assert client.get("/recommend/joy").status_code == 401
    assert client.post("/analyze/quick", json={"text": "hello world"}).status_code == 401
    assert client.post("/analyze/compare", json={"text": "hello world"}).status_code == 401

    # Even if client submits extra 'role': 'admin', register must force 'employee'
    reg = client.post(
        "/auth/register",
        json={"name": "Eve", "email": "eve@example.com", "password": "strongpass123", "role": "admin"},
    )
    assert reg.status_code == 201
    assert reg.json()["user"]["role"] == "employee"
    emp_headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    # Employee cannot access Admin endpoints (403 Forbidden)
    assert client.get("/admin/analytics", headers=emp_headers).status_code == 403
    assert client.get("/admin/overview", headers=emp_headers).status_code == 403
    assert client.get("/database/entities", headers=emp_headers).status_code == 403
    assert client.post("/team/burnout-risk", headers=emp_headers, json={}).status_code == 403

    # Employee can call authenticated personal recommendation endpoint
    assert client.get("/recommend/joy", headers=emp_headers).status_code == 200

    # Promote via trusted ADMIN_EMAILS allowlist and verify admin login + privacy redaction
    monkeypatch.setenv("ADMIN_EMAILS", "boss@example.com")
    adm_reg = client.post(
        "/auth/register",
        json={"name": "Boss", "email": "boss@example.com", "password": "strongpass123"},
    )
    assert adm_reg.status_code == 201
    assert adm_reg.json()["user"]["role"] == "admin"
    adm_headers = {"Authorization": f"Bearer {adm_reg.json()['access_token']}"}

    # Check genuine SQL counts without fake 80% / 1-user clamping when empty
    stats = client.get("/admin/analytics", headers=adm_headers)
    assert stats.status_code == 200
    body = stats.json()
    assert body["total_registered_users"] == 2
    assert body["active_users"] == 0
    assert body["acceptance_rate"] is None

    # Create employee entry and verify /database/entities redacts private journal text
    def fake_analyze(text: str, source_name: str) -> AnalysisReport:
        return AnalysisReport(
            input_summary=InputSummary("text", source_name, len(text), 1, 2),
            original_text=text,
            cleaned_text=text.lower(),
            processed_text="secret entry",
            tokens=("secret", "entry"),
            sentiment_scores=SentimentScores(0.0, 1.0, 0.0, 0.0),
            sentiment_label=SentimentLabel.NEUTRAL,
            counts={"characters": len(text), "lines": 1},
            vader_input_text=text,
            dominant_emotion="neutral",
            emotion_scores={"neutral": 1.0},
            recommendations={"title": "Take a mindful pause", "category": "Well-Being", "urgency": "Low"},
        )

    monkeypatch.setattr("wellness_api.main.analyze_text", fake_analyze)
    assert client.post("/analyze", headers=emp_headers, json={"text": "Confidential manager stress secret"}).status_code == 200

    entities = client.get("/database/entities", headers=adm_headers)
    assert entities.status_code == 200
    ent_json = entities.json()
    assert len(ent_json["journal_entries"]) == 1
    assert "Confidential manager stress secret" not in ent_json["journal_entries"][0]["journal_text"]
    assert "Redacted" in ent_json["journal_entries"][0]["journal_text"]
    assert ent_json["reports"][0]["report_data"].get("original_text") == "[Redacted]"

    overview = client.get("/admin/overview", headers=adm_headers)
    assert overview.status_code == 200
    assert len(overview.json()["records"]) == 1


def test_production_db_failure_refuses_sqlite_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("wellness_api.database._is_test_runtime", lambda: False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://bad:bad@127.0.0.1:1/bad_db")
    database.configure_database("postgresql+psycopg://bad:bad@127.0.0.1:1/bad_db")

    from sqlalchemy.exc import SQLAlchemyError

    try:
        with pytest.raises(SQLAlchemyError):
            database.create_tables()

        assert database.engine.dialect.name != "sqlite"
        assert database.get_database_status()["fallback_active"] is False
        assert database.get_database_status()["backend"] == "unavailable"
    finally:
        database.configure_database(f"sqlite:///{tmp_path / 'cleanup.db'}")


def test_admin_telemetry_role_update_and_structured_audit_trails(client: TestClient, monkeypatch):
    def fake_analyze(text: str, source_name: str) -> AnalysisReport:
        return AnalysisReport(
            input_summary=InputSummary("text", source_name, len(text), 1, 2),
            original_text=text,
            cleaned_text=text.lower(),
            processed_text="clear progress",
            tokens=("clear", "progress"),
            sentiment_scores=SentimentScores(0.0, 0.4, 0.6, 0.65),
            sentiment_label=SentimentLabel.POSITIVE,
            counts={"characters": len(text), "lines": 1},
            vader_input_text=text,
            dominant_emotion="joy",
            emotion_scores={"joy": 0.9},
            recommendations={"title": "Gratitude Walk", "category": "Well-Being", "urgency": "Low"},
        )

    monkeypatch.setattr("wellness_api.main.analyze_text", fake_analyze)

    emp_res = client.post(
        "/auth/register",
        json={"name": "Employee One", "email": "emp1@company.com", "password": "password123"},
    )
    assert emp_res.status_code == 201
    emp_token = emp_res.json()["access_token"]
    emp_id = emp_res.json()["user"]["id"]
    emp_headers = {"Authorization": f"Bearer {emp_token}"}

    # Employees are forbidden from admin telemetry and role endpoints
    assert client.get("/admin/telemetry", headers=emp_headers).status_code == 403
    assert (
        client.post(
            f"/admin/users/{emp_id}/role",
            headers=emp_headers,
            json={"role": "admin"},
        ).status_code
        == 403
    )

    # Admin registered via trusted ADMIN_EMAILS allowlist then logs in
    monkeypatch.setenv("ADMIN_EMAILS", "adm2@company.com")
    adm_reg = client.post(
        "/auth/register",
        json={"name": "Admin Two", "email": "adm2@company.com", "password": "password123"},
    )
    assert adm_reg.status_code == 201
    adm_id = adm_reg.json()["user"]["id"]

    adm_login = client.post(
        "/auth/login",
        json={"email": "adm2@company.com", "password": "password123"},
    )
    assert adm_login.status_code == 200
    adm_headers = {"Authorization": f"Bearer {adm_login.json()['access_token']}"}

    # Analyze & give feedback -> generates structured audit trail events & api_logs
    ana = client.post(
        "/analyze",
        headers=adm_headers,
        json={"text": "Grateful for clear team coordination and steady progress today.", "source_name": "Work"},
    )
    assert ana.status_code == 200
    fb = client.post(
        "/feedback",
        headers=adm_headers,
        json={"recommendation_id": None, "report_data": {"title": "Gratitude Walk"}, "accepted_or_rejected": "accepted"},
    )
    assert fb.status_code == 200

    # Admin promotes employee user, but cannot demote self
    role_upd = client.post(
        f"/admin/users/{emp_id}/role",
        headers=adm_headers,
        json={"role": "admin"},
    )
    assert role_upd.status_code == 200
    assert role_upd.json()["role"] == "admin"

    self_demote = client.post(
        f"/admin/users/{adm_id}/role",
        headers=adm_headers,
        json={"role": "employee"},
    )
    assert self_demote.status_code == 400

    # Real per-endpoint telemetry reflects logged invocations
    telemetry = client.get("/admin/telemetry", headers=adm_headers)
    assert telemetry.status_code == 200
    endpoints = telemetry.json()["endpoints"]
    assert any(row["endpoint"] == "/analyze" and row["invocations"] >= 1 for row in endpoints)
    assert all(row["avg_latency_ms"] >= 0.0 for row in endpoints)

    # Audit trails include structured action codes
    entities = client.get("/database/entities", headers=adm_headers)
    assert entities.status_code == 200
    actions = {item["action"] for item in entities.json()["audit_trails"]}
    assert {"USER_REGISTER", "ADMIN_LOGIN", "JOURNAL_ANALYSIS", "RECOMMENDATION_ACCEPT", "ADMIN_ROLE_UPDATE"}.issubset(actions)
