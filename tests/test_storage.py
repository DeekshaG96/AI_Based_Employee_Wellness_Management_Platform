from pathlib import Path

import pytest

from wellness.errors import InputValidationError
from wellness.storage import (
    authenticate_user,
    connect_database,
    delete_mood_entry,
    get_all_database_entities,
    get_user_mood_entries,
    register_user,
    save_audit_trail,
    save_journal_entry,
    save_mood_entry,
    save_recommendation,
    save_report,
    save_user_feedback,
)


def test_connect_database_initializes_tables(tmp_path: Path):
    connection = connect_database(tmp_path / "wellness.db")
    tables = {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    connection.close()
    assert "users" in tables
    assert "mood_entries" in tables


def test_register_and_authenticate_user(tmp_path: Path):
    connection = connect_database(tmp_path / "wellness.db")
    register_user(connection, "Ada Lovelace", "ADA@example.com", "securepass123")
    auth_success = authenticate_user(connection, "ada@example.com", "securepass123")
    assert auth_success is not None
    assert auth_success[1] == "Ada Lovelace"
    assert authenticate_user(connection, "ada@example.com", "wrongpass") is None
    connection.close()


def test_registration_validates_input(tmp_path: Path):
    connection = connect_database(tmp_path / "wellness.db")
    with pytest.raises(InputValidationError, match="6 characters"):
        register_user(connection, "Ada", "ada@example.com", "123")
    connection.close()


def test_save_and_get_mood_entries_with_emotion_telemetry(tmp_path: Path):
    connection = connect_database(tmp_path / "wellness.db")
    register_user(connection, "User One", "user1@example.com", "password123")
    user = authenticate_user(connection, "user1@example.com", "password123")
    assert user is not None
    user_id = user[0]

    entry_id = save_mood_entry(
        connection=connection,
        user_id=user_id,
        source_type="text",
        source_name="direct text",
        original_text="I feel happy and productive.",
        cleaned_text="i feel happy and productive",
        tokens=("feel", "happy", "productive"),
        sentiment_label="positive",
        neg_score=0.0,
        neu_score=0.3,
        pos_score=0.7,
        compound_score=0.65,
        dominant_emotion="joy",
        emotion_scores={"joy": 0.85, "neutral": 0.15},
        recommendation={"title": "Keep up the momentum"},
    )
    assert entry_id > 0

    entries = get_user_mood_entries(connection, user_id)
    assert len(entries) == 1
    assert entries[0]["sentiment_label"] == "positive"
    assert entries[0]["compound_score"] == 0.65
    assert entries[0]["dominant_emotion"] == "joy"
    assert "happy" in entries[0]["tokens"]

    # Test delete
    assert delete_mood_entry(connection, entry_id, user_id) is True
    assert len(get_user_mood_entries(connection, user_id)) == 0
    connection.close()


def test_six_database_entities_stored_and_retrieved(tmp_path: Path):
    """Test full persistence and retrieval across all 6 verified database entities:
    1. Users info (user_id, profile_details)
    2. Journal entries (journal text)
    3. Recommendations (recommendation id, user id, report data)
    4. Audit trails (user id, username, journal id, recommendation)
    5. User feedback (recommendation id, report data, accepted/rejected)
    6. Reports (report id, user id, username, timestamp)
    """
    connection = connect_database(tmp_path / "wellness_test_6entities.db")

    # 1. Users info
    uid = register_user(connection, "Dev User", "dev@example.com", "mypassword123")
    assert uid > 0

    # 2. Journal entries
    jid = save_journal_entry(connection, uid, "Sprint retrospective was positive and energetic.")
    assert jid > 0

    # 3. Recommendations
    rid = save_recommendation(connection, uid, {"title": "Celebration Reflection", "urgency": "Low"})
    assert rid > 0

    # 4. Audit trails
    aid = save_audit_trail(connection, uid, "Dev User", jid, "Celebration Reflection")
    assert aid > 0

    # 5. User feedback
    fid = save_user_feedback(connection, rid, {"practice": "Gratitude Anchor"}, "accepted")
    assert fid > 0

    # 6. Reports
    repid = save_report(connection, uid, "Dev User", {"dominant_emotion": "joy", "compound": 0.72})
    assert repid > 0

    # Retrieve and verify all 6 entities
    entities = get_all_database_entities(connection)
    assert len(entities["users"]) >= 1
    assert entities["users"][0]["name"] == "Dev User"

    assert len(entities["journal_entries"]) >= 1
    assert "Sprint retrospective" in entities["journal_entries"][0]["journal_text"]

    assert len(entities["recommendations"]) >= 1
    assert entities["recommendations"][0]["user_id"] == uid

    assert len(entities["audit_trails"]) >= 1
    assert entities["audit_trails"][0]["username"] == "Dev User"
    assert entities["audit_trails"][0]["journal_id"] == jid

    assert len(entities["user_feedback"]) >= 1
    assert entities["user_feedback"][0]["accepted_or_rejected"] == "accepted"

    assert len(entities["reports"]) >= 1
    assert entities["reports"][0]["username"] == "Dev User"

    # Privacy-filtered retrieval masks raw reflection text
    redacted = get_all_database_entities(connection, redact_private_text=True)
    assert "Sprint retrospective" not in redacted["journal_entries"][0]["journal_text"]
    assert "Redacted" in redacted["journal_entries"][0]["journal_text"]

    connection.close()


def test_storage_roles_and_genuine_analytics(tmp_path: Path, monkeypatch):
    from wellness import api_client
    from wellness.storage import connect_database, get_admin_analytics, register_user, set_user_role

    db_path = tmp_path / "rbac.db"
    monkeypatch.setattr("wellness.api_client.is_api_available", lambda: False)
    monkeypatch.setattr("wellness.storage.connect_database", lambda p=db_path: connect_database(db_path))

    conn = connect_database(db_path)
    empty_stats = get_admin_analytics(conn)
    assert empty_stats["total_registered_users"] == 0
    assert empty_stats["active_users"] == 0
    assert empty_stats["acceptance_rate"] is None

    register_user(conn, "Emp User", "emp@company.com", "secretpass1")
    conn.close()

    # Wrong password or nonexistent email must fail with ApiError (no fallback login bypass)
    with pytest.raises(api_client.ApiError):
        api_client.login("emp@company.com", "wrong-password")
    with pytest.raises(api_client.ApiError):
        api_client.login("nobody@company.com", "anypass123")

    emp_auth = api_client.login("emp@company.com", "secretpass1")
    assert emp_auth["user"]["role"] == "employee"
    with pytest.raises(api_client.ApiError):
        api_client.get_admin_analytics(emp_auth["access_token"])
    with pytest.raises(api_client.ApiError):
        api_client.get_database_entities(emp_auth["access_token"])

    conn2 = connect_database(db_path)
    assert set_user_role(conn2, "emp@company.com", "admin") is True
    conn2.close()

    adm_auth = api_client.login("emp@company.com", "secretpass1")
    assert adm_auth["user"]["role"] == "admin"
    adm_stats = api_client.get_admin_analytics(adm_auth["access_token"])
    assert adm_stats["total_registered_users"] == 1
    assert adm_stats["acceptance_rate"] is None


def test_api_telemetry_summary_and_delete_all_user_entries(tmp_path, monkeypatch):
    from wellness import api_client
    from wellness.storage import (
        connect_database,
        delete_user_mood_entries,
        get_admin_analytics,
        get_api_telemetry_summary,
        get_user_mood_entries,
        register_user,
        save_api_log,
        save_mood_entry,
        set_user_role,
    )

    db_path = tmp_path / "telemetry_delete.db"
    monkeypatch.setenv("MOODMENTOR_DB_PATH", str(db_path))
    monkeypatch.setattr("wellness.api_client.is_api_available", lambda: False)
    monkeypatch.setattr("wellness.storage.connect_database", lambda p=db_path: connect_database(db_path))

    conn = connect_database(db_path)
    uid_a = register_user(conn, "Alice", "alice@company.com", "secretpass1")
    uid_b = register_user(conn, "Bob", "bob@company.com", "secretpass1")

    # Empty telemetry reports clean No data state without fake numbers
    empty_tele = get_api_telemetry_summary(conn)
    assert empty_tele["endpoints"] == []
    assert get_admin_analytics(conn)["api_performance"]["avg_response_time_ms"] is None

    # Save reflections for both users
    for uid in (uid_a, uid_b):
        save_mood_entry(
            connection=conn,
            user_id=uid,
            source_type="text",
            source_name="Journal",
            original_text="Calm and focused day",
            cleaned_text="calm focused day",
            tokens=["calm", "focused", "day"],
            sentiment_label="positive",
            neg_score=0.0,
            neu_score=0.4,
            pos_score=0.6,
            compound_score=0.65,
            dominant_emotion="joy",
            emotion_scores={"joy": 0.9},
            recommendation={"title": "Keep journaling"},
        )

    save_api_log(conn, "POST", "/analyze", 200, 14.5)
    save_api_log(conn, "GET", "/history/1", 200, 3.5)
    tele = get_api_telemetry_summary(conn)
    assert len(tele["endpoints"]) == 2
    assert get_admin_analytics(conn)["api_performance"]["avg_response_time_ms"] == 9.0

    # Deleting all of Alice's entries leaves Bob's entries untouched
    deleted = delete_user_mood_entries(conn, uid_a)
    assert deleted == 1
    assert len(get_user_mood_entries(conn, uid_a)) == 0
    assert len(get_user_mood_entries(conn, uid_b)) == 1
    conn.close()

    # Verify api_client fallback for delete_all_user_entries and role update
    set_user_role(connect_database(db_path), "alice@company.com", "admin")
    adm_auth = api_client.login("alice@company.com", "secretpass1")
    updated = api_client.update_user_role_client(uid_b, "admin", adm_auth["access_token"])
    assert updated["role"] == "admin"
    assert api_client.delete_all_user_entries(uid_b, adm_auth["access_token"]) == 1
    conn_check = connect_database(db_path)
    assert len(get_user_mood_entries(conn_check, uid_b)) == 0
    conn_check.close()

