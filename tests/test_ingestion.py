"""Unit tests for ingestion engine (.txt, .csv, and direct text)."""

import pytest
from src.ingestion import TextIngestionEngine

def test_raw_text_ingestion():
    res = TextIngestionEngine.ingest_raw_text("Great work today!")
    assert res["success"] is True
    assert len(res["entries"]) == 1

def test_empty_raw_text():
    res = TextIngestionEngine.ingest_raw_text("   ")
    assert res["success"] is False
    assert "empty" in res["error"]

def test_txt_file_ingestion():
    content = "Line 1 feedback\nLine 2 feedback\n\nLine 3 feedback"
    res = TextIngestionEngine.ingest_txt_file(content)
    assert res["success"] is True
    assert len(res["entries"]) == 3

def test_csv_file_ingestion():
    csv_str = "employee_id,feedback_text\nE1,Great culture\nE2,Heavy stress"
    res = TextIngestionEngine.ingest_csv_file(csv_str)
    assert res["success"] is True
    assert res["total_rows"] == 2
    assert res["selected_column"] == "feedback_text"
