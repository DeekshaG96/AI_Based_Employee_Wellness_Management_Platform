"""
FastAPI REST API endpoints for Employee Wellness Management Platform.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocessing import TextPreprocessor
from src.sentiment import SentimentAnalyzer
from src.ingestion import TextIngestionEngine
from src.recommender import WellnessRecommender
from src.reporting import ReportGenerator

app = FastAPI(
    title="Infosys AI-Based Employee Wellness Management API",
    description="REST API for text preprocessing, VADER sentiment analysis, and wellness recommendations",
    version="1.0.0"
)

preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer()
recommender = WellnessRecommender()
reporter = ReportGenerator()

class FeedbackRequest(BaseModel):
    text: str

class BatchFeedbackRequest(BaseModel):
    texts: List[str]

@app.get("/")
def root():
    return {
        "project": "Infosys AI-Based Employee Wellness Management Platform",
        "milestone": "M1",
        "status": "Online",
        "endpoints": ["/api/v1/preprocess", "/api/v1/analyze", "/api/v1/batch-analyze", "/api/v1/upload-txt"]
    }

@app.post("/api/v1/preprocess")
def preprocess_text(payload: FeedbackRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    result = preprocessor.process(payload.text)
    if not result["is_valid"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/api/v1/analyze")
def analyze_single_feedback(payload: FeedbackRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    prep = preprocessor.process(payload.text)
    sent = analyzer.analyze(payload.text)
    recs = recommender.recommend(sent["sentiment"], sent["emotion_tone"])
    return {
        "raw_text": payload.text,
        "preprocessing": prep,
        "sentiment": sent,
        "recommendations": recs
    }

@app.post("/api/v1/batch-analyze")
def batch_analyze_feedback(payload: BatchFeedbackRequest):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="List of texts cannot be empty.")
    results = []
    for idx, t in enumerate(payload.texts):
        if t and t.strip():
            prep = preprocessor.process(t)
            sent = analyzer.analyze(t)
            results.append({
                "id": idx + 1,
                "text": t,
                "sentiment": sent["sentiment"],
                "compound": sent["compound"],
                "positive": sent["positive"],
                "neutral": sent["neutral"],
                "negative": sent["negative"],
                "emotion_tone": sent["emotion_tone"],
                "lemmatized_tokens": prep.get("lemmatized_tokens", [])
            })
    summary = reporter.generate_summary(results)
    return {
        "summary": summary,
        "results": results
    }

@app.post("/api/v1/upload-txt")
async def upload_txt(file: UploadFile = File(...)):
    content = await file.read()
    ingest = TextIngestionEngine.ingest_txt_file(content)
    if not ingest["success"]:
        raise HTTPException(status_code=400, detail=ingest["error"])
    return ingest
