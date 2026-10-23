"""Request and response validation models for the public API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("name")
    @classmethod
    def trim_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name is required.")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    source_name: str = Field(default="Direct Entry", max_length=255)

    @field_validator("text")
    @classmethod
    def trim_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text cannot be empty.")
        return value


class ActionPathResponse(BaseModel):
    title: str
    duration: str
    description: str
    steps: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    title: str
    category: str = "Well-Being"
    urgency: str = "Low"
    summary: str = ""
    action_items: list[str] = Field(default_factory=list)
    action_paths: list[ActionPathResponse] = Field(default_factory=list)
    is_crisis: bool = False


class AnalysisResponse(BaseModel):
    id: int
    created_at: datetime
    input_summary: dict[str, Any]
    original_text: str
    cleaned_text: str
    processed_text: str
    tokens: list[str]
    sentiment_scores: dict[str, float]
    sentiment_label: str
    counts: dict[str, int]
    vader_input_text: str
    dominant_emotion: str
    emotion_method: str
    emotion_scores: dict[str, float]
    recommendations: RecommendationResponse


class HistoryEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    source_type: str
    source_name: str
    original_text: str
    cleaned_text: str
    processed_text: str
    vader_input_text: str
    tokens: list[str]
    counts: dict[str, int]
    sentiment_label: str
    neg_score: float
    neu_score: float
    pos_score: float
    compound_score: float
    dominant_emotion: str
    emotion_method: str
    emotion_scores: dict[str, float]
    created_at: datetime
    recommendation: RecommendationResponse | None = None


class HistoryResponse(BaseModel):
    items: list[HistoryEntry]
    total: int
    limit: int
    offset: int


class MessageResponse(BaseModel):
    message: str
