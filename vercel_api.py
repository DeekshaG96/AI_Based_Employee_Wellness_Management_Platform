"""Vercel entry point for MoodMentor's persistent account API."""
from __future__ import annotations

import os

# Keep this serverless deployment small and avoid runtime downloads.
# Full Transformers inference remains available in the local Docker stack.
os.environ.setdefault("WELLNESS_DISABLE_TRANSFORMER", "1")
os.environ.setdefault("WELLNESS_LIGHTWEIGHT_PREPROCESSING", "1")
os.environ.setdefault("WELLNESS_BUNDLED_VADER", "1")

from wellness_api.main import app  # noqa: F401
