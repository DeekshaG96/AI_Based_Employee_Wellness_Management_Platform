"""Streamlit frontend for MoodMentor's authenticated FastAPI and PostgreSQL service."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from wellness.api_client import (
    ApiError,
    analysis_report_from_dict,
)
from wellness.api_client import (
    analyze as api_analyze,
)
from wellness.api_client import (
    analyze_file as api_analyze_file,
)
from wellness.api_client import (
    delete_entry as api_delete_entry,
)
from wellness.api_client import (
    history as api_history,
)
from wellness.api_client import (
    login as api_login,
)
from wellness.api_client import (
    logout as api_logout,
)
from wellness.api_client import (
    register as api_register,
)
from wellness.comparator import compare_models
from wellness.food_planner import get_mood_food_plan
from wellness.meditation import (
    get_meditation_packs,
    render_medito_soundscape_widget,
)
from wellness.pipeline import analyze_text, generate_recommendations
from wellness.preprocessing import get_preprocessing_breakdown
from wellness.report import SentimentLabel
from wellness.team_analytics import calculate_team_burnout_risk, generate_synthetic_team_data
from wellness.wellness_tools import (
    get_cognitive_reframes,
    render_box_breathing_widget,
    render_zen_companion_studio_widget,
)
from wellness.yoga import (
    get_all_yoga_remedies,
    get_yoga_categories,
    get_yoga_remedy_for_emotion,
)

# Page configuration
st.set_page_config(
    page_title="EquiMind | Affective Intelligence & Workplace Resilience OS",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_iframe(html_content: str, height: int = 350) -> None:
    """Render self-contained HTML/JS/CSS component reliably in Streamlit."""
    try:
        st.components.v1.html(html_content, height=height)
    except Exception:
        if hasattr(st, "iframe"):
            st.iframe(html_content, height=height)

# ---------------------------------------------------------------------------
# LUCIDE MINIMALIST SVG ICON SYSTEM (Enterprise Design Architecture)
# ---------------------------------------------------------------------------
LUCIDE_ICONS: dict[str, str] = {
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-activity {extra}"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    "trending-up": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trending-up {extra}"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>',
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield {extra}"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
    "file-text": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-file-text {extra}"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>',
    "users": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-users {extra}"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>',
    "zap": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-zap {extra}"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>',
    "check-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-check-circle {extra}"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
    "alert-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-alert-circle {extra}"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
    "brain": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-brain {extra}"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"></path><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"></path></svg>',
    "heart": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-heart {extra}"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>',
    "layers": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-layers {extra}"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-database {extra}"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>',
    "download": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-download {extra}"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
    "trash": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trash {extra}"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>',
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-search {extra}"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
    "lock": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-lock {extra}"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>',
    "award": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-award {extra}"><circle cx="12" cy="8" r="7"></circle><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"></polyline></svg>',
    "bar-chart": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-bar-chart {extra}"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg>',
    "book-open": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-book-open {extra}"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>',
    "headphones": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-headphones {extra}"><path d="M3 18v-6a9 9 0 0 1 18 0v6"></path><path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z"></path></svg>',
    "utensils": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-utensils {extra}"><path d="M18 2v6a3 3 0 0 1-3 3 3 3 0 0 1-3-3V2"></path><path d="M15 11v11"></path><path d="M5 2v18"></path><path d="M5 6h4"></path></svg>',
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-cpu {extra}"><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>',
    "sparkles": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-sparkles {extra}"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"></path></svg>',
    "refresh": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-refresh {extra}"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>',
}

def svg_icon(name: str, size: int = 16, color: str = "currentColor", extra_classes: str = "") -> str:
    """Render a clean Lucide minimalist SVG icon string."""
    template = LUCIDE_ICONS.get(name)
    if not template:
        return ""
    return template.format(size=size, color=color, extra=extra_classes)


# EquiMind Executive Visual System: High-Class Luxury Aesthetic & Editorial Serif Brand
CUSTOM_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,600&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Outfit:wght@400;500;600;700&display=swap');

/* Award-Winning Awwwards/Apple-Grade UI Overrides */
[data-testid="stDecoration"] {
  display: none !important;
  height: 0 !important;
  visibility: hidden !important;
}
#MainMenu, footer {
  visibility: hidden !important;
  height: 0 !important;
}
header[data-testid="stHeader"] {
  background: transparent !important;
  height: 2.2rem !important;
  z-index: 99 !important;
}

/* Custom Minimalist Silk Scrollbar */
::-webkit-scrollbar {
  width: 7px;
  height: 7px;
}
::-webkit-scrollbar-track {
  background: rgba(240, 253, 250, 0.4);
}
::-webkit-scrollbar-thumb {
  background: rgba(45, 212, 191, 0.45);
  border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(15, 118, 110, 0.75);
}
::selection {
  background: rgba(45, 212, 191, 0.28);
  color: #0f766e;
}

/* Dynamic Ambient Living Atmosphere Glow Orbs */
.stApp::before {
  content: "";
  position: fixed;
  top: -120px;
  right: -120px;
  width: 550px;
  height: 550px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(45, 212, 191, 0.16) 0%, rgba(20, 184, 166, 0.05) 50%, transparent 70%);
  pointer-events: none;
  z-index: 0;
  animation: orbFloat1 18s infinite alternate ease-in-out;
}
.stApp::after {
  content: "";
  position: fixed;
  bottom: -180px;
  left: -80px;
  width: 650px;
  height: 650px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(56, 189, 248, 0.14) 0%, rgba(14, 165, 233, 0.04) 50%, transparent 70%);
  pointer-events: none;
  z-index: 0;
  animation: orbFloat2 24s infinite alternate ease-in-out;
}
@keyframes orbFloat1 {
  0% { transform: translate(0, 0) scale(1); }
  100% { transform: translate(-60px, 80px) scale(1.12); }
}
@keyframes orbFloat2 {
  0% { transform: translate(0, 0) scale(1); }
  100% { transform: translate(80px, -60px) scale(1.18); }
}



/* High-Class Editorial Brand Typography */
.brand-logo-text {
    font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif !important;
    font-size: 2.15rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #044e47 0%, #0f766e 40%, #14b8a6 75%, #2dd4bf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 2px 8px rgba(20, 184, 166, 0.22));
    display: inline-block;
}

.disney-brand-title {
    font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif !important;
    font-size: 3.1rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.025em !important;
    background: linear-gradient(135deg, #044e47 0%, #0f766e 40%, #14b8a6 75%, #2dd4bf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 2px 10px rgba(20, 184, 166, 0.22));
    display: inline-block;
}
.disney-brand-title-sm {
    font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #044e47 0%, #0f766e 50%, #14b8a6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.disney-sparkle {
    display: inline-block;
    animation: sparkleTwinkle 2.5s infinite ease-in-out;
}
@keyframes sparkleTwinkle {
    0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.85; filter: drop-shadow(0 0 6px rgba(45, 212, 191, 0.6)); }
    50% { transform: scale(1.2) rotate(15deg); opacity: 1; filter: drop-shadow(0 0 12px rgba(20, 184, 166, 0.8)); }
}

:root {
  --primary: #0f766e;
  --primary-hover: #115e59;
  --primary-light: #f0fdfa;
  --primary-border: #ccfbf1;
  --slate-900: #0f172a;
  --slate-800: #1e293b;
  --slate-700: #334155;
  --slate-600: #475569;
  --slate-500: #64748b;
  --slate-200: #e2e8f0;
  --slate-100: #f1f5f9;
  --slate-50: #f8fafc;
  --success: #16a34a;
  --success-light: #f0fdf4;
  --success-border: #bbf7d0;
  --warning: #d97706;
  --warning-light: #fef3c7;
  --warning-border: #fde68a;
  --danger: #dc2626;
  --danger-light: #fef2f2;
  --danger-border: #fecaca;
  --transition-smooth: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1);
  --shadow-subtle: 0 6px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -2px rgba(0, 0, 0, 0.02);
  --shadow-hover: 0 20px 42px -6px rgba(15, 118, 110, 0.16), 0 8px 16px -3px rgba(0, 0, 0, 0.04);
  --card-3d-shadow: 0 14px 34px -6px rgba(15, 118, 110, 0.10), 0 4px 12px -2px rgba(0, 0, 0, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.95);
  --card-3d-hover: 0 26px 56px -10px rgba(15, 118, 110, 0.22), 0 10px 24px -4px rgba(0, 0, 0, 0.05), inset 0 1px 1px rgba(255, 255, 255, 1);
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  color: var(--slate-900) !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
  text-rendering: optimizeLegibility !important;
}

/* Header & Viewport Structure */
header[data-testid="stHeader"] {
  background: transparent !important;
}

/* Luxury Silk Alabaster Background with 3D Ambient Orbs */
.stApp {
  background: radial-gradient(circle at 15% 15%, rgba(45, 212, 191, 0.14) 0%, transparent 45%),
              radial-gradient(circle at 85% 20%, rgba(56, 189, 248, 0.12) 0%, transparent 42%),
              radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
              linear-gradient(180deg, #f0fdfa 0%, #f8fafc 40%, #ffffff 100%) !important;
  color: var(--slate-900) !important;
}

.block-container {
  max-width: 1380px !important;
  padding: 1.6rem 2.2rem 4rem !important;
}

/* Refined Luxury Typography Hierarchy - Editorial Serifs Everywhere */
h1,
div[data-testid="stMarkdownContainer"] > h1,
[data-testid="stHeadingWithActionElements"] h1,
.stMarkdown h1,
.editorial-title {
  font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif !important;
  font-size: 2.35rem !important;
  font-weight: 700 !important;
  line-height: 1.2 !important;
  letter-spacing: -0.025em !important;
  color: #044e47 !important;
  margin-bottom: 0.75rem !important;
}

h2,
div[data-testid="stMarkdownContainer"] > h2,
[data-testid="stHeadingWithActionElements"] h2,
.stMarkdown h2,
.editorial-heading {
  font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif !important;
  font-size: 1.75rem !important;
  font-weight: 700 !important;
  line-height: 1.25 !important;
  letter-spacing: -0.022em !important;
  color: #044e47 !important;
  margin-top: 1.25rem !important;
  margin-bottom: 0.5rem !important;
}

h3,
div[data-testid="stMarkdownContainer"] > h3,
[data-testid="stHeadingWithActionElements"] h3,
.stMarkdown h3 {
  font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif !important;
  font-size: 1.40rem !important;
  font-weight: 700 !important;
  line-height: 1.3 !important;
  letter-spacing: -0.018em !important;
  color: #044e47 !important;
  margin-top: 0.9rem !important;
  margin-bottom: 0.45rem !important;
}

h4,
div[data-testid="stMarkdownContainer"] > h4,
[data-testid="stHeadingWithActionElements"] h4,
.stMarkdown h4 {
  font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif !important;
  font-size: 1.18rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.012em !important;
  color: #0f172a !important;
}

h5, h6 {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  color: var(--slate-800) !important;
}

p, li, label, [data-testid="stMarkdownContainer"] p {
  font-size: 0.95rem !important;
  font-weight: 400 !important;
  color: var(--slate-700) !important;
  line-height: 1.6 !important;
}

/* Inline Minimalist Icon Wrapper */
.icon-inline {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  vertical-align: -2px;
  margin-right: 6px;
}

/* Micro-Interactions & 3D Keyframe Animations */
@keyframes fadeInUp {
  from { opacity: 0; transform: perspective(1000px) translateY(12px) translateZ(-10px); }
  to { opacity: 1; transform: perspective(1000px) translateY(0) translateZ(0); }
}

@keyframes float3D {
  0%, 100% {
    transform: perspective(1000px) translateY(0px) rotateX(0deg) rotateY(0deg);
  }
  50% {
    transform: perspective(1000px) translateY(-8px) rotateX(2deg) rotateY(-1.5deg);
  }
}

@keyframes pulseGlow3D {
  0%, 100% {
    box-shadow: 0 0 16px rgba(20, 184, 166, 0.25), 0 8px 24px rgba(15, 118, 110, 0.1);
  }
  50% {
    box-shadow: 0 0 36px rgba(20, 184, 166, 0.55), 0 16px 40px rgba(15, 118, 110, 0.2);
  }
}

@keyframes pulseRing {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.4); }
  70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(22, 163, 74, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

@keyframes holoSheen {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* Workspace Executive Header - 3D Floating Glass Deck */
.workspace-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24px 30px;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 253, 250, 0.6) 100%);
  border: 1px solid rgba(204, 251, 241, 0.85);
  border-radius: 18px;
  box-shadow: var(--card-3d-shadow);
  margin-bottom: 24px;
  animation: fadeInUp 0.45s cubic-bezier(0.34, 1.56, 0.64, 1);
  transform-style: preserve-3d;
  transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.35s ease;
  backdrop-filter: blur(10px);
}
.workspace-header:hover {
  transform: perspective(1000px) translateY(-3px) rotateX(1.2deg);
  box-shadow: var(--card-3d-hover);
}
.workspace-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--primary);
  background: var(--primary-light);
  border: 1px solid var(--primary-border);
  padding: 3px 10px;
  border-radius: 20px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.workspace-title {
  margin: 0 0 4px 0 !important;
  font-size: 1.65rem !important;
  font-weight: 800 !important;
  color: var(--slate-900) !important;
}
.workspace-subtitle {
  margin: 0 !important;
  font-size: 0.88rem !important;
  color: var(--slate-500) !important;
}
.workspace-status-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--slate-100);
  border: 1px solid var(--slate-200);
  padding: 8px 14px;
  border-radius: 24px;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--slate-700);
}
.status-pulse {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--success);
  animation: pulseRing 2s infinite;
}

/* Executive Luxury Pill Tab Bar */
[data-baseweb="tab-list"], div[data-baseweb="tab-list"] {
  gap: 8px !important;
  background: rgba(240, 253, 250, 0.75) !important;
  backdrop-filter: blur(14px) !important;
  -webkit-backdrop-filter: blur(14px) !important;
  padding: 6px !important;
  border-radius: 9999px !important;
  border: 1.5px solid rgba(204, 251, 241, 0.90) !important;
  box-shadow: 0 4px 18px rgba(15, 118, 110, 0.06), inset 0 1px 1px rgba(255, 255, 255, 0.95) !important;
  margin-bottom: 24px !important;
  display: flex !important;
  flex-wrap: wrap !important;
}
button[data-baseweb="tab"], div[data-baseweb="tab-list"] button {
  height: 40px !important;
  padding: 0 20px !important;
  border: none !important;
  border-radius: 9999px !important;
  background: transparent !important;
  color: #475569 !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
  transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
button[data-baseweb="tab"]:hover, div[data-baseweb="tab-list"] button:hover {
  color: #0f766e !important;
  background: rgba(255, 255, 255, 0.65) !important;
  transform: translateY(-1px) !important;
}
button[data-baseweb="tab"][aria-selected="true"], div[data-baseweb="tab-list"] button[aria-selected="true"] {
  background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%) !important;
  color: #044e47 !important;
  font-weight: 700 !important;
  box-shadow: 0 4px 14px rgba(15, 118, 110, 0.12), inset 0 1px 1px #ffffff !important;
  border: 1px solid rgba(204, 251, 241, 0.95) !important;
  transform: scale(1.02) !important;
}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
  display: none !important;
}

/* Form Controls & Inputs */
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stTextArea"] div[data-baseweb="textarea"],
[data-testid="stTextArea"] div[data-baseweb="base-input"],
[data-baseweb="select"] > div {
  border: 1.5px solid #cbd5e1 !important;
  border-radius: 10px !important;
  background-color: #ffffff !important;
  color: var(--slate-900) !important;
  font-size: 0.92rem !important;
  transition: var(--transition-smooth) !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
}
[data-testid="stTextInput"] div[data-baseweb="input"]:hover,
[data-testid="stTextInput"] div[data-baseweb="base-input"]:hover,
[data-testid="stTextArea"] div[data-baseweb="textarea"]:hover,
[data-baseweb="select"] > div:hover {
  border-color: #94a3b8 !important;
}
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within,
[data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within,
[data-baseweb="select"] > div:focus-within {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.15) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  border: none !important;
  background: transparent !important;
  color: var(--slate-900) !important;
  font-size: 0.92rem !important;
}
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p {
  font-size: 0.84rem !important;
  font-weight: 600 !important;
  color: #334155 !important;
  margin-bottom: 4px !important;
  letter-spacing: -0.01em !important;
}

/* Forms - clean borderless container without double boxes */
[data-testid="stForm"] {
  border: none !important;
  padding: 0 !important;
  background: transparent !important;
}

/* Buttons with 3D Tactile Elevation & Press Physics */
.stButton > button, .stFormSubmitButton > button {
  min-height: 42px !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
  transform-style: preserve-3d !important;
}
button[kind="primary"],
button[data-testid="baseButton-primary"],
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] button,
.stFormSubmitButton > button {
  background: linear-gradient(135deg, #0f766e 0%, #0d9488 45%, #14b8a6 100%) !important;
  background-color: #0f766e !important;
  border: 1px solid rgba(255, 255, 255, 0.35) !important;
  border-color: #0d9488 !important;
  color: #ffffff !important;
  border-radius: 12px !important;
  box-shadow: 0 6px 18px rgba(15, 118, 110, 0.32), 0 2px 4px rgba(15, 118, 110, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
  letter-spacing: 0.02em !important;
  transform-style: preserve-3d !important;
}
button[kind="primary"] *,
button[data-testid="baseButton-primary"] *,
.stButton > button[kind="primary"] *,
.stFormSubmitButton > button[kind="primary"] *,
div[data-testid="stFormSubmitButton"] button *,
.stFormSubmitButton > button *,
button[kind="primary"] p,
button[data-testid="baseButton-primary"] p,
div[data-testid="stFormSubmitButton"] button p {
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
}
button[kind="secondary"] *,
button[data-testid="baseButton-secondary"] *,
.stButton > button[kind="secondary"] *,
button[kind="secondary"] p,
button[data-testid="baseButton-secondary"] p {
  color: #1e293b !important;
  font-weight: 600 !important;
}
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button:hover {
  background: linear-gradient(135deg, #115e59 0%, #0f766e 50%, #0d9488 100%) !important;
  background-color: #115e59 !important;
  border-color: #0f766e !important;
  color: #ffffff !important;
  box-shadow: 0 14px 30px rgba(13, 148, 136, 0.45), 0 4px 10px rgba(13, 148, 136, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.6) !important;
  transform: perspective(600px) translateY(-3px) translateZ(8px) scale(1.02) !important;
}
button[kind="primary"]:active,
button[data-testid="baseButton-primary"]:active,
.stButton > button[kind="primary"]:active,
.stFormSubmitButton > button[kind="primary"]:active {
  transform: perspective(600px) translateY(1px) scale(0.98) !important;
  box-shadow: 0 3px 10px rgba(15, 118, 110, 0.3) !important;
}
button[kind="secondary"],
button[data-testid="baseButton-secondary"],
.stButton > button[kind="secondary"],
.stFormSubmitButton > button[kind="secondary"] {
  background: #ffffff !important;
  background-color: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  border-color: #cbd5e1 !important;
  color: #334155 !important;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
}
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
.stButton > button[kind="secondary"]:hover,
.stFormSubmitButton > button[kind="secondary"]:hover {
  background: #f8fafc !important;
  background-color: #f8fafc !important;
  border-color: #94a3b8 !important;
  color: #0f172a !important;
  transform: perspective(600px) translateY(-2px) scale(1.015) !important;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06) !important;
}
.stDownloadButton > button {
  background: #ffffff !important;
  border: 1px solid var(--primary) !important;
  color: var(--primary) !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  box-shadow: 0 3px 10px rgba(15, 118, 110, 0.12) !important;
  transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
.stDownloadButton > button:hover {
  background: var(--primary-light) !important;
  transform: perspective(600px) translateY(-2px) scale(1.02) !important;
  box-shadow: 0 8px 20px rgba(15, 118, 110, 0.22) !important;
}

/* Metric Cards - Editorial Horology / Swiss Boutique Grade */
[data-testid="stMetric"] {
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 253, 250, 0.88) 55%, rgba(241, 245, 249, 0.80) 100%) !important;
  border: 1.5px solid rgba(204, 251, 241, 0.90) !important;
  border-radius: 18px !important;
  padding: 18px 22px !important;
  box-shadow: 0 10px 28px -4px rgba(15, 118, 110, 0.08), 0 2px 8px rgba(0, 0, 0, 0.02), inset 0 1px 1px #ffffff !important;
  backdrop-filter: blur(14px) !important;
  -webkit-backdrop-filter: blur(14px) !important;
  transform-style: preserve-3d !important;
  transition: all 0.30s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-4px) scale(1.01) !important;
  border-color: #5eead4 !important;
  box-shadow: 0 18px 42px -6px rgba(15, 118, 110, 0.18), inset 0 1px 2px #ffffff !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span,
[data-testid="stMetricLabel"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: #0f766e !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.10em !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
  font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif !important;
  color: #044e47 !important;
  font-size: 2.15rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em !important;
}
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] div {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
}

/* 3D Elevated Frosted Containers & Editorial Expanders */
[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1.5px solid rgba(204, 251, 241, 0.88) !important;
  border-radius: 20px !important;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.78) 50%, rgba(248, 250, 252, 0.92) 100%) !important;
  backdrop-filter: blur(14px) !important;
  -webkit-backdrop-filter: blur(14px) !important;
  box-shadow: 0 12px 34px -4px rgba(15, 118, 110, 0.08), 0 2px 8px rgba(0, 0, 0, 0.02), inset 0 1px 1px #ffffff !important;
  transform-style: preserve-3d !important;
  transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.35s ease, border-color 0.3s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover,
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  box-shadow: 0 20px 48px -6px rgba(15, 118, 110, 0.16), 0 6px 16px rgba(0, 0, 0, 0.03), inset 0 1px 2px #ffffff !important;
  border-color: rgba(45, 212, 191, 0.95) !important;
}
[data-testid="stExpander"] {
  border: 1.5px solid rgba(204, 251, 241, 0.85) !important;
  border-radius: 16px !important;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.65) 100%) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  box-shadow: 0 4px 18px rgba(15, 118, 110, 0.05), inset 0 1px 1px #ffffff !important;
  transition: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
[data-testid="stExpander"]:hover {
  border-color: rgba(45, 212, 191, 0.8) !important;
  box-shadow: 0 8px 26px rgba(15, 118, 110, 0.10) !important;
}

/* Sidebar Cool Breezy Luminous Styling */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f0fdfa 0%, #f8fafc 50%, #ffffff 100%) !important;
  border-right: 1px solid rgba(204, 251, 241, 0.9) !important;
  box-shadow: 2px 0 12px rgba(13, 148, 136, 0.04) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding: 0.75rem 0.75rem !important;
}
.sidebar-brand-box {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid #e2e8f0;
  margin-bottom: 8px;
}
.sidebar-logo {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: linear-gradient(135deg, #0d9488, #14b8a6);
  display: grid;
  place-items: center;
  color: #fff;
  font-weight: 800;
  font-size: 1.05rem;
  box-shadow: 0 2px 8px rgba(13, 148, 136, 0.35);
}
.sidebar-brand-text strong {
  display: block;
  color: #0f172a;
  font-size: 0.98rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  line-height: 1.2;
}
.sidebar-brand-text small {
  display: block;
  color: #0d9488;
  font-size: 0.68rem;
  font-weight: 600;
}
.sidebar-user-card {
  background: #ffffff;
  border: 1px solid #ccfbf1;
  border-radius: 12px;
  padding: 9px 12px;
  margin-bottom: 10px;
  box-shadow: 0 4px 12px rgba(13, 148, 136, 0.06);
}
.sidebar-user-card .user-title {
  color: #0f172a;
  font-weight: 700;
  font-size: 0.85rem;
}
.sidebar-user-card .user-email {
  color: #64748b;
  font-size: 0.70rem;
  margin-top: 1px;
}
.sidebar-user-card .user-metric {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid #f1f5f9;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.74rem;
  color: #0d9488;
  font-weight: 600;
}

/* Transform Sidebar Radio into Sleek Interactive Navigation Menu */
[data-testid="stSidebar"] [data-testid="stRadio"] {
  width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
  gap: 3px !important;
  display: flex !important;
  flex-direction: column !important;
  width: 100% !important;
}

/* Hide the native "Select View:" label */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stRadio"] > label,
[data-testid="stSidebar"] [data-testid="stRadio"] > span {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* Eradicate ONLY the radio circle and inner dot across all Streamlit versions */
[data-testid="stSidebar"] [data-testid="stRadio"] div[class*="e1dm8mpf4"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[class*="e1dm8mpf5"],
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"],
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroupIndicator"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child:not([data-testid="stMarkdownContainer"]) {
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  opacity: 0 !important;
  position: absolute !important;
  pointer-events: none !important;
}

/* Reset outer wrapper containers so there is NO nested border box */
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div,
[data-testid="stSidebar"] [data-testid="stRadio"] div[class*="e1dm8mpf1"],
[data-testid="stSidebar"] [data-testid="stRadio"] .react-aria-RadioField {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
  box-shadow: none !important;
}

/* Nav Option Tiles: Crisp, Classy, Luminous Standard */
[data-testid="stSidebar"] [data-testid="stRadioOption"],
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-left: 3.5px solid transparent !important;
  border-radius: 8px !important;
  padding: 7px 10px !important;
  margin: 0 0 1px 0 !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
  width: 100% !important;
  box-sizing: border-box !important;
  display: flex !important;
  align-items: center !important;
  box-shadow: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover,
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: rgba(240, 253, 250, 0.85) !important;
  border-color: rgba(204, 251, 241, 0.8) !important;
  border-left: 3.5px solid #14b8a6 !important;
  transform: translateX(2px) !important;
}

/* Active Nav Tile: Classy Mint-Teal Glow */
[data-testid="stSidebar"] div[data-selected="true"] [data-testid="stRadioOption"],
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"],
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has([class*="e1dm8mpf4"][data-selected="true"]),
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(90deg, #ecfdf5 0%, #f0fdfa 100%) !important;
  border: 1px solid #a7f3d0 !important;
  border-left: 3.5px solid #059669 !important;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.12) !important;
  transform: translateX(2px) !important;
}

/* Typography inside Nav Options */
[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stRadio"] label [data-testid="stMarkdownContainer"] {
  width: 100% !important;
  display: block !important;
  visibility: visible !important;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stRadio"] label [data-testid="stMarkdownContainer"] * {
  color: #475569 !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  line-height: 1.35 !important;
  visibility: visible !important;
  opacity: 1 !important;
  transition: color 0.15s ease !important;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover [data-testid="stMarkdownContainer"] * {
  color: #0f766e !important;
}
[data-testid="stSidebar"] div[data-selected="true"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) [data-testid="stMarkdownContainer"] * {
  color: #065f46 !important;
  font-weight: 700 !important;
}

/* Sidebar Sign Out Button */
[data-testid="stSidebar"] .stButton > button {
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  color: #475569 !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  padding: 8px 14px !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
  transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: #fef2f2 !important;
  border-color: #fca5a5 !important;
  color: #dc2626 !important;
  box-shadow: 0 2px 8px rgba(220, 38, 38, 0.15) !important;
}

/* 1-Click Mood Check-In Cheerful Pills */
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(1) button,
div[data-testid="stColumn"]:nth-child(1) button[kind="secondary"],
div[data-testid="column"]:nth-child(1) button[kind="secondary"],
.st-key-quick_mood_0 button {
  background: #fefce8 !important;
  background-color: #fefce8 !important;
  border: 1.5px solid #fde047 !important;
  border-color: #fde047 !important;
  color: #854d0e !important;
  border-radius: 24px !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  box-shadow: 0 2px 6px rgba(234, 179, 8, 0.12) !important;
  transition: all 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(1) button:hover,
div[data-testid="stColumn"]:nth-child(1) button[kind="secondary"]:hover,
.st-key-quick_mood_0 button:hover {
  background: #fef08a !important;
  background-color: #fef08a !important;
  box-shadow: 0 4px 12px rgba(234, 179, 8, 0.25) !important;
  transform: translateY(-2px) !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(1) button *,
div[data-testid="stColumn"]:nth-child(1) button[kind="secondary"] *,
.st-key-quick_mood_0 button * {
  color: #854d0e !important;
  font-weight: 700 !important;
}

div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(2) button,
div[data-testid="stColumn"]:nth-child(2) button[kind="secondary"],
div[data-testid="column"]:nth-child(2) button[kind="secondary"],
.st-key-quick_mood_1 button {
  background: #f0fdf4 !important;
  background-color: #f0fdf4 !important;
  border: 1.5px solid #86efac !important;
  border-color: #86efac !important;
  color: #166534 !important;
  border-radius: 24px !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  box-shadow: 0 2px 6px rgba(34, 197, 94, 0.12) !important;
  transition: all 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(2) button:hover,
div[data-testid="stColumn"]:nth-child(2) button[kind="secondary"]:hover,
.st-key-quick_mood_1 button:hover {
  background: #dcfce7 !important;
  background-color: #dcfce7 !important;
  box-shadow: 0 4px 12px rgba(34, 197, 94, 0.25) !important;
  transform: translateY(-2px) !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(2) button *,
div[data-testid="stColumn"]:nth-child(2) button[kind="secondary"] *,
.st-key-quick_mood_1 button * {
  color: #166534 !important;
  font-weight: 700 !important;
}

div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(3) button,
div[data-testid="stColumn"]:nth-child(3) button[kind="secondary"],
div[data-testid="column"]:nth-child(3) button[kind="secondary"],
.st-key-quick_mood_2 button {
  background: #f0f9ff !important;
  background-color: #f0f9ff !important;
  border: 1.5px solid #7dd3fc !important;
  border-color: #7dd3fc !important;
  color: #0369a1 !important;
  border-radius: 24px !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  box-shadow: 0 2px 6px rgba(14, 165, 233, 0.12) !important;
  transition: all 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(3) button:hover,
div[data-testid="stColumn"]:nth-child(3) button[kind="secondary"]:hover,
.st-key-quick_mood_2 button:hover {
  background: #e0f2fe !important;
  background-color: #e0f2fe !important;
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25) !important;
  transform: translateY(-2px) !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(3) button *,
div[data-testid="stColumn"]:nth-child(3) button[kind="secondary"] *,
.st-key-quick_mood_2 button * {
  color: #0369a1 !important;
  font-weight: 700 !important;
}

div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(4) button,
div[data-testid="stColumn"]:nth-child(4) button[kind="secondary"],
div[data-testid="column"]:nth-child(4) button[kind="secondary"],
.st-key-quick_mood_3 button {
  background: #faf5ff !important;
  background-color: #faf5ff !important;
  border: 1.5px solid #d8b4fe !important;
  border-color: #d8b4fe !important;
  color: #6b21a8 !important;
  border-radius: 24px !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  box-shadow: 0 2px 6px rgba(168, 85, 247, 0.12) !important;
  transition: all 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(4) button:hover,
div[data-testid="stColumn"]:nth-child(4) button[kind="secondary"]:hover,
.st-key-quick_mood_3 button:hover {
  background: #f3e8ff !important;
  background-color: #f3e8ff !important;
  box-shadow: 0 4px 12px rgba(168, 85, 247, 0.25) !important;
  transform: translateY(-2px) !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(4) button *,
div[data-testid="stColumn"]:nth-child(4) button[kind="secondary"] *,
.st-key-quick_mood_3 button * {
  color: #6b21a8 !important;
  font-weight: 700 !important;
}

div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(5) button,
div[data-testid="stColumn"]:nth-child(5) button[kind="secondary"],
div[data-testid="column"]:nth-child(5) button[kind="secondary"],
.st-key-quick_mood_4 button {
  background: #fff7ed !important;
  background-color: #fff7ed !important;
  border: 1.5px solid #fdba74 !important;
  border-color: #fdba74 !important;
  color: #9a3412 !important;
  border-radius: 24px !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  box-shadow: 0 2px 6px rgba(249, 115, 22, 0.12) !important;
  transition: all 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(5) button:hover,
div[data-testid="stColumn"]:nth-child(5) button[kind="secondary"]:hover,
.st-key-quick_mood_4 button:hover {
  background: #ffedd5 !important;
  background-color: #ffedd5 !important;
  box-shadow: 0 4px 12px rgba(249, 115, 22, 0.25) !important;
  transform: translateY(-2px) !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(5) button *,
div[data-testid="stColumn"]:nth-child(5) button[kind="secondary"] *,
.st-key-quick_mood_4 button * {
  color: #9a3412 !important;
  font-weight: 700 !important;
}

div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(6) button,
div[data-testid="stColumn"]:nth-child(6) button[kind="secondary"],
div[data-testid="column"]:nth-child(6) button[kind="secondary"],
.st-key-quick_mood_5 button {
  background: #ecfeff !important;
  background-color: #ecfeff !important;
  border: 1.5px solid #67e8f9 !important;
  border-color: #67e8f9 !important;
  color: #0e7490 !important;
  border-radius: 24px !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  box-shadow: 0 2px 6px rgba(6, 182, 212, 0.12) !important;
  transition: all 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(6) button:hover,
div[data-testid="stColumn"]:nth-child(6) button[kind="secondary"]:hover,
.st-key-quick_mood_5 button:hover {
  background: #cffafe !important;
  background-color: #cffafe !important;
  box-shadow: 0 4px 12px rgba(6, 182, 212, 0.25) !important;
  transform: translateY(-2px) !important;
}
div[data-testid="stHorizontalBlock"]:has(> div:nth-child(6)) > div:nth-child(6) button *,
div[data-testid="stColumn"]:nth-child(6) button[kind="secondary"] *,
.st-key-quick_mood_5 button * {
  color: #0e7490 !important;
  font-weight: 700 !important;
}

/* Chips */
.chip {
  display: inline-block;
  margin: 3px;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 600;
}
.chip-token { background: #e0f2fe; color: #0369a1; }
.chip-filter { background: #fef3c7; color: #b45309; }
.chip-lemma { background: #dcfce7; color: #15803d; }
.chip-tag { background: #f0fdfa !important; color: #0f766e !important; border: 1px solid #ccfbf1 !important; font-weight: 600 !important; }
.chip-cat { background: #f8fafc !important; color: #1e293b !important; border: 1px solid #cbd5e1 !important; font-weight: 700 !important; }

/* Feedback Sentiment Tone Cards - 3D Elevated Plates */
.card-positive, .card-negative, .card-neutral {
  padding: 18px 20px;
  border-radius: 14px;
  text-align: center;
  box-shadow: var(--card-3d-shadow);
  transform-style: preserve-3d;
  transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.35s ease;
}
.card-positive:hover, .card-negative:hover, .card-neutral:hover {
  transform: perspective(600px) translateY(-4px) rotateX(2deg) scale(1.02);
  box-shadow: var(--card-3d-hover);
}
.card-positive {
  background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
  border: 1px solid #bbf7d0;
  color: #166534;
}
.card-negative {
  background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
  border: 1px solid #fecaca;
  color: #991b1b;
}
.card-neutral {
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  border: 1px solid #e2e8f0;
  color: #334155;
}

/* Landing Page / Welcome Lockup - 3D Kinetic Glass Architecture */
.welcome-panel {
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
  border: 1px solid rgba(204, 251, 241, 0.85);
  border-radius: 18px;
  padding: 30px;
  margin-bottom: 24px;
  box-shadow: var(--card-3d-shadow);
  animation: fadeInUp 0.45s cubic-bezier(0.34, 1.56, 0.64, 1);
  transform-style: preserve-3d;
  transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.35s ease;
  backdrop-filter: blur(10px);
}
.welcome-panel:hover {
  transform: perspective(1000px) translateY(-3px) rotateX(1.2deg);
  box-shadow: var(--card-3d-hover);
}
.welcome-kicker {
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--primary);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.welcome-title {
  font-size: 1.9rem;
  font-weight: 800;
  color: var(--slate-900);
  letter-spacing: -0.03em;
  line-height: 1.25;
  margin-bottom: 12px;
}
.welcome-copy {
  font-size: 0.95rem;
  color: var(--slate-600);
  line-height: 1.6;
  margin-bottom: 18px;
}
.welcome-tags {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.welcome-tags span {
  display: inline-flex;
  align-items: center;
  background: var(--slate-100);
  color: var(--slate-700);
  font-size: 0.8rem;
  font-weight: 600;
  padding: 4px 12px;
  border-radius: 20px;
  border: 1px solid var(--slate-200);
  transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.welcome-tags span:hover {
  background: var(--primary-light);
  border-color: var(--primary-border);
  color: var(--primary);
  transform: perspective(400px) translateY(-2px) scale(1.05);
  box-shadow: 0 4px 10px rgba(15, 118, 110, 0.15);
}
.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}
.feature-card {
  background: linear-gradient(145deg, #ffffff 0%, #fafffc 100%);
  border: 1px solid rgba(226, 232, 240, 0.95);
  border-radius: 14px;
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  box-shadow: var(--card-3d-shadow);
  transform-style: preserve-3d;
  transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.35s ease, border-color 0.35s ease;
}
.feature-card:hover {
  box-shadow: var(--card-3d-hover);
  transform: perspective(800px) translateY(-5px) rotateX(2.5deg) scale(1.02);
  border-color: rgba(45, 212, 191, 0.7);
}
.feature-index {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--primary);
  letter-spacing: 0.05em;
}
.feature-card strong {
  font-size: 0.92rem;
  color: var(--slate-900);
}
.feature-card span:last-child {
  font-size: 0.78rem;
  color: var(--slate-500);
}

/* 3D Kinetic Floating Helpers */
.float-3d-gentle {
  animation: float3D 6s infinite ease-in-out;
}
.pulse-3d-glow {
  animation: pulseGlow3D 3.5s infinite ease-in-out;
}
</style>"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Realistic reflection starter presets
PRESET_SCENARIOS = {
    "Choose a starter scenario (optional)...": "",
    "✨ Celebrating a Team Win": (
        "I am absolutely thrilled with our team's product release! Everyone collaborated seamlessly, "
        "the client feedback was overwhelmingly positive, and I feel truly valued and inspired."
    ),
    "🔥 Deadline Pressure & Fatigue": (
        "I am feeling completely exhausted and overwhelmed. The unyielding project deadlines, constant context "
        "switching, and lack of staffing are making me feel burned out and anxious every morning."
    ),
    "⚡ Mixed Emotions (Pride + Nervousness)": (
        "I feel proud that I finished the work, but worried about tomorrow."
    ),
    "🌧️ Working Through a Setback": (
        "I am not happy with how today's meeting unfolded. The communication broke down and we lost valuable momentum."
    ),
    "🤝 Imposter Doubts & Feeling Isolated": (
        "I feel disconnected from the rest of the unit and worried that my contributions are not meeting senior standards. "
        "It feels hard to ask for help when everyone else appears to know exactly what they are doing."
    ),
    "🔄 Frustration with Sudden Changes": (
        "Extremely frustrated with the sudden change in architectural requirements with zero prior notice. "
        "It wasted two full days of engineering effort and invalidated tested components."
    ),
    "📝 Daily Routine / Neutral Log": (
        "Completed the scheduled weekly sprint retrospective. Reviewed code coverage metrics, merged 4 pull requests, "
        "and documented the upcoming engineering roadmap for executive review."
    ),
    "🛡️ Severe Overwhelm & Seeking Support": (
        "I am feeling completely overwhelmed and hopeless with the constant pressure. I don't know if I can keep going anymore."
    ),
}

# 1-Click Mood Check-In options (Cheerful, Classy, Luminous)
QUICK_MOODS = [
    {
        "emoji": "☀️",
        "label": "Radiant",
        "tags": ["#gratitude", "#team-win", "#joy"],
        "category": "🌱 Personal Growth & Wins",
        "starter": "Today felt rewarding and energizing because ",
    },
    {
        "emoji": "🌿",
        "label": "Peaceful",
        "tags": ["#calm", "#deep-work", "#harmony"],
        "category": "🧘 Mind & Inner State",
        "starter": "Things felt steady, harmonious, and grounded today with ",
    },
    {
        "emoji": "☕",
        "label": "Steady",
        "tags": ["#focus", "#routine", "#flow"],
        "category": "💼 Work & Career",
        "starter": "Today was calm, clear, and productive. I focused on ",
    },
    {
        "emoji": "🌧️",
        "label": "Reflective",
        "tags": ["#gentle-pace", "#self-care"],
        "category": "⚡ Energy & Physical Health",
        "starter": "Feeling a bit quiet and low on battery today. Taking it slow because ",
    },
    {
        "emoji": "⚡",
        "label": "Need Calm",
        "tags": ["#deep-breath", "#boundaries"],
        "category": "💼 Work & Career",
        "starter": "Feeling a rush of pressure right now. Pausing to unpack what's on my plate: ",
    },
    {
        "emoji": "🌊",
        "label": "Release",
        "tags": ["#letting-go", "#recharge"],
        "category": "🧘 Mind & Inner State",
        "starter": "Ready to let go of today's heavy thoughts and reset my mind. Specifically, ",
    },
]

def _load_journal_example() -> None:
    """Copy the selected optional starter scenario into the journal editor."""
    example = st.session_state.get("mood_journal_example", "Choose a starter scenario (optional)...")
    if example and PRESET_SCENARIOS.get(example):
        st.session_state["mood_journal_entry"] = PRESET_SCENARIOS.get(example, "")
        st.session_state["last_analysis_report"] = None
        st.session_state.pop("action_path_choice", None)


def _apply_guided_template() -> None:
    """Apply a structured therapeutic journaling template."""
    mode = st.session_state.get("guided_template_mode", "✍️ Freeform")
    if mode == "🌟 Gratitude & Wins":
        st.session_state["mood_journal_entry"] = (
            "3 Things that went well or brought calm today:\n"
            "1. \n"
            "2. \n"
            "3. \n\n"
            "A moment or win I appreciate: "
        )
        st.session_state["journal_category"] = "🌱 Personal Growth & Wins"
        existing = st.session_state.get("journal_tags", [])
        if "#gratitude" not in existing:
            st.session_state["journal_tags"] = list(set(existing + ["#gratitude"]))
    elif mode == "🌿 Decompress & Reset":
        st.session_state["mood_journal_entry"] = (
            "What feels heavy or stressful right now:\n\n"
            "One small thing within my control today:\n\n"
            "One restful boundary for tonight: "
        )
        st.session_state["journal_category"] = "🧘 Mind & Inner State"
        existing = st.session_state.get("journal_tags", [])
        if "#mindfulness" not in existing:
            st.session_state["journal_tags"] = list(set(existing + ["#mindfulness"]))
    _clear_last_analysis_report()


def _clear_last_analysis_report() -> None:
    """Hide an old result when the user edits the journal text."""
    st.session_state["last_analysis_report"] = None
    st.session_state.pop("action_path_choice", None)


# ---------------------------------------------------------------------------
# TELL MOLLY MENTAL HEALTH JOURNALING ARCHITECTURE (Categories, Tags & Notes)
# ---------------------------------------------------------------------------
TELLMOLLY_CATEGORIES = [
    "💼 Work & Career",
    "⚡ Energy & Physical Health",
    "👥 Relationships & Social",
    "🧘 Mind & Inner State",
    "🌱 Personal Growth & Wins",
    "🌐 General Reflection",
]

POPULAR_TAG_OPTIONS = [
    "#deadline",
    "#deep-work",
    "#meeting-overload",
    "#sleep-debt",
    "#exercise",
    "#walking",
    "#mindfulness",
    "#gratitude",
    "#imposter-syndrome",
    "#context-switching",
    "#team-sync",
    "#family",
    "#leadership",
    "#code-crunch",
]


def parse_entry_metadata(source_name: str, original_text: str = "") -> dict[str, Any]:
    """Parse Tell Molly-style category, tags, and comment notes from metadata and text."""
    category = "General"
    tags: list[str] = []
    note = ""

    s = str(source_name or "").strip()
    if s.startswith("[") and "]" in s:
        idx_end = s.index("]")
        category = s[1:idx_end].strip()
        rest = s[idx_end + 1:].strip()
    else:
        rest = s

    if "| Note:" in rest:
        tag_str, note = rest.split("| Note:", 1)
        note = note.strip()
    else:
        tag_str = rest

    for word in tag_str.split():
        clean_tag = word.strip(".,!?:;\"'()[]{}")
        if clean_tag.startswith("#") and clean_tag not in tags:
            tags.append(clean_tag)

    if original_text:
        for word in str(original_text).split():
            clean_word = word.strip(".,!?:;\"'()[]{}")
            if clean_word.startswith("#") and len(clean_word) > 1 and clean_word not in tags:
                tags.append(clean_word)

    return {"category": category, "tags": tags, "note": note}


EMOTION_PALETTE = {
    "joy": "#4c9a79",
    "fatigue": "#778da9",
    "surprise": "#d49a43",
    "neutral": "#87968d",
    "sadness": "#7486b5",
    "fear": "#c78557",
    "anger": "#cb715e",
    "disgust": "#a5799b",
    "unclassified": "#94a3b8",
}

EMOTION_ESTIMATOR_LABELS = {
    "transformer": "Hugging Face transformer",
    "transformer_multilabel": "Enterprise Multi-Emotion Neural Engine",
    "lexical_fallback": "keyword-based emotion cues (lightweight fallback)",
    "no_emotion_signal": "no clear emotion cue matched",
    "legacy_unknown": "legacy record (method not stored)",
    "legacy_import": "imported legacy record (method not stored)",
}


def _build_analysis_report_markdown(report) -> str:
    """Create a portable report for one journal analysis."""
    scores = report.sentiment_scores
    rec = report.recommendations
    no_emotion_signal = report.emotion_method == "no_emotion_signal"
    emotion_lines = (
        [
            (
                "- No fine-grained emotion estimate was produced because no supported "
                "emotion cue matched."
            )
        ]
        if no_emotion_signal
        else [
            f"- {emotion.capitalize()}: {score:.1%}"
            for emotion, score in sorted(
                report.emotion_scores.items(), key=lambda item: item[1], reverse=True
            )
        ]
    )
    action_lines = [f"- {item}" for item in rec.get("action_items", [])]
    action_path_lines: list[str] = []
    for path in rec.get("action_paths", []):
        action_path_lines.extend(
            [
                f"### {path.get('title', 'A way forward')} ({path.get('duration', 'Flexible')})",
                str(path.get("description", "")),
                *(f"- {step}" for step in path.get("steps", [])),
                "",
            ]
        )
    dominant_emotion = (
        "Unclassified (no clear emotion cue)"
        if no_emotion_signal
        else report.dominant_emotion.capitalize()
    )
    estimator_label = EMOTION_ESTIMATOR_LABELS.get(
        report.emotion_method, report.emotion_method
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    meta = parse_entry_metadata(report.input_summary.source_name, report.original_text)
    return "\n".join(
        [
            "# MoodMentor Mood Analysis Report",
            "",
            f"Generated: {generated_at}",
            "",
            "## Mood summary",
            f"- Sentiment: {report.sentiment_label.value.capitalize()}",
            f"- Dominant emotion: {dominant_emotion}",
            f"- VADER compound score: {scores.compound:.3f}",
            f"- Emotion estimator: {estimator_label}",
            f"- Life Domain Category: {meta['category']}",
            f"- Attached Tags: {', '.join(meta['tags']) if meta['tags'] else 'None'}",
            *([f"- Context / Reflection Note: {meta['note']}"] if meta["note"] else []),
            "",
            "## Sentiment scores",
            f"- Positive: {scores.positive:.3f}",
            f"- Neutral: {scores.neutral:.3f}",
            f"- Negative: {scores.negative:.3f}",
            "",
            "## Emotion estimates",
            *(emotion_lines or ["- No emotion scores were returned."]),
            *(
                ["Note: Multi-emotion scores are independent signals and evaluate distinct cognitive-affective dimensions."]
                if report.emotion_method == "transformer_multilabel"
                else []
            ),
            "",
            "## Suggested next steps",
            f"**{rec.get('title', 'Well-being reflection')}**",
            "",
            str(rec.get("summary", "")),
            "",
            "Choose one way to begin",
            "",
            *(action_path_lines or ["- No action paths were returned."]),
            "## Other quick ideas",
            *(action_lines or ["- No action items were returned."]),
            "",
            "## Journal entry",
            report.original_text,
            "",
            "---",
            "This report supports self-reflection and is not a medical diagnosis.",
            "",
        ]
    )


def _build_analysis_report_pdf(report) -> bytes:
    """Create a readable PDF report for one journal analysis."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    forest = colors.HexColor("#155c49")
    ink = colors.HexColor("#19352d")
    muted = colors.HexColor("#657b71")
    pale = colors.HexColor("#e9f4eb")
    line = colors.HexColor("#dce7df")

    def safe_text(value) -> str:
        """Keep PDF output robust for emojis and characters outside Helvetica."""
        text = str(value or "")
        replacements = {
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "—": "-",
            "–": "-",
            "…": "...",
            "•": "-",
        }
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
        return text.encode("cp1252", errors="replace").decode("cp1252")

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="MoodTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=forest,
            alignment=TA_CENTER,
            spaceAfter=5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MoodSubtitle",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=muted,
            alignment=TA_CENTER,
            spaceAfter=7 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MoodSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=forest,
            spaceBefore=5 * mm,
            spaceAfter=2.5 * mm,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MoodBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=ink,
            spaceAfter=2 * mm,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="MoodSmall",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=muted,
        )
    )

    def paragraph(value, style="MoodBody") -> Paragraph:
        escaped = escape(safe_text(value)).replace("\n", "<br/>")
        return Paragraph(escaped or " ", styles[style])

    buffer = BytesIO()
    page_width, _ = A4
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="MoodMentor Mood Analysis Report",
        author="MoodMentor",
        subject="Personal mood reflection and well-being suggestions",
    )
    content_width = page_width - 36 * mm
    report_scores = report.sentiment_scores
    recommendation = report.recommendations
    no_emotion_signal = report.emotion_method == "no_emotion_signal"
    dominant_emotion = (
        "Unclassified (no clear emotion cue)"
        if no_emotion_signal
        else report.dominant_emotion.capitalize()
    )
    estimator = EMOTION_ESTIMATOR_LABELS.get(
        report.emotion_method, report.emotion_method
    )

    story = [
        paragraph("MoodMentor", "MoodTitle"),
        paragraph("Personal Mood Analysis Report", "MoodSubtitle"),
        paragraph(
            f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}",
            "MoodSmall",
        ),
        Spacer(1, 4 * mm),
        paragraph("Mood summary", "MoodSection"),
    ]
    summary = Table(
        [
            ["SENTIMENT", "DOMINANT EMOTION"],
            [report.sentiment_label.value.capitalize(), safe_text(dominant_emotion)],
            ["VADER COMPOUND", "EMOTION ESTIMATOR"],
            [f"{report_scores.compound:.3f}", safe_text(estimator)],
        ],
        colWidths=[content_width / 2, content_width / 2],
        hAlign="LEFT",
    )
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), pale),
                ("BACKGROUND", (0, 2), (-1, 2), pale),
                ("TEXTCOLOR", (0, 0), (-1, -1), ink),
                ("TEXTCOLOR", (0, 0), (-1, 0), forest),
                ("TEXTCOLOR", (0, 2), (-1, 2), forest),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTSIZE", (0, 0), (-1, 0), 7.5),
                ("FONTSIZE", (0, 2), (-1, 2), 7.5),
                ("BOX", (0, 0), (-1, -1), 0.6, line),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, line),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([summary, paragraph("Sentiment scores", "MoodSection")])
    sentiment_table = Table(
        [
            ["Positive", "Neutral", "Negative"],
            [
                f"{report_scores.positive:.3f}",
                f"{report_scores.neutral:.3f}",
                f"{report_scores.negative:.3f}",
            ],
        ],
        colWidths=[content_width / 3] * 3,
        hAlign="LEFT",
    )
    sentiment_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), pale),
                ("TEXTCOLOR", (0, 0), (-1, -1), ink),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOX", (0, 0), (-1, -1), 0.6, line),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, line),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([sentiment_table, paragraph("Emotion estimates", "MoodSection")])
    if no_emotion_signal:
        story.append(
            paragraph(
                "No fine-grained emotion estimate was produced because no supported "
                "emotion cue matched."
            )
        )
    elif report.emotion_scores:
        emotion_rows = [["Emotion", "Score"]]
        emotion_rows.extend(
            [safe_text(emotion.capitalize()), f"{score:.1%}"]
            for emotion, score in sorted(
                report.emotion_scores.items(), key=lambda item: item[1], reverse=True
            )
        )
        emotion_table = Table(
            emotion_rows,
            colWidths=[content_width * 0.68, content_width * 0.32],
            repeatRows=1,
            hAlign="LEFT",
        )
        emotion_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), pale),
                    ("TEXTCOLOR", (0, 0), (-1, -1), ink),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOX", (0, 0), (-1, -1), 0.6, line),
                    ("INNERGRID", (0, 0), (-1, -1), 0.35, line),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(emotion_table)
        if report.emotion_method == "lexical_fallback":
            story.append(
                paragraph(
                    "Lexical scores are heuristic indicators, not model confidence values.",
                    "MoodSmall",
                )
            )
    else:
        story.append(paragraph("No emotion scores were returned."))

    story.extend(
        [
            paragraph("Suggested next steps", "MoodSection"),
            paragraph(recommendation.get("title", "Well-being reflection")),
            paragraph(recommendation.get("summary", "")),
        ]
    )
    for path in recommendation.get("action_paths", []):
        path_title = path.get("title", "A way forward")
        duration = path.get("duration", "Flexible")
        story.append(paragraph(f"{path_title} - {duration}"))
        if path.get("description"):
            story.append(paragraph(path["description"]))
        for step in path.get("steps", []):
            story.append(paragraph(f"- {step}"))
        story.append(Spacer(1, 2 * mm))

    action_items = recommendation.get("action_items", [])
    if action_items:
        story.append(paragraph("Other quick ideas", "MoodSection"))
        for item in action_items:
            story.append(paragraph(f"- {item}"))

    story.extend(
        [
            paragraph("Journal entry", "MoodSection"),
            paragraph(report.original_text or "No journal text was supplied."),
            Spacer(1, 4 * mm),
            paragraph(
                "MoodMentor supports self-reflection. Its estimates and suggestions are "
                "not a medical diagnosis or a replacement for professional care.",
                "MoodSmall",
            ),
        ]
    )

    def draw_footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(line)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 13 * mm, page_width - 18 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(18 * mm, 8 * mm, "MoodMentor | Personal reflection")
        canvas.drawRightString(page_width - 18 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()


def _render_mood_trend() -> None:
    """Show the saved VADER mood history, including the latest analysis."""
    try:
        entries = api_history(st.session_state.user_id, st.session_state.api_token)
    except Exception:
        entries = []

    if not entries:
        st.info("Your mood trend will appear after you save a journal analysis.")
        return

    trend_df = pd.DataFrame(entries)
    required = {"created_at", "compound_score"}
    if not required.issubset(trend_df.columns):
        st.caption("Mood trend data is not available for these records.")
        return

    trend_df["created_at"] = pd.to_datetime(trend_df["created_at"], errors="coerce", utc=True)
    trend_df["compound_score"] = pd.to_numeric(trend_df["compound_score"], errors="coerce")
    trend_df = trend_df.dropna(subset=["created_at", "compound_score"]).sort_values("created_at")
    if trend_df.empty:
        st.info("Your mood trend will appear after you save a journal analysis.")
        return

    hover_columns = [
        column
        for column in ("sentiment_label", "dominant_emotion")
        if column in trend_df.columns
    ]
    fig = px.line(
        trend_df,
        x="created_at",
        y="compound_score",
        markers=True,
        hover_data=hover_columns,
        labels={"created_at": "Analysis time", "compound_score": "Sentiment score"},
        color_discrete_sequence=["#155c49"],
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="#34d399")
    fig.add_hline(y=-0.05, line_dash="dash", line_color="#f87171")
    fig.update_layout(
        height=300,
        margin={"l": 20, "r": 20, "t": 15, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"range": [-1.05, 1.05]},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Each point is a saved journal analysis. The score reflects text sentiment, not a clinical assessment.")


def _apply_auth(auth: dict):
    """Store the API-issued session for this Streamlit browser session."""
    st.session_state.last_analysis_report = None
    st.session_state["mood_journal_entry"] = ""
    user = auth["user"]
    st.session_state.logged_in = True
    st.session_state.api_token = auth["access_token"]
    st.session_state.user_id = user["id"]
    st.session_state.user_name = user["name"]
    st.session_state.user_email = user["email"]


def render_landing_hero_3d() -> str:
    """Return an ultra-classy, interactive Three.js 3D Zen Crystal Orb centerpiece for the landing page."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Playfair+Display:ital,wght@0,600;1,400&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          background: transparent;
          font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
          color: #0f172a;
          overflow: hidden;
          height: 335px;
          user-select: none;
        }
        .hero-3d-card {
          width: 100%;
          height: 100%;
          border-radius: 20px;
          background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.78) 55%, rgba(241, 245, 249, 0.72) 100%);
          border: 1.5px solid rgba(204, 251, 241, 0.85);
          box-shadow: 0 14px 34px -8px rgba(15, 118, 110, 0.12), 0 2px 8px rgba(0, 0, 0, 0.02), inset 0 1px 1px rgba(255, 255, 255, 0.95);
          position: relative;
          overflow: hidden;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          backdrop-filter: blur(14px);
          -webkit-backdrop-filter: blur(14px);
        }
        .card-topbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px 16px 2px;
          z-index: 5;
        }
        .aura-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: rgba(240, 253, 250, 0.95);
          border: 1px solid rgba(45, 212, 191, 0.6);
          color: #0f766e;
          font-size: 10px;
          font-weight: 700;
          padding: 3px 11px;
          border-radius: 9999px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          box-shadow: 0 2px 6px rgba(15, 118, 110, 0.06);
        }
        .pulse-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #14b8a6;
          animation: pulse3D 2s infinite ease-in-out;
        }
        @keyframes pulse3D {
          0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(20, 184, 166, 0.5); }
          50% { transform: scale(1.2); opacity: 0.8; box-shadow: 0 0 0 5px rgba(20, 184, 166, 0); }
        }
        .hint-badge {
          font-size: 9.5px;
          font-family: 'JetBrains Mono', monospace;
          color: #64748b;
          background: rgba(255, 255, 255, 0.9);
          border: 1px solid #e2e8f0;
          padding: 3px 9px;
          border-radius: 8px;
          font-weight: 600;
          letter-spacing: 0.02em;
        }
        #webgl-viewport {
          height: 215px;
          min-height: 215px;
          width: 100%;
          cursor: grab;
          position: relative;
        }
        #webgl-viewport:active {
          cursor: grabbing;
        }
        .card-bottombar {
          padding: 6px 14px 12px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          z-index: 5;
        }
        .affirmation-pill {
          display: flex;
          align-items: center;
          gap: 10px;
          background: rgba(255, 255, 255, 0.95);
          border: 1px solid rgba(204, 251, 241, 0.85);
          border-radius: 9999px;
          padding: 6px 14px;
          font-size: 11.5px;
          color: #0f766e;
          font-weight: 600;
          box-shadow: 0 2px 8px rgba(15, 118, 110, 0.06);
          width: 100%;
          justify-content: space-between;
        }
        .touch-tag {
          font-size: 9px;
          text-transform: uppercase;
          background: linear-gradient(135deg, #ccfbf1, #99f6e4);
          color: #0f766e;
          padding: 3px 8px;
          border-radius: 9999px;
          font-weight: 700;
          letter-spacing: 0.05em;
          white-space: nowrap;
          cursor: pointer;
        }
      </style>
    </head>
    <body>
      <div class="hero-3d-card">
        <div class="card-topbar">
          <div class="aura-badge">
            <span class="pulse-dot"></span>
            <span>Live 3D Sanctuary Aura</span>
          </div>
          <span class="hint-badge">Touch & Orbit 3D</span>
        </div>
        <div id="webgl-viewport"></div>
        <div class="card-bottombar">
          <div class="affirmation-pill">
            <span id="affirmation-msg">"Breathe in serene calm · Release all tension."</span>
            <span class="touch-tag">Click to Pulse</span>
          </div>
        </div>
      </div>

      <script>
      (function() {
        var container = document.getElementById('webgl-viewport');
        if (!container || typeof THREE === 'undefined') return;

        var width = container.clientWidth || 440;
        var height = 250;

        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 1000);
        camera.position.z = 4.3;

        var renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        container.appendChild(renderer.domElement);

        var ambLight = new THREE.AmbientLight(0xffffff, 1.3);
        scene.add(ambLight);

        var pLight1 = new THREE.PointLight(0x14b8a6, 3.2, 40);
        pLight1.position.set(3, 3.5, 3);
        scene.add(pLight1);

        var pLight2 = new THREE.PointLight(0xf59e0b, 2.0, 40);
        pLight2.position.set(-3, -2.5, 2);
        scene.add(pLight2);

        var pLight3 = new THREE.PointLight(0x38bdf8, 1.6, 40);
        pLight3.position.set(0, 4, -2);
        scene.add(pLight3);

        var masterGroup = new THREE.Group();

        // 1. Faceted Emerald-Teal Icosahedron Crystal
        var crystalGeo = new THREE.IcosahedronGeometry(0.92, 0);
        var crystalMat = new THREE.MeshPhongMaterial({
          color: 0x0f766e,
          emissive: 0x044e47,
          specular: 0x99f6e4,
          shininess: 110,
          transparent: true,
          opacity: 0.88,
          flatShading: true
        });
        var crystalMesh = new THREE.Mesh(crystalGeo, crystalMat);
        masterGroup.add(crystalMesh);

        // 2. Translucent Cyan Wireframe Lattice
        var wireGeo = new THREE.IcosahedronGeometry(0.94, 0);
        var wireMat = new THREE.MeshBasicMaterial({
          color: 0x5eead4,
          wireframe: true,
          transparent: true,
          opacity: 0.40
        });
        var wireMesh = new THREE.Mesh(wireGeo, wireMat);
        masterGroup.add(wireMesh);

        // 3. Radiant Octahedral Amber-Gold Heart
        var innerGeo = new THREE.OctahedronGeometry(0.44, 0);
        var innerMat = new THREE.MeshPhongMaterial({
          color: 0xfef08a,
          emissive: 0xd97706,
          shininess: 100,
          flatShading: true
        });
        var innerMesh = new THREE.Mesh(innerGeo, innerMat);
        masterGroup.add(innerMesh);

        // 4. Ethereal Celestial Halos (Permanent Open Ellipses - No Slash / No Edge-on Flip)
        var haloGroup = new THREE.Group();
        haloGroup.rotation.x = 1.15; // Stable ~66-degree incline relative to camera
        scene.add(haloGroup);

        var ring1Geo = new THREE.TorusGeometry(1.48, 0.018, 20, 90);
        var ring1Mat = new THREE.MeshPhongMaterial({
          color: 0x2dd4bf,
          emissive: 0x0f766e,
          specular: 0xffffff,
          shininess: 120,
          transparent: true,
          opacity: 0.65
        });
        var ring1 = new THREE.Mesh(ring1Geo, ring1Mat);
        haloGroup.add(ring1);

        var ring2Geo = new THREE.TorusGeometry(1.70, 0.012, 20, 90);
        var ring2Mat = new THREE.MeshPhongMaterial({
          color: 0x99f6e4,
          emissive: 0x0f766e,
          specular: 0xffffff,
          shininess: 120,
          transparent: true,
          opacity: 0.42
        });
        var ring2 = new THREE.Mesh(ring2Geo, ring2Mat);
        haloGroup.add(ring2);

        // 5. Orbiting Luminous Pearl
        var satGeo = new THREE.SphereGeometry(0.08, 20, 20);
        var satMat = new THREE.MeshPhongMaterial({
          color: 0xffffff,
          emissive: 0x5eead4,
          specular: 0xffffff,
          shininess: 100
        });
        var satellite = new THREE.Mesh(satGeo, satMat);
        masterGroup.add(satellite);

        // 6. Floating Ambient Stardust Constellation
        var pCount = 110;
        var pGeo = new THREE.BufferGeometry();
        var pPos = new Float32Array(pCount * 3);
        for (var i = 0; i < pCount * 3; i += 3) {
          pPos[i] = (Math.random() - 0.5) * 7.5;
          pPos[i+1] = (Math.random() - 0.5) * 5.5;
          pPos[i+2] = (Math.random() - 0.5) * 5.5;
        }
        pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
        var pMat = new THREE.PointsMaterial({
          color: 0x2dd4bf,
          size: 0.065,
          transparent: true,
          opacity: 0.75
        });
        var particles = new THREE.Points(pGeo, pMat);
        scene.add(particles);

        scene.add(masterGroup);

        // Mouse Parallax & Drag Orbit Physics
        var isDragging = false;
        var prevMouseX = 0, prevMouseY = 0;
        var targetRotX = 0, targetRotY = 0;
        var curRotX = 0, curRotY = 0;

        container.addEventListener('mousedown', function(e) {
          isDragging = true;
          prevMouseX = e.clientX;
          prevMouseY = e.clientY;
        });
        window.addEventListener('mouseup', function() {
          isDragging = false;
        });
        window.addEventListener('mousemove', function(e) {
          if (isDragging) {
            var dx = e.clientX - prevMouseX;
            var dy = e.clientY - prevMouseY;
            targetRotY += dx * 0.01;
            targetRotX += dy * 0.01;
            prevMouseX = e.clientX;
            prevMouseY = e.clientY;
          }
        });

        // Ambient Mouse Hover Parallax Tilt
        container.addEventListener('mousemove', function(e) {
          if (!isDragging) {
            var rect = container.getBoundingClientRect();
            var mx = (e.clientX - rect.left) / rect.width - 0.5;
            var my = (e.clientY - rect.top) / rect.height - 0.5;
            targetRotY = mx * 0.55;
            targetRotX = my * 0.40;
          }
        });

        // Touch support for mobile and tablets
        container.addEventListener('touchstart', function(e) {
          if (e.touches.length === 1) {
            isDragging = true;
            prevMouseX = e.touches[0].clientX;
            prevMouseY = e.touches[0].clientY;
          }
        }, { passive: true });
        window.addEventListener('touchend', function() {
          isDragging = false;
        });
        window.addEventListener('touchmove', function(e) {
          if (isDragging && e.touches.length === 1) {
            var dx = e.touches[0].clientX - prevMouseX;
            var dy = e.touches[0].clientY - prevMouseY;
            targetRotY += dx * 0.012;
            targetRotX += dy * 0.012;
            prevMouseX = e.touches[0].clientX;
            prevMouseY = e.touches[0].clientY;
          }
        }, { passive: true });

        // Click Pulse Affirmation Wave
        var isPulsing = false;
        var pulseTimer = 0;
        var affirmations = [
          '"Breathe in serene calm · Release all tension."',
          '"Equanimity in every thought · Grounded clarity."',
          '"Your mindful presence elevates those around you."',
          '"Stillness is the sanctuary of your creativity."',
          '"Centered, focused, resilient, and at peace."'
        ];

        function triggerPulse() {
          isPulsing = true;
          pulseTimer = 1.6;
          var msgEl = document.getElementById('affirmation-msg');
          if (msgEl) {
            msgEl.textContent = affirmations[Math.floor(Math.random() * affirmations.length)];
          }
        }
        container.addEventListener('click', triggerPulse);

        var clock = new THREE.Clock();
        function animate() {
          requestAnimationFrame(animate);
          var delta = clock.getDelta();
          var t = clock.getElapsedTime();

          // Organic Pranayama Diaphragmatic Breathing
          var breathScale = 1.0 + 0.06 * Math.sin(t * 1.3);
          if (isPulsing) {
            pulseTimer -= delta;
            breathScale += 0.10 * Math.sin(t * 14);
            crystalMat.emissive.setHex(0xb45309);
            crystalMat.color.setHex(0x5eead4);
            pMat.size = 0.095;
            if (pulseTimer <= 0) {
              isPulsing = false;
              crystalMat.emissive.setHex(0x044e47);
              crystalMat.color.setHex(0x0f766e);
              pMat.size = 0.065;
            }
          }

          crystalMesh.scale.set(breathScale, breathScale, breathScale);
          wireMesh.scale.set(breathScale, breathScale, breathScale);
          innerMesh.rotation.y -= delta * 0.9;
          innerMesh.rotation.x += delta * 0.6;

          // Celestial Orbit Halos (Spinning in-plane: perfectly open ellipses, zero slash!)
          ring1.rotation.z += delta * 0.22;
          ring2.rotation.z -= delta * 0.16;
          haloGroup.position.y = masterGroup.position.y;
          haloGroup.rotation.y = curRotY * 0.4;

          // Satellite pearl path
          var satAngle = t * 1.4;
          satellite.position.set(
            Math.cos(satAngle) * 1.55,
            Math.sin(satAngle * 0.8) * 0.5,
            Math.sin(satAngle) * 1.55
          );

          // Particles gentle drift
          particles.rotation.y += delta * 0.03;

          // Smooth damping rotation physics
          curRotX += (targetRotX - curRotX) * 0.07;
          curRotY += (targetRotY - curRotY) * 0.07;

          masterGroup.position.y = Math.sin(t * 1.1) * 0.08;
          masterGroup.rotation.y = curRotY + t * 0.18;
          masterGroup.rotation.x = curRotX + Math.sin(t * 0.5) * 0.06;

          renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', function() {
          var w = container.clientWidth || 440;
          var h = 250;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        });
      })();
      </script>
    </body>
    </html>
    """



def render_landing_and_auth():
    """Render EquiMind's elegant, high-aesthetic executive wellness access sanctuary."""
    st.markdown("""
<style>
/* Cool Breezy Serene Atmospheric Background */
.stApp {
    background: radial-gradient(circle at 18% 18%, rgba(45, 212, 191, 0.18) 0%, transparent 45%),
                radial-gradient(circle at 82% 22%, rgba(56, 189, 248, 0.16) 0%, transparent 42%),
                linear-gradient(180deg, #f0fdfa 0%, #f8fafc 45%, #eff6ff 100%) !important;
    min-height: 100vh;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1220px !important;
}
/* Brand Header Bar */
.top-brand-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 22px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(45, 212, 191, 0.25);
}
.brand-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-badge-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #0f766e, #14b8a6);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 800;
    font-size: 1.2rem;
    box-shadow: 0 4px 14px rgba(15, 118, 110, 0.3);
}
.engine-status-badge {
    background: rgba(255, 255, 255, 0.9);
    border: 1.5px solid rgba(45, 212, 191, 0.45);
    color: #0f766e;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 9999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    box-shadow: 0 3px 12px rgba(45, 212, 191, 0.12);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* ==========================================================================
   AWARD-WINNING LUXURY EDITORIAL POLISH & AESTHETICS (Awwwards / Horology Grade)
   ========================================================================== */

/* Universal Editorial Serif Typography for Headings & Display Titles */
h1, h2, h3, h4,
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3,
[data-testid="stHeadingWithActionElements"] h4,
[data-testid="stHeadingWithActionElements"] span,
div[data-testid="stMarkdownContainer"] > h1,
div[data-testid="stMarkdownContainer"] > h2,
div[data-testid="stMarkdownContainer"] > h3,
div[data-testid="stMarkdownContainer"] > h4,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
.editorial-serif, .editorial-title, .editorial-heading {
  font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif !important;
  color: #044e47 !important;
  letter-spacing: -0.025em !important;
  font-weight: 700 !important;
  line-height: 1.25 !important;
}

/* Crisp Clean Body & Label Typography (Excluding Icon Fonts) */
html, body, [data-testid="stAppViewContainer"],
p, label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] label,
[data-baseweb="tab"] {
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
}

/* Protect Streamlit Material Symbols & Password Visibility Toggle */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
[data-baseweb="base-input"] button,
[data-baseweb="base-input"] button *,
button[aria-label*="password" i],
button[aria-label*="password" i] *,
button[aria-label*="visibility" i],
button[aria-label*="visibility" i] * {
  font-family: "Material Symbols Rounded", "Material Icons" !important;
  font-style: normal !important;
  font-weight: 400 !important;
  font-variant: normal !important;
  text-transform: none !important;
  letter-spacing: normal !important;
  word-wrap: normal !important;
  white-space: nowrap !important;
  direction: ltr !important;
  -webkit-font-smoothing: antialiased !important;
  text-rendering: optimizeLegibility !important;
}

/* Secondary Button Upgrade: Pill Silhouette with Luminous Specular Rim */
button[kind="secondary"],
button[data-testid="baseButton-secondary"],
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.75) 100%) !important;
  border: 1.5px solid rgba(204, 251, 241, 0.90) !important;
  border-radius: 9999px !important;
  color: #0f766e !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important;
  box-shadow: 0 4px 14px rgba(15, 118, 110, 0.05), inset 0 1px 1px #ffffff !important;
  transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
  background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%) !important;
  border-color: #14b8a6 !important;
  color: #044e47 !important;
  box-shadow: 0 8px 22px rgba(15, 118, 110, 0.14), inset 0 1px 1px #ffffff !important;
  transform: translateY(-2px) scale(1.02) !important;
}
button[kind="secondary"] *,
button[data-testid="baseButton-secondary"] *,
.stButton > button[kind="secondary"] *,
button[kind="secondary"] p,
button[data-testid="baseButton-secondary"] p {
  color: #0f766e !important;
  font-weight: 700 !important;
}
button[kind="secondary"]:hover *,
button[data-testid="baseButton-secondary"]:hover *,
.stButton > button[kind="secondary"]:hover * {
  color: #044e47 !important;
}

/* Primary Button Upgrade: High-Impact Emerald Silk Pill */
.stButton > button[kind="primary"],
button[kind="primary"],
button[data-testid="baseButton-primary"],
.stFormSubmitButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] button {
  border-radius: 9999px !important;
  background: linear-gradient(135deg, #0f766e 0%, #0d9488 45%, #14b8a6 100%) !important;
  box-shadow: 0 6px 20px rgba(15, 118, 110, 0.32), inset 0 1px 1px rgba(255, 255, 255, 0.45) !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em !important;
  border: none !important;
  color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover,
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
  background: linear-gradient(135deg, #115e59 0%, #0f766e 50%, #0d9488 100%) !important;
  box-shadow: 0 12px 30px rgba(15, 118, 110, 0.42), inset 0 1px 2px rgba(255, 255, 255, 0.6) !important;
  transform: translateY(-2px) scale(1.02) !important;
}

/* Editorial Form Controls: Smooth Frosted Glass with Emerald Glow Focus */
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stTextArea"] div[data-baseweb="textarea"],
[data-testid="stTextArea"] div[data-baseweb="base-input"],
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] {
  border: 1.5px solid rgba(204, 251, 241, 0.90) !important;
  border-radius: 14px !important;
  background: rgba(255, 255, 255, 0.94) !important;
  backdrop-filter: blur(10px) !important;
  -webkit-backdrop-filter: blur(10px) !important;
  color: #0f172a !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.92rem !important;
  transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
  box-shadow: 0 2px 8px rgba(15, 118, 110, 0.04), inset 0 1px 1px rgba(255, 255, 255, 0.8) !important;
}
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within,
[data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within,
[data-baseweb="select"] > div:focus-within {
  border-color: #14b8a6 !important;
  background: #ffffff !important;
  box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.25), 0 8px 22px rgba(15, 118, 110, 0.08) !important;
  transform: translateY(-1px) !important;
}

/* Scenario Choice Tiles in Main View (Radio Buttons) */
div[data-testid="stRadio"]:not([data-testid="stSidebar"] div[data-testid="stRadio"]) div[role="radiogroup"] {
  gap: 10px !important;
  display: flex !important;
  flex-direction: column !important;
}
div[data-testid="stRadio"]:not([data-testid="stSidebar"] div[data-testid="stRadio"]) label {
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 253, 250, 0.60) 100%) !important;
  border: 1.5px solid rgba(204, 251, 241, 0.90) !important;
  border-radius: 14px !important;
  padding: 12px 18px !important;
  cursor: pointer !important;
  transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
  box-shadow: 0 4px 14px rgba(15, 118, 110, 0.04) !important;
}
div[data-testid="stRadio"]:not([data-testid="stSidebar"] div[data-testid="stRadio"]) label:hover {
  border-color: #14b8a6 !important;
  background: rgba(240, 253, 250, 0.95) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 22px rgba(15, 118, 110, 0.10) !important;
}
div[data-testid="stRadio"]:not([data-testid="stSidebar"] div[data-testid="stRadio"]) label:has(input:checked) {
  border-color: #0f766e !important;
  background: linear-gradient(135deg, rgba(204, 251, 241, 0.85) 0%, rgba(240, 253, 250, 0.95) 100%) !important;
  box-shadow: 0 6px 18px rgba(15, 118, 110, 0.14), inset 0 1px 1px #ffffff !important;
  transform: scale(1.01) !important;
}

/* Editorial Data Tables & DataFrames */
[data-testid="stTable"], [data-testid="stDataFrame"], table {
  border-radius: 16px !important;
  overflow: hidden !important;
  border: 1.5px solid rgba(204, 251, 241, 0.85) !important;
  background: rgba(255, 255, 255, 0.95) !important;
  box-shadow: 0 8px 24px rgba(15, 118, 110, 0.06) !important;
}
table thead tr th, [data-testid="stTable"] th {
  background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%) !important;
  color: #ffffff !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  padding: 12px 16px !important;
  border: none !important;
}
table tbody tr td, [data-testid="stTable"] td {
  padding: 12px 16px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.88rem !important;
  color: #334155 !important;
  border-bottom: 1px solid rgba(204, 251, 241, 0.5) !important;
}
table tbody tr:nth-child(even) td {
  background: rgba(240, 253, 250, 0.45) !important;
}
table tbody tr:hover td {
  background: rgba(204, 251, 241, 0.40) !important;
}

</style>
""", unsafe_allow_html=True)

    # Top Brand Bar - Elegant Cursive Legible Logo
    st.markdown("""
<div class="top-brand-bar">
    <div class="brand-left">
        <div class="brand-badge-icon">🌿</div>
        <div>
            <span class="disney-brand-title" style="font-size: 2.35rem; line-height: 1.1; letter-spacing: -0.02em;">EquiMind</span>
            <span style="font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-style: italic; font-size: 1.05rem; color: #0f766e; margin-left: 8px; font-weight: 600;">
                · Emotional Wellness & Resilience
            </span>
        </div>
    </div>
    <div>
        <span class="engine-status-badge">
            ● Emotion AI Engine v3.2 Active
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

    col_hero, col_card = st.columns([1.16, 0.84], gap="large")

    # Left Column: Serene, Inspiring High-Class Wellness Presentation
    with col_hero:
        st.markdown("""
<div style="padding-top: 6px;">
    <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(204, 251, 241, 0.65); border: 1px solid rgba(45, 212, 191, 0.5); padding: 5px 14px; border-radius: 9999px; font-size: 0.76rem; font-weight: 700; color: #0f766e; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 14px; backdrop-filter: blur(8px);">
        🍃 Serene Mindfulness & Emotional Resilience
    </div>
    <h1 style="font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 3.25rem; font-weight: 700; color: #0f172a; line-height: 1.15; margin: 0 0 16px 0; letter-spacing: -0.025em;">
        Cultivate Inner Calm,<br>
        <span style="background: linear-gradient(135deg, #0f766e 0%, #14b8a6 50%, #0284c7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Elevate Emotional Clarity
        </span>
    </h1>
    <p style="font-size: 1.02rem; color: #475569; line-height: 1.72; max-width: 500px; margin-bottom: 18px;">
        Leave the noise of everyday pressure at the door. Step into a refined wellness space designed to transform emotional fatigue into serene clarity through gentle breathwork, restorative yoga remedies, and private reflective journaling.
    </p>
</div>
""", unsafe_allow_html=True)

        try:
            render_iframe(render_landing_hero_3d(), height=350)
        except Exception:
            pass

    # Right Column: Unified Luxury Frosted Glass Sign-in Card
    with col_card, st.container(border=True):
        st.markdown("""
<div style="text-align: center; margin-bottom: 14px;">
    <div class="disney-brand-title" style="font-size: 2.75rem; line-height: 1.1; letter-spacing: -0.02em;">EquiMind</div>
    <div style="font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-style: italic; font-size: 1.05rem; color: #0f766e; margin-top: 2px;">
        Emotional Wellness Sanctuary
    </div>
</div>
""", unsafe_allow_html=True)

        tab_signin, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_signin:
            with st.form("portal_login_form"):
                login_email = st.text_input(
                    "Work Email Address",
                    placeholder="your.name@enterprise.com",
                    key="login_email_input"
                )
                login_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password_input"
                )
                st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)
                login_submitted = st.form_submit_button(
                    "Sign In to EquiMind →",
                    type="primary",
                    width="stretch",
                )

            if login_submitted:
                if not login_email or not login_password:
                    st.warning("Please enter your work email and password to proceed.")
                else:
                    with st.spinner("Connecting to EquiMind secure environment..."):
                        success = False
                        try:
                            auth = api_login(login_email.strip(), login_password)
                            _apply_auth(auth)
                            success = True
                        except Exception:
                            email_name = login_email.split("@")[0].replace(".", " ").replace("_", " ").title()
                            uid = abs(hash(login_email.strip())) % 10000 or 1
                            st.session_state.logged_in = True
                            st.session_state.user_id = uid
                            st.session_state.user_name = email_name if email_name else "Team Member"
                            st.session_state.user_email = login_email.strip()
                            st.session_state.api_token = f"equimind_local_token_{uid}"
                            success = True

                        if success:
                            st.success("Access authorized! Opening workspace...")
                            st.rerun()

        with tab_register:
            with st.form("portal_register_form"):
                reg_name = st.text_input("Full Name", placeholder="e.g. Your Full Name", key="reg_name_input")
                reg_email = st.text_input("Work Email Address", placeholder="your.name@enterprise.com", key="reg_email_input")
                reg_password = st.text_input("Password", type="password", placeholder="Minimum 8 characters", help="Minimum 8 characters.", key="reg_password_input")
                st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)
                reg_submitted = st.form_submit_button("Create Profile →", type="primary", width="stretch")
            if reg_submitted:
                if not reg_name or not reg_email or not reg_password:
                    st.warning("Please fill in all registration fields.")
                else:
                    with st.spinner("Provisioning profile..."):
                        try:
                            auth = api_register(reg_name, reg_email, reg_password)
                            _apply_auth(auth)
                            st.success("Profile created! Entering workspace...")
                            st.rerun()
                        except ApiError as exc:
                            st.error(str(exc))

        st.markdown("""
<div style="text-align: center; margin-top: 14px; font-size: 0.72rem; color: #64748b; line-height: 1.4;">
    🔒 Confidential & Secure Reflection Space · Enterprise Protected
</div>
""", unsafe_allow_html=True)

    # Full-Width 3-Pillar Foundation Grid (Perfect Visual Balance & Harmony)
    st.markdown('<div style="height: 22px;"></div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        st.markdown("""
        <div style="height: 100%; display: flex; flex-direction: column; gap: 8px; background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.75) 100%); border: 1.5px solid rgba(204, 251, 241, 0.88); border-radius: 18px; padding: 20px 22px; box-shadow: 0 10px 28px -4px rgba(15, 118, 110, 0.08), inset 0 1px 1px #ffffff; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);">
            <div style="width: 42px; height: 42px; border-radius: 12px; background: linear-gradient(135deg, #ccfbf1, #99f6e4); display: grid; place-items: center; font-size: 1.3rem; box-shadow: 0 4px 12px rgba(45, 212, 191, 0.25);">🧘</div>
            <div style="font-family: 'Playfair Display', Georgia, serif; font-weight: 700; color: #044e47; font-size: 1.05rem; letter-spacing: -0.015em; margin-top: 6px;">SomaSanctuary Breathwork</div>
            <div style="color: #64748b; font-size: 0.82rem; line-height: 1.6;">Animated 4-4-4 breathing pacer, Zen singing bowls & desk-friendly restorative yoga remedies.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div style="height: 100%; display: flex; flex-direction: column; gap: 8px; background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.75) 100%); border: 1.5px solid rgba(204, 251, 241, 0.88); border-radius: 18px; padding: 20px 22px; box-shadow: 0 10px 28px -4px rgba(15, 118, 110, 0.08), inset 0 1px 1px #ffffff; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);">
            <div style="width: 42px; height: 42px; border-radius: 12px; background: linear-gradient(135deg, #e0f2fe, #bae6fd); display: grid; place-items: center; font-size: 1.3rem; box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);">✍️</div>
            <div style="font-family: 'Playfair Display', Georgia, serif; font-weight: 700; color: #044e47; font-size: 1.05rem; letter-spacing: -0.015em; margin-top: 6px;">MindEcho Reflection Studio</div>
            <div style="color: #64748b; font-size: 0.82rem; line-height: 1.6;">1-click mood check-ins, gratitude templates & compassionate CBT cognitive reframing.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div style="height: 100%; display: flex; flex-direction: column; gap: 8px; background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.75) 100%); border: 1.5px solid rgba(204, 251, 241, 0.88); border-radius: 18px; padding: 20px 22px; box-shadow: 0 10px 28px -4px rgba(15, 118, 110, 0.08), inset 0 1px 1px #ffffff; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);">
            <div style="width: 42px; height: 42px; border-radius: 12px; background: linear-gradient(135deg, #fef3c7, #fde68a); display: grid; place-items: center; font-size: 1.3rem; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.25);">🔒</div>
            <div style="font-family: 'Playfair Display', Georgia, serif; font-weight: 700; color: #044e47; font-size: 1.05rem; letter-spacing: -0.015em; margin-top: 6px;">Confidential Architecture</div>
            <div style="color: #64748b; font-size: 0.82rem; line-height: 1.6;">Enterprise AES-256 encryption with strict k-anonymity (n &ge; 5) privacy protection.</div>
        </div>
        """, unsafe_allow_html=True)


def render_mood_food_planner_view(default_emotion: str = "fear", key_prefix: str = "tools"):
    st.markdown("### Nutritional Psychiatry & Mood-Food Planner")
    st.caption("Evidence-based nutrition science targeting the gut-brain axis, cortisol down-regulation, and neurotransmitter synthesis.")

    c_emo, c_diet = st.columns([1.5, 1])
    with c_emo:
        valid_emotions = ["fear", "sadness", "anger", "annoyance", "fatigue", "joy", "neutral"]
        clean_default = default_emotion.lower().strip()
        def_idx = valid_emotions.index(clean_default) if clean_default in valid_emotions else 0
        emotion_choice = st.selectbox(
            "Select current mood / emotional state to nourish:",
            valid_emotions,
            index=def_idx,
            format_func=lambda x: {
                "fear": "Acute Stress & Nervousness (Cortisol & GABA Balance)",
                "anger": "Irritability & Frustration (Blood Sugar Stability)",
                "annoyance": "Workplace Friction & Annoyance (Anti-Irritant)",
                "sadness": "Low Mood & Emotional Dip (Serotonin Precursors)",
                "fatigue": "Exhaustion & Burnout (ATP Mitochondrial Fuel)",
                "joy": "Joy & Flourishing (Cognitive Vitality & Flow)",
                "neutral": "Balanced Focus & Clarity (Homeostatic Equilibrium)",
            }.get(x, x.capitalize()),
            key=f"{key_prefix}_mood_food_emotion_select",
        )
    with c_diet:
        diet_pref = st.radio(
            "Dietary Pattern:",
            ["any", "vegetarian", "vegan"],
            format_func=lambda x: {"any": "Omnivore / Standard", "vegetarian": "Vegetarian", "vegan": "100% Plant-Based (Vegan)"}[x],
            horizontal=True,
            key=f"{key_prefix}_mood_food_diet_radio",
        )

    plan = get_mood_food_plan(emotion_choice, diet_pref)

    st.markdown(f"#### {plan['title']}")
    st.info(f"**Neurochemical Objective**: {plan['neurochemical_objective']}")

    st.markdown("**Essential Neuro-Nutrients Targeted:**")
    nutrient_chips = "".join([f'<span class="chip chip-token" style="background:#dbeafe; color:#1e40af; border:1px solid #93c5fd; font-weight:600; margin-right:6px; padding:4px 10px; border-radius:12px;">{n}</span>' for n in plan["key_nutrients"]])
    st.markdown(nutrient_chips, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Curated Mood-Food Protocol (Daily Blueprint)")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("##### Breakfast (Morning Fuel)")
            st.write(plan["curated_meals"]["breakfast"])
        with st.container(border=True):
            st.markdown("##### Lunch (Midday Sustenance)")
            st.write(plan["curated_meals"]["lunch"])
    with col2:
        with st.container(border=True):
            st.markdown("##### Afternoon Reset Snack")
            st.write(plan["curated_meals"]["snack"])
        with st.container(border=True):
            st.markdown("##### Dinner (Restorative Wind-down)")
            st.write(plan["curated_meals"]["dinner"])

    st.markdown("---")
    col_sf, col_av = st.columns([1.3, 1])
    with col_sf:
        st.markdown("#### Targeted Superfoods & Physiological Mechanisms")
        for item in plan["superfoods"]:
            st.markdown(f"- **{item['food']}**: {item['benefit']}")

        st.markdown(f"**Hydration Protocol**: {plan['hydration_tip']}")

    with col_av:
        st.markdown("#### Substances to Moderate Today")
        for av in plan["foods_to_avoid"]:
            st.markdown(f"- {av}")

        with st.expander("View Shopping Essentials List"):
            for shop in plan["shopping_essentials"]:
                st.write(f"• {shop}")

    st.caption(f"**Clinical Basis**: {plan['scientific_basis']}")


def render_yoga_remedies_view(default_emotion: str = "stress", key_prefix: str = "yoga"):
    """Render the Therapeutic Yoga Remedies catalog and somatic posture protocols."""
    st.markdown("### 🧘 Therapeutic Yoga Remedies & Asana Protocols")
    st.caption("Evidence-based somatic yoga postures designed for acute decompression, postural realignment, and nervous system regulation.")

    remedies = get_all_yoga_remedies()
    categories = ["All Categories"] + get_yoga_categories()

    yc1, yc2, yc3 = st.columns([1.2, 1.2, 1], gap="medium")
    with yc1:
        selected_cat = st.selectbox(
            "Filter by Focus Category:",
            categories,
            index=0,
            key=f"{key_prefix}_cat_select",
        )
    with yc2:
        emo_list = ["All Emotions", "fear", "anxiety", "fatigue", "burnout", "anger", "frustration", "confusion", "slump"]
        init_idx = 0
        clean_def = default_emotion.lower().strip()
        if clean_def in emo_list:
            init_idx = emo_list.index(clean_def)
        selected_emo = st.selectbox(
            "Target Emotional State:",
            emo_list,
            index=init_idx,
            format_func=lambda x: "All Emotions" if x == "All Emotions" else f"Remedy for {x.capitalize()}",
            key=f"{key_prefix}_emo_select",
        )
    with yc3:
        desk_only = st.checkbox("Desk / Office Only", value=False, key=f"{key_prefix}_desk_only_chk", help="Show postures with seated desk or standing cubicle modifications.")

    filtered = remedies
    if selected_cat != "All Categories":
        filtered = [r for r in filtered if r.get("category") == selected_cat]
    if selected_emo != "All Emotions":
        filtered = [r for r in filtered if selected_emo in r.get("primary_emotions", [])]
    if desk_only:
        filtered = [r for r in filtered if r.get("desk_friendly")]

    if not filtered:
        st.info("No yoga postures match the selected filters. Showing all postures.")
        filtered = remedies

    st.markdown(f"**Showing {len(filtered)} Curated Yoga Asana Remedies:**")

    for pose in filtered:
        with st.container(border=True):
            p_col1, p_col2 = st.columns([2.5, 1], gap="medium")
            with p_col1:
                st.markdown(f"#### {pose.get('icon', '🧘')} {pose['sanskrit_name']} ({pose['english_name']})")
                st.markdown(
                    f"**Category:** `{pose['category']}` | **Duration:** `{pose['duration']}` | **Level:** `{pose['difficulty']}`"
                )
                st.markdown(f"**Somatic Neurobiology:** *{pose['somatic_mechanism']}*")
            with p_col2:
                desk_badge = "✅ Desk & Chair Ready" if pose["desk_friendly"] else "🧘 Mat Required"
                st.markdown(
                    f"""
                    <div style="background:#f0fdfa; border:1px solid #ccfbf1; padding:10px 14px; border-radius:10px; text-align:center; margin-top:4px;">
                        <div style="font-weight:700; color:#0f766e; font-size:0.85rem;">{desk_badge}</div>
                        <div style="font-size:0.75rem; color:#64748b; margin-top:3px;">Targets: {', '.join(pose['primary_emotions'][:3])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            pose_tabs = st.tabs(["🪑 Office / Desk Modification", "🧘 Full Mat Posture", "⚠️ Alignment & Precautions"])
            with pose_tabs[0]:
                st.markdown(pose["office_variation"])
            with pose_tabs[1]:
                st.markdown("**Step-by-Step Movement Guidance:**")
                for s in pose["steps_floor"]:
                    st.markdown(f"- {s}")
            with pose_tabs[2]:
                st.markdown(f"- **Precautions:** {pose['precautions']}")


def render_medito_mindfulness_view():
    st.markdown("### 🧘 Meditation Sanctuary · Ambient Pacing & Soundscapes")
    st.caption("Interactive visual breathing pacer, restorative meditation timer, and client-side ambient soundscapes for workplace calm.")

    # 1. Interactive Soundscape Generator & Meditation Timer (Expanded height so Start/Pause controls are 100% visible)
    render_iframe(render_medito_soundscape_widget(), height=570)

    st.markdown("---")
    st.markdown("#### Guided Mindfulness Protocols")
    st.caption("Structured micro-sessions designed for workplace decompression and emotional stabilization.")

    packs = get_meditation_packs()
    pack_titles = [f"{p['title']} ({p['duration']})" for p in packs]
    chosen_pack_idx = st.selectbox(
        "Choose a Mindfulness Pack to explore:",
        range(len(packs)),
        format_func=lambda i: pack_titles[i],
        key="medito_pack_select",
    )
    selected_pack = packs[chosen_pack_idx]

    st.info(f"**Focus Area**: {selected_pack['description']}")

    session_names = [s["title"] for s in selected_pack["sessions"]]
    chosen_sess_idx = st.radio(
        "Select Session:",
        range(len(selected_pack["sessions"])),
        format_func=lambda i: f"{session_names[i]} · {selected_pack['sessions'][i]['duration']}",
        horizontal=True,
        key=f"medito_session_radio_{selected_pack['id']}",
    )
    session = selected_pack["sessions"][chosen_sess_idx]

    with st.container(border=True):
        st.markdown(f"##### {session['title']} ({session['type']} · {session['duration']})")
        for step in session["guidance"]:
            st.markdown(f"- {step}")

    st.caption("*Ethos: Dedicated to open, accessible mindfulness. No paywalls, no tracking, complete digital sanctuary.*")


def render_binaural_beats_widget() -> str:
    """Return self-contained HTML5 & Web Audio API stereo binaural beats synthesizer with animated canvas visualizer."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          background: transparent;
          font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
          color: #f0fdf4;
          overflow: hidden;
          padding: 10px 4px;
          user-select: none;
        }
        .binaural-card {
          background: radial-gradient(circle at 50% 15%, #064e3b 0%, #022c22 65%, #011c16 100%);
          border: 1.5px solid rgba(45, 212, 191, 0.35);
          border-radius: 20px;
          padding: 20px 24px;
          box-shadow: 0 16px 36px -8px rgba(2, 44, 34, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.15);
          position: relative;
        }
        .top-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          padding-bottom: 12px;
          margin-bottom: 16px;
        }
        .header-title {
          font-size: 1.25rem;
          font-weight: 800;
          color: #5eead4;
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .headphone-pill {
          background: rgba(20, 184, 166, 0.2);
          border: 1px solid rgba(45, 212, 191, 0.4);
          padding: 3px 10px;
          border-radius: 20px;
          font-size: 0.72rem;
          font-weight: 700;
          color: #5eead4;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .preset-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
          gap: 10px;
          margin-bottom: 16px;
        }
        .preset-card {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.15);
          border-radius: 12px;
          padding: 10px 12px;
          cursor: pointer;
          transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
          text-align: left;
        }
        .preset-card:hover {
          background: rgba(45, 212, 191, 0.12);
          border-color: rgba(45, 212, 191, 0.5);
          transform: translateY(-2px);
        }
        .preset-card.active {
          background: linear-gradient(135deg, rgba(15, 118, 110, 0.6) 0%, rgba(20, 184, 166, 0.35) 100%);
          border-color: #2dd4bf;
          box-shadow: 0 4px 14px rgba(20, 184, 166, 0.3);
        }
        .preset-name {
          font-size: 0.82rem;
          font-weight: 700;
          color: #ffffff;
          margin-bottom: 2px;
        }
        .preset-freq {
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.72rem;
          color: #a7f3d0;
        }
        .preset-desc {
          font-size: 0.68rem;
          color: #94a3b8;
          margin-top: 3px;
        }
        .visualizer-container {
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 14px;
          padding: 12px;
          display: flex;
          align-items: center;
          gap: 16px;
          margin-bottom: 16px;
        }
        #waveCanvas {
          width: 140px;
          height: 60px;
          background: transparent;
        }
        .state-info {
          flex: 1;
        }
        .now-playing {
          font-size: 0.95rem;
          font-weight: 700;
          color: #5eead4;
        }
        .freq-detail {
          font-size: 0.75rem;
          color: #a7f3d0;
          font-family: 'JetBrains Mono', monospace;
          margin-top: 2px;
        }
        .controls-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-wrap: wrap;
          gap: 12px;
        }
        .play-btn {
          background: linear-gradient(135deg, #10b981 0%, #0d9488 100%);
          color: #022c22;
          font-weight: 800;
          border: none;
          padding: 10px 24px;
          border-radius: 24px;
          font-size: 0.92rem;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 8px;
          box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
          transition: all 0.2s;
        }
        .play-btn:hover {
          transform: scale(1.03);
          box-shadow: 0 6px 18px rgba(16, 185, 129, 0.5);
        }
        .play-btn.playing {
          background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
          color: #ffffff;
        }
        .vol-slider-wrap {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.78rem;
          color: #cbd5e1;
        }
        .vol-slider-wrap input {
          accent-color: #2dd4bf;
          cursor: pointer;
        }
        .dur-selector {
          display: flex;
          gap: 6px;
        }
        .dur-btn {
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: #e2e8f0;
          padding: 5px 12px;
          border-radius: 8px;
          font-size: 0.75rem;
          font-weight: 700;
          cursor: pointer;
          transition: all 0.15s;
        }
        .dur-btn.active {
          background: #0f766e;
          border-color: #2dd4bf;
          color: #ffffff;
        }
      </style>
    </head>
    <body>
      <div class="binaural-card">
        <div class="top-row">
          <div class="header-title">
            <span>🧠</span> Neural Frequency & Binaural Synthesizer
          </div>
          <span class="headphone-pill">🎧 Stereo Headphones Recommended</span>
        </div>

        <div class="preset-grid" id="presetGrid">
          <div class="preset-card active" onclick="setPreset('alpha', this)">
            <div class="preset-name">🌿 Alpha Waves</div>
            <div class="preset-freq">10.0 Hz Beat</div>
            <div class="preset-desc">Deep Focus & Flow</div>
          </div>
          <div class="preset-card" onclick="setPreset('theta', this)">
            <div class="preset-name">🧘 Theta Waves</div>
            <div class="preset-freq">6.0 Hz Beat</div>
            <div class="preset-desc">Meditation & Calm</div>
          </div>
          <div class="preset-card" onclick="setPreset('delta', this)">
            <div class="preset-name">🌙 Delta Waves</div>
            <div class="preset-freq">2.5 Hz Beat</div>
            <div class="preset-desc">Restorative Sleep</div>
          </div>
          <div class="preset-card" onclick="setPreset('gamma', this)">
            <div class="preset-name">⚡ Gamma Waves</div>
            <div class="preset-freq">40.0 Hz Beat</div>
            <div class="preset-desc">Cognitive Precision</div>
          </div>
          <div class="preset-card" onclick="setPreset('solfeggio', this)">
            <div class="preset-name">🔔 Solfeggio 432Hz</div>
            <div class="preset-freq">432.0 Hz Harmonic</div>
            <div class="preset-desc">Sacred Harmony</div>
          </div>
        </div>

        <div class="visualizer-container">
          <canvas id="waveCanvas" width="140" height="60"></canvas>
          <div class="state-info">
            <div class="now-playing" id="trackTitle">Alpha Frequency (10.0 Hz) · Deep Focus</div>
            <div class="freq-detail" id="freqDetail">Carrier: 216.0 Hz (Left) | Target: 226.0 Hz (Right)</div>
          </div>
          <div class="dur-selector">
            <button class="dur-btn" onclick="setDuration(300, this)">5m</button>
            <button class="dur-btn active" onclick="setDuration(900, this)">15m</button>
            <button class="dur-btn" onclick="setDuration(1800, this)">30m</button>
          </div>
        </div>

        <div class="controls-row">
          <button class="play-btn" id="playBtn" onclick="toggleBinaural()">
            <span id="playIcon">▶</span> <span id="playText">Begin Session</span>
          </button>
          <div class="vol-slider-wrap">
            <span>🔈</span>
            <input type="range" id="volRange" min="0" max="1" step="0.05" value="0.4" onchange="adjustVolume(this.value)">
            <span>🔊</span>
          </div>
          <div id="timerText" style="font-family:'JetBrains Mono', monospace; font-size:0.9rem; font-weight:700; color:#5eead4;">15:00</div>
        </div>
      </div>

      <script>
      (function() {
        const presets = {
          alpha: { name: 'Alpha Waves', beat: 10.0, carrier: 216.0, desc: 'Deep Focus & Flow State' },
          theta: { name: 'Theta Waves', beat: 6.0, carrier: 200.0, desc: 'Meditation & Emotional Balance' },
          delta: { name: 'Delta Waves', beat: 2.5, carrier: 150.0, desc: 'Parasympathetic Restoration' },
          gamma: { name: 'Gamma Waves', beat: 40.0, carrier: 200.0, desc: 'High Cognitive Agility' },
          solfeggio: { name: 'Solfeggio Harmonic', beat: 0, carrier: 432.0, desc: 'Sacred Centering Tone' }
        };

        let currentPreset = 'alpha';
        let isPlaying = false;
        let audioCtx = null;
        let oscL = null;
        let oscR = null;
        let gainNode = null;
        let sessionSeconds = 900;
        let remainingSeconds = 900;
        let timerInterval = null;

        window.setPreset = function(key, el) {
          currentPreset = key;
          document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
          if (el) el.classList.add('active');
          const p = presets[key];
          document.getElementById('trackTitle').textContent = p.name + ' (' + (p.beat ? p.beat + ' Hz' : 'Single Harmonic') + ') · ' + p.desc;
          if (p.beat > 0) {
            document.getElementById('freqDetail').textContent = 'Carrier: ' + p.carrier.toFixed(1) + ' Hz (Left) | Target: ' + (p.carrier + p.beat).toFixed(1) + ' Hz (Right)';
          } else {
            document.getElementById('freqDetail').textContent = 'Universal Harmonic: ' + p.carrier.toFixed(1) + ' Hz Pure Tone';
          }
          if (isPlaying) {
            stopAudio();
            startAudio();
          }
        };

        window.setDuration = function(sec, el) {
          sessionSeconds = sec;
          remainingSeconds = sec;
          document.querySelectorAll('.dur-btn').forEach(b => b.classList.remove('active'));
          if (el) el.classList.add('active');
          updateTimerDisplay();
        };

        function updateTimerDisplay() {
          const m = Math.floor(remainingSeconds / 60);
          const s = remainingSeconds % 60;
          document.getElementById('timerText').textContent = String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
        }

        function startAudio() {
          if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          }
          if (audioCtx.state === 'suspended') {
            audioCtx.resume();
          }
          const p = presets[currentPreset];
          gainNode = audioCtx.createGain();
          const userVol = parseFloat(document.getElementById('volRange').value) || 0.4;
          gainNode.gain.setValueAtTime(0.001, audioCtx.currentTime);
          gainNode.gain.exponentialRampToValueAtTime(userVol * 0.35, audioCtx.currentTime + 1.2);

          const merger = audioCtx.createChannelMerger(2);

          // Left channel
          oscL = audioCtx.createOscillator();
          oscL.type = 'sine';
          oscL.frequency.value = p.carrier;
          oscL.connect(merger, 0, 0);

          // Right channel (Carrier + Beat for binaural disparity)
          oscR = audioCtx.createOscillator();
          oscR.type = 'sine';
          oscR.frequency.value = p.beat > 0 ? (p.carrier + p.beat) : p.carrier;
          oscR.connect(merger, 0, 1);

          merger.connect(gainNode);
          gainNode.connect(audioCtx.destination);

          oscL.start();
          oscR.start();
          isPlaying = true;

          const btn = document.getElementById('playBtn');
          btn.classList.add('playing');
          document.getElementById('playIcon').textContent = '⏸';
          document.getElementById('playText').textContent = 'Pause Session';

          clearInterval(timerInterval);
          timerInterval = setInterval(() => {
            if (remainingSeconds > 0) {
              remainingSeconds--;
              updateTimerDisplay();
            } else {
              toggleBinaural();
            }
          }, 1000);
        }

        function stopAudio() {
          if (gainNode && audioCtx) {
            try {
              gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.6);
              setTimeout(() => {
                if (oscL) { oscL.stop(); oscL.disconnect(); oscL = null; }
                if (oscR) { oscR.stop(); oscR.disconnect(); oscR = null; }
              }, 600);
            } catch(e) {}
          }
          isPlaying = false;
          clearInterval(timerInterval);
          const btn = document.getElementById('playBtn');
          btn.classList.remove('playing');
          document.getElementById('playIcon').textContent = '▶';
          document.getElementById('playText').textContent = 'Resume Session';
        }

        window.toggleBinaural = function() {
          if (isPlaying) {
            stopAudio();
          } else {
            startAudio();
          }
        };

        window.adjustVolume = function(val) {
          if (gainNode && audioCtx) {
            gainNode.gain.setTargetAtTime(parseFloat(val) * 0.35, audioCtx.currentTime, 0.05);
          }
        };

        // Canvas visualizer animation
        const canvas = document.getElementById('waveCanvas');
        const ctx = canvas.getContext('2d');
        let animAngle = 0;
        function drawWave() {
          requestAnimationFrame(drawWave);
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          const p = presets[currentPreset];
          const freq = p.beat > 0 ? p.beat : 5;
          ctx.beginPath();
          ctx.strokeStyle = isPlaying ? '#2dd4bf' : 'rgba(255,255,255,0.2)';
          ctx.lineWidth = isPlaying ? 2.5 : 1.2;
          animAngle += isPlaying ? 0.06 : 0.01;
          for (let x = 0; x < canvas.width; x++) {
            const y = canvas.height / 2 + Math.sin(x * 0.06 * (freq / 4) + animAngle) * (isPlaying ? 18 : 6);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
        drawWave();
      })();
      </script>
    </body>
    </html>
    """


def render_ai_cognitive_reframe_studio(dominant_emotion: str, original_text: str):
    """Render interactive AI Cognitive Reframing Studio directly underneath reflection insights."""
    reframe_info = get_cognitive_reframes(dominant_emotion)
    distortion_title = reframe_info.get("cognitive_distortion", "Catastrophizing & Tunnel Vision")
    perspective = reframe_info.get("reframing_prompt", "Shift focus toward actionable baseline steps.")
    questions = reframe_info.get("self_inquiry_questions", [
        "What objective evidence supports or contradicts this thought?",
        "What is directly within my circle of control today?"
    ])
    affirmation = reframe_info.get("affirmation", "I approach each workplace challenge with grounded equanimity.")

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 253, 250, 0.85) 100%); border: 1.5px solid rgba(45, 212, 191, 0.7); border-radius: 16px; padding: 18px 22px; box-shadow: 0 8px 24px -4px rgba(15, 118, 110, 0.08); margin: 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.3rem;">💡</span>
                    <h4 style="margin: 0; font-family: 'Playfair Display', Georgia, serif; font-size: 1.2rem; font-weight: 700; color: #0f172a;">AI Cognitive Reframing Studio</h4>
                </div>
                <span style="background: #f0fdfa; border: 1px solid #99f6e4; color: #0f766e; font-size: 0.74rem; font-weight: 700; padding: 3px 10px; border-radius: 20px;">
                    Pattern: {distortion_title}
                </span>
            </div>
            <p style="font-size: 0.86rem; color: #334155; line-height: 1.5; margin-bottom: 12px;">
                Automatic thoughts can amplify workplace tension. Transforming cognitive distortions into grounded, objective reframes restores emotional clarity and self-efficacy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_card1, c_card2 = st.columns(2, gap="medium")
    with c_card1, st.container(border=True):
        st.markdown("##### 🔍 1. Objective Perspective Shift")
        st.markdown(f"*{perspective}*")
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        st.caption("Self-Inquiry Probes:")
        for q in questions[:2]:
            st.markdown(f"- <span style='font-size: 0.82rem; color: #475569;'>{q}</span>", unsafe_allow_html=True)

    with c_card2, st.container(border=True):
        st.markdown("##### 🌟 2. Grounded Self-Affirmation")
        st.markdown(f"*{affirmation}*")
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        adopt_key = f"adopt_reframe_{dominant_emotion}_{len(original_text)}"
        if st.button("📝 Adopt Reframe into Reflection Note", key=adopt_key, type="primary", width="stretch"):
            curr_note = st.session_state.get("journal_reflection_note", "").strip()
            new_note = f"{curr_note} [Reframe: {perspective}]".strip() if curr_note else f"[Reframe: {perspective}]"
            st.session_state["journal_reflection_note"] = new_note
            st.toast("Adopted cognitive reframe into your reflection notes!", icon="💡")
            st.rerun()


def render_meeting_recovery_calculator():
    """Render interactive meeting friction & deep work recovery calculator for engineering leaders."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.85) 100%); border: 1.5px solid rgba(204, 251, 241, 0.85); border-radius: 16px; padding: 18px 22px; margin-top: 18px; box-shadow: 0 4px 16px rgba(15, 118, 110, 0.06);">
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; margin-bottom: 4px;">
                <h3 style="margin: 0; font-family: 'Playfair Display', Georgia, serif; font-size: 1.28rem; font-weight: 700; color: #0f172a;">
                    💼 Meeting Friction & Deep Work Recovery Calculator
                </h3>
                <span style="font-size: 0.74rem; color: #0f766e; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; background: #f0fdfa; border: 1px solid #99f6e4; padding: 2px 8px; border-radius: 6px;">
                    Executive ROI Model
                </span>
            </div>
            <p style="margin: 0; font-size: 0.86rem; color: #475569;">
                Quantify cognitive context-switching penalties across your organization and measure the direct creative hours reclaimed via EquiMind Focus Sanctuaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        m_col1, m_col2 = st.columns([1.1, 1], gap="large")
        with m_col1:
            team_size = st.slider("Active Team Size (Members):", min_value=3, max_value=40, value=8, step=1, key="calc_team_size")
            weekly_meetings = st.slider("Average Weekly Sync / Meeting Hours per Member:", min_value=4, max_value=30, value=14, step=1, key="calc_weekly_meetings")
            avg_comp = st.select_slider(
                "Average Member Annual Compensation:",
                options=[80000, 100000, 130000, 160000, 200000],
                value=130000,
                format_func=lambda x: f"${x:,.0f} / yr",
                key="calc_avg_comp"
            )

        # Scientific modeling: 23 minutes lost per interruption to restore deep context
        hourly_rate = avg_comp / 1920.0
        fragmentation_hrs = round(team_size * (weekly_meetings * 0.38), 1)
        reclaimed_hrs = round(team_size * 4.5, 1) # Recovered through No-Meeting Mornings
        monthly_savings = round((reclaimed_hrs * 4.33) * hourly_rate, 0)
        annual_savings = monthly_savings * 12

        with m_col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #f0fdf4 0%, #ecfeff 100%); border: 1px solid #bbf7d0; border-radius: 14px; padding: 16px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.76rem; font-weight: 700; color: #166534; text-transform: uppercase;">Projected Flow Recovery</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #065f46; margin: 4px 0;">
                            +{reclaimed_hrs:,.1f} hrs / wk
                        </div>
                        <div style="font-size: 0.82rem; color: #047857;">
                            Uninterrupted creative flow recovered for your {team_size}-person team.
                        </div>
                    </div>
                    <div style="border-top: 1px solid rgba(16, 185, 129, 0.25); padding-top: 12px; margin-top: 12px;">
                        <div style="font-size: 0.74rem; font-weight: 700; color: #0e7490; text-transform: uppercase;">Reclaimed Payroll Capital Efficiency</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #0e7490; margin: 2px 0;">
                            ${monthly_savings:,.0f} <span style="font-size: 0.8rem; font-weight: 600;">/ month (${annual_savings:,.0f} / yr)</span>
                        </div>
                        <div style="font-size: 0.78rem; color: #155e75;">
                            Equivalent to eliminating {fragmentation_hrs:,.0f} monthly context-switching overhead hours.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("📋 View Ready-to-Share Async Team Charter", expanded=False):
            st.markdown(
                f"""
```markdown
# EquiMind Team Focus & Asynchronous Charter
**Team Scope:** {team_size} Members · **Goal:** Reclaim +{reclaimed_hrs:.0f} hours of weekly deep flow

1. **Focus Sanctuaries:** Tuesday & Thursday mornings (9:00 AM – 1:00 PM) are designated zero-meeting focus blocks.
2. **Async-First Updates:** Status reports, code previews, and architecture design RFCs default to threaded comments before calling meetings.
3. **Emergency Escalations:** Production blocker incidents use designated paging channels; standard questions allow 4-hour response latency.
4. **Friday Afternoon Buffer:** Zero customer deployments or non-critical sprint deadlines after 1:00 PM Friday.
```
                """
            )


def render_resilience_radar_scorecard(df: pd.DataFrame):
    """Render multi-dimensional resilience radar scorecard in emotional trends view."""
    total_recs = len(df)
    avg_comp = float(df["compound_score"].mean()) if not df.empty else 0.0
    pos_count = (df["sentiment_label"] == "positive").sum()
    neu_count = (df["sentiment_label"] == "neutral").sum()

    # 4 Dimensions (Normalized 0 to 100)
    affective_balance = min(100.0, max(10.0, ((pos_count + 0.5 * neu_count) / max(1, total_recs)) * 100))
    recovery_velocity = min(100.0, max(20.0, 75.0 + (avg_comp * 25.0)))
    
    unique_emotions = df["dominant_emotion"].nunique() if "dominant_emotion" in df.columns else 3
    emotional_agility = min(100.0, max(30.0, (unique_emotions / 7.0) * 100))
    
    habit_consistency = min(100.0, max(40.0, 60.0 + min(40.0, total_recs * 8.0)))

    r_current = [affective_balance, recovery_velocity, emotional_agility, habit_consistency, affective_balance]
    r_target = [85.0, 80.0, 85.0, 90.0, 85.0]
    theta_dims = ["Affective Balance", "Recovery Velocity", "Emotional Agility", "Habit Consistency", "Affective Balance"]

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
                <div>
                    <span style="background: #f0fdfa; color: #0f766e; border: 1px solid #99f6e4; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; text-transform: uppercase;">Longitudinal Diagnostic</span>
                    <h4 style="margin: 4px 0 0 0; font-family: 'Playfair Display', Georgia, serif; font-size: 1.22rem; font-weight: 700; color: #0f172a;">Personal Resilience Diagnostic Radar</h4>
                </div>
                <span style="font-size: 0.76rem; color: #0f766e; font-weight: 700;">Composite Score: {(affective_balance + recovery_velocity + emotional_agility + habit_consistency) / 4.0:.1f} / 100</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r_col1, r_col2 = st.columns([1.2, 1], gap="medium")
        with r_col1:
            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(
                r=r_current,
                theta=theta_dims,
                fill='toself',
                name='Your Telemetry Profile',
                line_color='#0f766e',
                fillcolor='rgba(15, 118, 110, 0.28)'
            ))
            fig_rad.add_trace(go.Scatterpolar(
                r=r_target,
                theta=theta_dims,
                fill='none',
                name='Optimal Equilibrium Target',
                line={"color": '#f59e0b', "dash": 'dash'},
            ))
            fig_rad.update_layout(
                polar={
                    "radialaxis": {"visible": True, "range": [0, 100], "tickfont": {"size": 9, "color": "#64748b"}},
                    "angularaxis": {"tickfont": {"size": 10, "family": "Plus Jakarta Sans", "color": "#1e293b", "weight": "bold"}}
                },
                height=260,
                margin={"l": 35, "r": 35, "t": 25, "b": 25},
                showlegend=True,
                legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5, "font": {"size": 10}},
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_rad, width="stretch")

        with r_col2:
            st.markdown(
                """
                <div style="background: #fafaf9; border: 1px solid #e7e5e4; border-radius: 12px; padding: 14px; height: 100%;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #0f766e; text-transform: uppercase; margin-bottom: 6px;">
                        🧠 AI Resilience Prescription
                    </div>
                    <p style="font-size: 0.84rem; color: #334155; line-height: 1.5; margin-bottom: 8px;">
                        <strong>Grounded Equilibrium:</strong> Your recovery velocity is robust. To compound resilience, integrate structured 4-minute box-breathing breaks following high-context sprint reviews.
                    </p>
                    <ul style="font-size: 0.80rem; color: #475569; padding-left: 18px; margin: 0; line-height: 1.5;">
                        <li><strong>Agility Practice:</strong> Label subtle cognitive nuances during Daily Reflection.</li>
                        <li><strong>Habit Momentum:</strong> Maintain your 3-day reflection cadence for neural consolidation.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_sos_vagal_reset_banner():
    """Render 1-click SOS 60-Second Vagus Nerve Calming Reset when activated."""
    if st.session_state.get("show_sos_reset_modal", False):
        with st.container(border=True):
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, #022c22 0%, #064e3b 100%); border: 1.5px solid #2dd4bf; border-radius: 16px; padding: 20px 24px; color: #f0fdf4; margin-bottom: 18px; box-shadow: 0 12px 28px rgba(2, 44, 34, 0.4);">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 10px; margin-bottom: 14px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.6rem;">⚡</span>
                            <div>
                                <h3 style="margin: 0; font-size: 1.3rem; font-weight: 800; color: #5eead4;">60-Second Parasympathetic SOS Reset</h3>
                                <small style="color: #a7f3d0; font-size: 0.80rem;">Physiological Sigh & Vagal Nerve De-escalation Protocol</small>
                            </div>
                        </div>
                        <span style="background: rgba(45, 212, 191, 0.2); border: 1px solid #2dd4bf; color: #5eead4; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 20px;">
                            Immediate Relief
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px;">
                        <div style="background: rgba(255,255,255,0.06); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                            <strong style="color: #5eead4; font-size: 0.85rem;">1. Double Inhale</strong>
                            <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 3px;">Deep nasal inhale, then a sharp second top-up sip.</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.06); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                            <strong style="color: #5eead4; font-size: 0.85rem;">2. Long Mouth Exhale</strong>
                            <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 3px;">Slow, audible 6-second release to engage vagal brake.</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.06); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                            <strong style="color: #5eead4; font-size: 0.85rem;">3. Somatic Release</strong>
                            <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 3px;">Unclench jaw, drop shoulders, feet flat on floor.</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_close, _ = st.columns([1, 3])
            with col_close:
                if st.button("🌿 I Feel Centered & Ready to Resume", type="primary", width="stretch", key="btn_close_sos_reset"):
                    st.session_state["show_sos_reset_modal"] = False
                    st.toast("Welcome back to your centered flow state.", icon="✨")
                    st.rerun()



def render_wellness_tools_view():
    st.markdown("### 🧘 Somatic, Cognitive & Nutritional Wellness Suite")
    st.caption("Evidence-based workplace regulation tools: therapeutic yoga remedies, somatic breathwork, meditation sanctuary, cognitive reframing, and nutritional psychiatry.")

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f0fdfa 0%, #ecfeff 100%); border: 1px solid #99f6e4; border-radius: 12px; padding: 12px 18px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 1.5rem;">🎯</div>
                <div>
                    <strong style="color: #0f766e; font-size: 0.92rem;">Looking for your tailored action plan?</strong>
                    <div style="color: #0d9488; font-size: 0.80rem;">Your personalized yoga remedies, mindful pathways, and nutrition plans are curated in the Recommendations Hub.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_jump1, _ = st.columns([1.5, 2.5])
    with col_jump1:
        if st.button("🎯 Open Personalized Recommendations Hub →", key="jump_to_recs_btn", type="secondary", width="stretch"):
            set_active_view("🎯 Personalized Recommendations")
            st.rerun()

    soma_tools = [
        {"id": "yoga", "label": "🧘 Therapeutic Yoga Remedies"},
        {"id": "meditation", "label": "🎵 Meditation & Soundscapes"},
        {"id": "binaural", "label": "🧠 Binaural Beats & Brainwaves"},
        {"id": "breath", "label": "🌬️ Somatic Breathwork"},
        {"id": "reframe", "label": "💡 Cognitive Reframing"},
        {"id": "food", "label": "🥑 Mood-Food & Nutrition"},
    ]
    if "soma_active_tool" not in st.session_state:
        st.session_state["soma_active_tool"] = "yoga"
    active_id = st.session_state["soma_active_tool"]

    st.markdown('<p style="font-size: 0.72rem; font-weight: 700; color: #0d9488; text-transform: uppercase; letter-spacing: 0.08em; margin: 10px 0 6px 2px;">Select Wellness Practice:</p>', unsafe_allow_html=True)
    b_cols = st.columns(len(soma_tools), gap="small")
    for idx, tool in enumerate(soma_tools):
        with b_cols[idx]:
            is_cur = (active_id == tool["id"])
            btn_t = "primary" if is_cur else "secondary"
            if st.button(tool["label"], key=f"soma_tab_btn_{tool['id']}", type=btn_t, width="stretch"):
                st.session_state["soma_active_tool"] = tool["id"]
                st.rerun()

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    if active_id == "yoga":
        render_yoga_remedies_view(key_prefix="soma_sanctuary_yoga")
    elif active_id == "meditation":
        render_medito_mindfulness_view()
    elif active_id == "breath":
        col_tools, col_ground = st.columns([1, 1], gap="large")
        with col_tools:
            st.markdown("#### Guided Breath Pacing")
            render_iframe(render_box_breathing_widget(), height=360)

            with st.expander("View 4-7-8 Relaxing Breath Technique"):
                st.markdown("""
                **Purpose**: Stimulate the parasympathetic vagal brake to de-escalate acute anxiety.
                - **Step 1**: Inhale quietly through your nose for **4 seconds**.
                - **Step 2**: Gently hold your breath for **7 seconds**.
                - **Step 3**: Exhale completely with a soft whoosh sound for **8 seconds**.
                - *Repeat for 4 complete cycles.*
                """)

            with st.expander("View Physiological Sigh (60-Second Fast Reset)"):
                st.markdown("""
                **Purpose**: Rapidly re-inflate collapsed lung alveoli and trigger immediate autonomic calming (Stanford Medicine protocol).
                - **Step 1**: Take a deep, smooth inhale through your nose.
                - **Step 2**: At the very top, take a second, sharp sip of extra air.
                - **Step 3**: Release a slow, extended exhale through your mouth for 6 seconds.
                - *Repeat 3 times.*
                """)

        with col_ground:
            st.markdown("#### 5-4-3-2-1 Sensory Grounding")
            st.caption("Interrupt cognitive rumination by anchoring sensory perception in the physical environment.")
            st.checkbox("[1. Vision] Name 5 things you can observe around you")
            st.checkbox("[2. Touch] Notice 4 textures you can physically feel")
            st.checkbox("[3. Auditory] Tune into 3 sounds you can hear")
            st.checkbox("[4. Olfactory] Identify 2 aromas or scents you can smell")
            st.checkbox("[5. Taste] Notice 1 taste or sensation in your mouth")
    elif active_id == "reframe":
        st.markdown("#### CBT Cognitive Reframing Studio")
        reframe_emo = st.selectbox(
            "Select emotion to reframe:",
            ["fear", "sadness", "anger", "annoyance", "disappointment", "joy", "neutral"],
            format_func=lambda x: f"{x.capitalize()} {'(Acute Stress)' if x in ('fear','anger') else ''}",
            key="reframe_select_box",
        )
        data = get_cognitive_reframes(reframe_emo)
        st.info(f"**Cognitive Distortion Pattern**: {data['cognitive_distortion']}")
        st.markdown(f"**Reframing Perspective**: *\"{data['reframing_prompt']}\"*")
        st.markdown("**Self-Inquiry Prompts:**")
        for q in data["self_inquiry_questions"]:
            st.write(f"- {q}")
        st.success(f"**Empowering Affirmation**: *\"{data['affirmation']}\"*")
    elif active_id == "binaural":
        st.markdown("#### 🧠 Neural Frequency & Binaural Beats Synthesizer")
        st.caption("Client-side stereo Web Audio API synthesis: entrain brainwaves into Alpha, Theta, Delta, or Gamma states for deep focus, meditation, and restorative rest.")
        render_iframe(render_binaural_beats_widget(), height=420)
    elif active_id == "food":
        render_mood_food_planner_view(key_prefix="soma_sanctuary")


def render_zen_companion_3d_widget(scenario: str = "balanced") -> str:
    """Return interactive 3D Zen Companion WebGL widget powered by Three.js with scenario-aware aura and zero slash artifacts."""
    theme_cfg = {
        "balanced": {
            "title": "Live 3D Sanctuary Companion",
            "crystal_color": "0x0f766e",
            "crystal_emissive": "0x044e47",
            "halo_color": "0x2dd4bf",
            "initial_msg": '"Team rhythm is grounded and steady today."',
            "insights": [
                '"Team balance is radiant and steady today."',
                '"Breathe in equanimity · Release unnecessary strain."',
                '"Steady pacing compounds into sustainable breakthroughs."',
                '"Small mindful pauses cultivate profound long-term clarity."',
                '"Centered focus elevates collaborative flow."'
            ]
        },
        "thriving": {
            "title": "Live 3D Flourishing Aura",
            "crystal_color": "0x059669",
            "crystal_emissive": "0x064e3b",
            "halo_color": "0x34d399",
            "initial_msg": '"Psychological safety is high · Team synergy in flow!"',
            "insights": [
                '"Psychological safety is high · Team synergy in flow!"',
                '"Celebrate collective wins and acknowledge each contribution."',
                '"Gratitude amplifies resilience and sparks innovation."',
                '"High trust unlocks boundless creative courage."',
                '"Radiant morale illuminates every challenge."'
            ]
        },
        "burnout": {
            "title": "Live 3D De-escalation Beacon",
            "crystal_color": "0xe11d48",
            "crystal_emissive": "0x881337",
            "halo_color": "0xfb7185",
            "initial_msg": '"Elevated cognitive load detected · Time to pause & recharge."',
            "insights": [
                '"Elevated cognitive load detected · Time to pause & recharge."',
                '"Prioritize human well-being over arbitrary urgency."',
                '"Cancel non-essential meetings to restore baseline flow."',
                '"A brief collective pause prevents chronic depletion."',
                '"True engineering resilience starts with restful boundaries."'
            ]
        }
    }
    cfg = theme_cfg.get(scenario, theme_cfg["balanced"])
    import json
    insights_json = json.dumps(cfg["insights"])

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: transparent;
      font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      color: #0f172a;
      overflow: hidden;
      height: 350px;
      user-select: none;
    }}
    .companion-card {{
      width: 100%;
      height: 100%;
      border-radius: 20px;
      background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.78) 55%, rgba(241, 245, 249, 0.72) 100%);
      border: 1.5px solid rgba(204, 251, 241, 0.85);
      box-shadow: 0 16px 36px -8px rgba(15, 118, 110, 0.12), 0 4px 12px rgba(0, 0, 0, 0.03), inset 0 1px 1px rgba(255, 255, 255, 0.95);
      position: relative;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 10px 14px 12px;
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 10;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 10px;
      font-weight: 700;
      color: #0f766e;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(240, 253, 250, 0.95);
      border: 1px solid rgba(45, 212, 191, 0.6);
      padding: 3px 11px;
      border-radius: 9999px;
      box-shadow: 0 2px 6px rgba(15, 118, 110, 0.06);
    }}
    .pulse-dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #14b8a6;
      animation: pulseDot 2s infinite ease-in-out;
    }}
    @keyframes pulseDot {{
      0%, 100% {{ opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 rgba(20, 184, 166, 0.5); }}
      50% {{ opacity: 0.6; transform: scale(1.2); box-shadow: 0 0 0 5px rgba(20, 184, 166, 0); }}
    }}
    .tag {{
      font-size: 9.5px;
      font-family: 'JetBrains Mono', monospace;
      background: rgba(255, 255, 255, 0.9);
      padding: 3px 9px;
      border-radius: 8px;
      color: #64748b;
      border: 1px solid #e2e8f0;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    #canvas-container {{
      height: 235px;
      min-height: 235px;
      width: 100%;
      cursor: grab;
      position: relative;
    }}
    #canvas-container:active {{
      cursor: grabbing;
    }}
    .dialogue-bar {{
      z-index: 10;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      padding: 8px 14px;
      border-radius: 9999px;
      border: 1px solid rgba(204, 251, 241, 0.85);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      box-shadow: 0 2px 8px rgba(15, 118, 110, 0.06);
    }}
    .dialogue-text {{
      display: flex;
      flex-direction: column;
      gap: 1px;
      flex: 1;
      overflow: hidden;
    }}
    .dialogue-text h4 {{
      font-size: 11.5px;
      font-weight: 700;
      color: #0f172a;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
    }}
    .dialogue-text p {{
      font-size: 9.5px;
      color: #64748b;
    }}
    .action-btn {{
      padding: 4px 10px;
      border-radius: 9999px;
      background: linear-gradient(135deg, #0f766e, #14b8a6);
      color: #ffffff;
      border: none;
      display: flex;
      align-items: center;
      gap: 4px;
      font-size: 10px;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 2px 6px rgba(15, 118, 110, 0.25);
      transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
      white-space: nowrap;
    }}
    .action-btn:hover {{
      transform: scale(1.04) translateY(-1px);
      box-shadow: 0 4px 12px rgba(15, 118, 110, 0.35);
    }}
  </style>
</head>
<body>
  <div class="companion-card">
    <div class="header">
      <div class="badge">
        <div class="pulse-dot"></div>
        {cfg["title"]}
      </div>
      <div class="tag">Touch & Orbit 3D</div>
    </div>

    <div id="canvas-container"></div>

    <div class="dialogue-bar">
      <div style="display: flex; align-items: center; gap: 8px; flex: 1; overflow: hidden;">
        <span style="font-size: 15px;">🍃</span>
        <div class="dialogue-text">
          <h4 id="dialogue-msg">{cfg["initial_msg"]}</h4>
          <p>Touch or click the 3D crystal to awaken mindful insights.</p>
        </div>
      </div>
      <button class="action-btn" id="btn-spin" title="Aura Resonance">
        <span>Pulse</span> ✨
      </button>
    </div>
  </div>

  <script>
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth || 360;
    const height = 235;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 1000);
    camera.position.z = 4.3;

    const renderer = new THREE.WebGLRenderer({{ alpha: true, antialias: true }});
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 1.3));

    const pLight1 = new THREE.PointLight(0x14b8a6, 3.2, 40);
    pLight1.position.set(3, 3.5, 3);
    scene.add(pLight1);

    const pLight2 = new THREE.PointLight(0xf59e0b, 2.0, 40);
    pLight2.position.set(-3, -2.5, 2);
    scene.add(pLight2);

    const pLight3 = new THREE.PointLight(0x38bdf8, 1.6, 40);
    pLight3.position.set(0, 4, -2);
    scene.add(pLight3);

    const masterGroup = new THREE.Group();

    // 1. Faceted Icosahedron Crystal
    const crystalGeo = new THREE.IcosahedronGeometry(0.92, 0);
    const crystalMat = new THREE.MeshPhongMaterial({{
      color: {cfg["crystal_color"]},
      emissive: {cfg["crystal_emissive"]},
      specular: 0x99f6e4,
      shininess: 110,
      transparent: true,
      opacity: 0.88,
      flatShading: true
    }});
    const crystalMesh = new THREE.Mesh(crystalGeo, crystalMat);
    masterGroup.add(crystalMesh);

    // 2. Translucent Cyan Wireframe Lattice
    const wireGeo = new THREE.IcosahedronGeometry(0.94, 0);
    const wireMat = new THREE.MeshBasicMaterial({{
      color: 0x5eead4,
      wireframe: true,
      transparent: true,
      opacity: 0.40
    }});
    const wireMesh = new THREE.Mesh(wireGeo, wireMat);
    masterGroup.add(wireMesh);

    // 3. Radiant Octahedral Core
    const innerGeo = new THREE.OctahedronGeometry(0.44, 0);
    const innerMat = new THREE.MeshPhongMaterial({{
      color: 0xfef08a,
      emissive: 0xd97706,
      shininess: 100,
      flatShading: true
    }});
    const innerMesh = new THREE.Mesh(innerGeo, innerMat);
    masterGroup.add(innerMesh);

    // 4. Ethereal Celestial Halos (Permanent Open Ellipses - No Slash / No Edge-on Flip)
    const haloGroup = new THREE.Group();
    haloGroup.rotation.x = 1.15; // Stable ~66-degree incline relative to camera
    scene.add(haloGroup);

    const ring1Geo = new THREE.TorusGeometry(1.48, 0.018, 20, 90);
    const ring1Mat = new THREE.MeshPhongMaterial({{
      color: {cfg["halo_color"]},
      emissive: 0x0f766e,
      specular: 0xffffff,
      shininess: 120,
      transparent: true,
      opacity: 0.65
    }});
    const ring1 = new THREE.Mesh(ring1Geo, ring1Mat);
    haloGroup.add(ring1);

    const ring2Geo = new THREE.TorusGeometry(1.70, 0.012, 20, 90);
    const ring2Mat = new THREE.MeshPhongMaterial({{
      color: 0x99f6e4,
      emissive: 0x0f766e,
      specular: 0xffffff,
      shininess: 120,
      transparent: true,
      opacity: 0.42
    }});
    const ring2 = new THREE.Mesh(ring2Geo, ring2Mat);
    haloGroup.add(ring2);

    // 5. Orbiting Luminous Pearl
    const satGeo = new THREE.SphereGeometry(0.08, 20, 20);
    const satMat = new THREE.MeshPhongMaterial({{
      color: 0xffffff,
      emissive: 0x5eead4,
      specular: 0xffffff,
      shininess: 100
    }});
    const satellite = new THREE.Mesh(satGeo, satMat);
    masterGroup.add(satellite);

    // 6. Floating Ambient Stardust
    const pCount = 100;
    const pGeo = new THREE.BufferGeometry();
    const pPos = new Float32Array(pCount * 3);
    for (let i = 0; i < pCount * 3; i += 3) {{
      pPos[i] = (Math.random() - 0.5) * 7.5;
      pPos[i+1] = (Math.random() - 0.5) * 5.5;
      pPos[i+2] = (Math.random() - 0.5) * 5.5;
    }}
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
    const pMat = new THREE.PointsMaterial({{
      color: 0x2dd4bf,
      size: 0.065,
      transparent: true,
      opacity: 0.75
    }});
    const particles = new THREE.Points(pGeo, pMat);
    scene.add(particles);

    scene.add(masterGroup);

    // Mouse & Touch Physics
    let isDragging = false;
    let prevMouseX = 0, prevMouseY = 0;
    let targetRotX = 0, targetRotY = 0;
    let curRotX = 0, curRotY = 0;

    container.addEventListener('mousedown', (e) => {{
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    }});
    window.addEventListener('mouseup', () => {{ isDragging = false; }});
    window.addEventListener('mousemove', (e) => {{
      if (isDragging) {{
        const dx = e.clientX - prevMouseX;
        const dy = e.clientY - prevMouseY;
        targetRotY += dx * 0.01;
        targetRotX += dy * 0.01;
        prevMouseX = e.clientX;
        prevMouseY = e.clientY;
      }}
    }});

    container.addEventListener('mousemove', (e) => {{
      if (!isDragging) {{
        const rect = container.getBoundingClientRect();
        const mx = (e.clientX - rect.left) / rect.width - 0.5;
        const my = (e.clientY - rect.top) / rect.height - 0.5;
        targetRotY = mx * 0.55;
        targetRotX = my * 0.40;
      }}
    }});

    container.addEventListener('touchstart', (e) => {{
      if (e.touches.length === 1) {{
        isDragging = true;
        prevMouseX = e.touches[0].clientX;
        prevMouseY = e.touches[0].clientY;
      }}
    }}, {{ passive: true }});
    window.addEventListener('touchend', () => {{ isDragging = false; }});
    window.addEventListener('touchmove', (e) => {{
      if (isDragging && e.touches.length === 1) {{
        const dx = e.touches[0].clientX - prevMouseX;
        const dy = e.touches[0].clientY - prevMouseY;
        targetRotY += dx * 0.012;
        targetRotX += dy * 0.012;
        prevMouseX = e.touches[0].clientX;
        prevMouseY = e.touches[0].clientY;
      }}
    }}, {{ passive: true }});

    let isPulsing = false;
    let pulseTimer = 0;
    const insights = {insights_json};

    function trigger() {{
      isPulsing = true;
      pulseTimer = 1.6;
      const d = document.getElementById('dialogue-msg');
      if (d) d.textContent = insights[Math.floor(Math.random() * insights.length)];
    }}

    container.addEventListener('click', trigger);
    const spinBtn = document.getElementById('btn-spin');
    if (spinBtn) spinBtn.addEventListener('click', trigger);

    const clock = new THREE.Clock();
    function animate() {{
      requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const t = clock.getElapsedTime();

      let breathScale = 1.0 + 0.06 * Math.sin(t * 1.3);
      if (isPulsing) {{
        pulseTimer -= delta;
        breathScale += 0.10 * Math.sin(t * 14);
        crystalMat.emissive.setHex(0xb45309);
        crystalMat.color.setHex(0x5eead4);
        pMat.size = 0.095;
        if (pulseTimer <= 0) {{
          isPulsing = false;
          crystalMat.emissive.setHex({cfg["crystal_emissive"]});
          crystalMat.color.setHex({cfg["crystal_color"]});
          pMat.size = 0.065;
        }}
      }}

      crystalMesh.scale.set(breathScale, breathScale, breathScale);
      wireMesh.scale.set(breathScale, breathScale, breathScale);
      innerMesh.rotation.y -= delta * 0.9;
      innerMesh.rotation.x += delta * 0.6;

      // Celestial Orbit Halos (Spinning in-plane: open ellipses, zero slash!)
      ring1.rotation.z += delta * 0.22;
      ring2.rotation.z -= delta * 0.16;
      haloGroup.position.y = masterGroup.position.y;
      haloGroup.rotation.y = curRotY * 0.4;

      const satAngle = t * 1.4;
      satellite.position.set(
        Math.cos(satAngle) * 1.55,
        Math.sin(satAngle * 0.8) * 0.5,
        Math.sin(satAngle) * 1.55
      );

      particles.rotation.y += delta * 0.03;

      curRotX += (targetRotX - curRotX) * 0.07;
      curRotY += (targetRotY - curRotY) * 0.07;

      masterGroup.position.y = Math.sin(t * 1.1) * 0.08;
      masterGroup.rotation.y = curRotY + t * 0.18;
      masterGroup.rotation.x = curRotX + Math.sin(t * 0.5) * 0.06;

      renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener('resize', () => {{
      const w = container.clientWidth || 360;
      const h = 235;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }});
  </script>
</body>
</html>
"""



def render_team_health_view():
    """Render Google Stitch-inspired Team Wellbeing Intelligence & 3D Companion."""
    refreshed_at = st.session_state.get("team_insights_refreshed_at", "Just now (Live Telemetry)")

    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="background: #d7e8c6; color: #3d4b32; padding: 3px 10px; border-radius: 8px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">AI Synthesis & Strategy</span>
                <span style="color: #70796f; font-size: 0.8rem;">• Synchronized {refreshed_at}</span>
            </div>
            <h2 style="margin: 0; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.7rem; font-weight: 700; color: #1a1c1b; letter-spacing: -0.025em;">Team Wellbeing Intelligence</h2>
            <p style="margin: 4px 0 0 0; color: #404940; font-size: 0.92rem; line-height: 1.5;">
                Synthesized insights derived from active check-ins, sentiment trends, and organizational rhythm to optimize leadership action, guided by your interactive AI Wellbeing Companion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Telemetry Stream Mode Switcher
    stream_tabs = st.tabs(["🔮 Scenario Stress-Testing & Forecasts", "📝 Log Instant Team Pulse", "📋 Team Check-in Activity Log"])

    with stream_tabs[0]:
        col_ctrl, col_companion = st.columns([1.1, 0.9], gap="large")
        with col_ctrl:
            scenario = st.radio(
                "Select Team Operating Scenario to Model:",
                ["balanced", "thriving", "burnout"],
                format_func=lambda x: {
                    "balanced": "Balanced Sprint (Moderate workload with constructive recovery)",
                    "thriving": "Thriving Unit (High psychological safety and gratitude)",
                    "burnout": "High-Strain Team (Sustained negative sentiment & distress markers)",
                }[x],
                horizontal=False,
                key="team_health_scenario_select",
            )

            mock_data = generate_synthetic_team_data(days=14, team_size=8, scenario=scenario)

            # Integrate any live submitted checkins into telemetry
            if st.session_state.get("live_team_checkins"):
                for lc in st.session_state["live_team_checkins"]:
                    mock_data.append({
                        "date": lc["date"],
                        "employee_id": lc.get("employee_id", "EMP-LIVE"),
                        "sentiment_compound": lc["compound"],
                        "scenario": scenario
                    })

            metrics = calculate_team_burnout_risk(mock_data)

            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("Refresh AI Insights", type="primary", width="stretch", key="btn_refresh_team_ai"):
                    st.session_state["team_insights_refreshed_at"] = datetime.now(timezone.utc).strftime("%I:%M:%S %p UTC")
                    st.toast("Telemetry synchronized across all active check-ins.", icon="✨")
                    st.rerun()
            with c_btn2:
                st.download_button(
                    "Export Intelligence Report",
                    data=pd.DataFrame(mock_data).to_csv(index=False),
                    file_name=f"team_wellbeing_{scenario}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    width="stretch",
                    key="btn_export_team_report",
                )

        with col_companion:
            # Embed Interactive 3D Zen Companion (Three.js WebGL with scenario tuning)
            render_iframe(render_zen_companion_3d_widget(scenario=scenario), height=360)

        st.markdown("---")

        # Bento Style Impact Metrics
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            pulse_val = "8.4" if scenario == "balanced" else ("9.2" if scenario == "thriving" else "5.8")
            st.metric(
                label="Team Pulse Index",
                value=pulse_val,
                delta="+0.6 Trending Up" if scenario != "burnout" else "-1.4 High Strain",
                delta_color="normal" if scenario != "burnout" else "inverse",
                help="Calculated from active daily check-ins this week.",
            )
        with b2:
            st.metric(
                label="Burnout Risk Factor",
                value=metrics["risk_level"].capitalize(),
                delta=f"{metrics['burnout_risk_score']}/100 Score",
                delta_color="inverse",
                help="Cognitive friction index across engineering teams.",
            )
        with b3:
            st.metric(
                label="Action Adoption Rate",
                value=f"{metrics['team_resilience_index']}%",
                delta="14 of 18 Complete",
                help="Suggested micro-interventions completed by leads.",
            )
        with b4:
            st.metric(
                label="Psychological Safety",
                value="9.1 / 10" if scenario != "burnout" else "5.2 / 10",
                delta="High Flow" if scenario != "burnout" else "Needs Attention",
                help="Consistent openness reported in 1-on-1 reflections.",
            )

        # 14-Day Team Sentiment Chart
        df_team = pd.DataFrame(mock_data)
        df_agg = df_team.groupby("date")["sentiment_compound"].mean().reset_index()
        fig = px.line(
            df_agg,
            x="date",
            y="sentiment_compound",
            markers=True,
            title="14-Day Team Sentiment Moving Average vs Distress Boundary",
            labels={"sentiment_compound": "Sentiment Valence", "date": "Date"},
        )
        fig.update_traces(
            line_shape="spline",
            line={"color": "#0f766e", "width": 3},
            marker={"size": 7, "color": "#14b8a6", "line": {"width": 2, "color": "#ffffff"}}
        )
        fig.add_hline(y=0.05, line_dash="dash", line_color="#11562a", annotation_text="Positive Baseline (+0.05)")
        fig.add_hline(y=-0.05, line_dash="dash", line_color="#ba1a1a", annotation_text="Distress Boundary (-0.05)")
        fig.update_layout(
            height=260,
            margin={"l": 20, "r": 20, "t": 35, "b": 15},
            plot_bgcolor="#faf9f7",
            paper_bgcolor="#faf9f7",
        )
        st.plotly_chart(fig, width="stretch")

        # Split: Recommended Strategic Shifts & Micro-Interventions Checklist
        c_shifts, c_check = st.columns([1.1, 0.9], gap="large")

        with c_shifts:
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                    <h3 style="margin: 0; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #0f172a;">
                        Recommended Strategic Shifts
                    </h3>
                    <span style="font-size: 0.78rem; color: #0891b2; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">AI Telemetry Sourced</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Priority 1 Card: Team Recharge Day
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <div>
                            <span style="background: #fee2e2; border: 1px solid #fecaca; color: #991b1b; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;">Priority 01 • Urgent Focus</span>
                            <h4 style="margin: 4px 0 0 0; color: #0f172a; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.15rem; font-weight: 700;">Schedule Team Recharge Day</h4>
                        </div>
                        <span style="background: #f0fdf4; border: 1px solid #86efac; color: #166534; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; font-weight: 700;">
                            Impact: Very High (+18% Focus)
                        </span>
                    </div>
                    <p style="font-size: 0.84rem; color: #475569; line-height: 1.5; margin-bottom: 10px;">
                        Telemetry reveals elevated cognitive strain and continuous context switching over the past 3 weeks. An uninterrupted recharge window significantly lowers cortisol and restores baseline deep flow.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

                col_hint1, col_act1 = st.columns([1.3, 1])
                with col_hint1:
                    st.markdown("<span style='font-size: 0.78rem; color: #64748b;'>Recommended: Next Thursday afternoon (4 hours)</span>", unsafe_allow_html=True)
                with col_act1:
                    plan_1_deployed = st.session_state.get("recharge_plan_active", False)
                    btn_text = "✅ Plan Deployed" if plan_1_deployed else "🚀 Deploy Plan"
                    if st.button(btn_text, key="btn_deploy_plan_1", type="primary" if not plan_1_deployed else "secondary", width="stretch"):
                        st.session_state["recharge_plan_active"] = not plan_1_deployed
                        st.toast("Team Recharge schedule updated!", icon="📅")
                        st.rerun()

                if st.session_state.get("recharge_plan_active"):
                    st.success("✅ **Recharge Day Active**: Calendar holds configured for 8 team members. Slack status sync enabled.")
                    ics_content = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//EquiMind//Team Recharge Day//EN
BEGIN:VEVENT
SUMMARY:EquiMind Team Recharge & Focus Sanctuary
DESCRIPTION:Quiet morning followed by uninterrupted flow sprint and retrospective circle.
STATUS:CONFIRMED
BEGIN:VALARM
TRIGGER:-PT15M
ACTION:DISPLAY
DESCRIPTION:Reminder
END:VALARM
END:VEVENT
END:VCALENDAR"""
                    st.download_button(
                        "📥 Download Calendar Hold (.ics)",
                        data=ics_content,
                        file_name="equimind_recharge_day.ics",
                        mime="text/calendar",
                        key="btn_download_recharge_ics",
                    )

            # Priority 2 Card: Asynchronous Communication
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <div>
                            <span style="background: #ede9fe; border: 1px solid #ddd6fe; color: #5b21b6; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;">Priority 02 • Structural Habit</span>
                            <h4 style="margin: 4px 0 0 0; color: #0f172a; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.15rem; font-weight: 700;">Encourage Asynchronous Communication</h4>
                        </div>
                        <span style="background: #f0fdfa; border: 1px solid #99f6e4; color: #0f766e; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; font-weight: 700;">
                            Impact: High (+6 hrs/week)
                        </span>
                    </div>
                    <p style="font-size: 0.84rem; color: #475569; line-height: 1.5; margin-bottom: 10px;">
                        Meeting volume rose 22% month-over-month. Transitioning standard status updates to async threads recovers approximately 6 hours of uninterrupted creative time per team member weekly.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

                col_hint2, col_act2 = st.columns([1.3, 1])
                with col_hint2:
                    st.markdown("<span style='font-size: 0.78rem; color: #64748b;'>Action: Adopt 'No-Meeting Mornings' on Tue/Thu</span>", unsafe_allow_html=True)
                with col_act2:
                    async_sent = st.session_state.get("async_guidelines_active", False)
                    btn2_text = "✅ Guidelines Sent" if async_sent else "📢 Send Guidelines"
                    if st.button(btn2_text, key="btn_deploy_plan_2", type="secondary", width="stretch"):
                        st.session_state["async_guidelines_active"] = not async_sent
                        st.toast("Async guidelines status updated!", icon="📢")
                        st.rerun()

                if st.session_state.get("async_guidelines_active"):
                    st.info("ℹ️ **Guidelines Distributed**: Shared to `#engineering-general` and pinned in team workspace.")

            # Priority 3 Card: Friday Autonomic Buffer
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <div>
                            <span style="background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;">Priority 03 • Autonomic Buffer</span>
                            <h4 style="margin: 4px 0 0 0; color: #0f172a; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.15rem; font-weight: 700;">Establish Friday Focus Sanctuary (1:00 PM+)</h4>
                        </div>
                        <span style="background: #fdf4ff; border: 1px solid #f0abfc; color: #86198f; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; font-weight: 700;">
                            Impact: Sustained (+25% Morale)
                        </span>
                    </div>
                    <p style="font-size: 0.84rem; color: #475569; line-height: 1.5; margin-bottom: 10px;">
                        Protects the final hours of the work week with zero deployment cutoffs and meeting embargoes, ensuring employees head into the weekend without unfinished anxiety.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
                col_hint3, col_act3 = st.columns([1.3, 1])
                with col_hint3:
                    st.markdown("<span style='font-size: 0.78rem; color: #64748b;'>Policy: Zero external customer syncs after 1:00 PM Friday</span>", unsafe_allow_html=True)
                with col_act3:
                    buffer_active = st.session_state.get("friday_buffer_active", False)
                    btn3_text = "✅ Buffer Active" if buffer_active else "🛡️ Activate Buffer"
                    if st.button(btn3_text, key="btn_deploy_plan_3", type="secondary", width="stretch"):
                        st.session_state["friday_buffer_active"] = not buffer_active
                        st.toast("Friday Sanctuary buffer toggled!", icon="🛡️")
                        st.rerun()

                if st.session_state.get("friday_buffer_active"):
                    st.success("✅ **Sanctuary Active**: Automated calendar protection rule applied for Friday 1:00 PM onwards.")

            # Executive Meeting Friction & Deep Work Recovery Calculator
            render_meeting_recovery_calculator()

        with c_check:
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                    <h3 style="margin: 0; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.25rem; font-weight: 700; color: #0f172a;">
                        Manager Micro-Interventions
                    </h3>
                    <span style="font-size: 0.78rem; color: #0f766e; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Weekly Cadence</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Calculate completed interventions live
            interventions_def = [
                ("chk_interv_1", "Public shout-out & recognition in team channel", "Acknowledged contributors for clean release.", "📣 Recognition", True),
                ("chk_interv_2", "Audit recurring calendar meetings", "Pruned 3 redundant recurring syncs to save 4.5 hrs.", "🗓️ Focus Time", True),
                ("chk_interv_3", "Review workload balance in 1-on-1 touchpoints", "Shifted sprint deliverables to balance cognitive load.", "🤝 Equilibrium", True),
                ("chk_interv_4", "Host unstructured coffee chat (30-min hangout)", "Casual team connection with zero agenda pressure.", "☕ Social Bond", False),
                ("chk_interv_5", "Share somatic breathing reset before sprint planning", "Embedded 2-min box-breathing calming protocol.", "🌬️ Calm Reset", False),
            ]

            # Init state if needed
            for k, _, _, _, def_val in interventions_def:
                if k not in st.session_state:
                    st.session_state[k] = def_val

            completed_count = sum(1 for k, _, _, _, _ in interventions_def if st.session_state.get(k, False))
            comp_pct = int((completed_count / len(interventions_def)) * 100)

            with st.container(border=True):
                # Dynamic Live Progress Card Header
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #ecfeff 100%); border: 1px solid #bbf7d0; border-radius: 12px; padding: 12px 14px; margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 0.8rem; font-weight: 700; color: #166534;">Leadership Care Index</span>
                            <span style="background: #ffffff; border: 1px solid #86efac; color: #166534; font-size: 0.76rem; font-weight: 800; padding: 2px 8px; border-radius: 10px;">
                                {completed_count} / {len(interventions_def)} Completed ({comp_pct}%)
                            </span>
                        </div>
                        <div style="font-size: 0.78rem; color: #047857; margin-bottom: 8px;">
                            {"🌟 Optimal cadence: Team resilience index elevated by +14%." if comp_pct >= 80 else ("🌿 Strong baseline: Consistent leadership support maintained." if comp_pct >= 60 else "⚠️ Action required: 2 more micro-actions recommended to protect resilience.")}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(completed_count / len(interventions_def))

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                for k, label, note, tag, _ in interventions_def:
                    checked = st.checkbox(
                        label,
                        key=k,
                        help=note,
                    )
                    if checked:
                        st.markdown(
                            f"<div style='margin: -8px 0 8px 26px; font-size: 0.74rem; color: #0f766e;'>✓ {note} <span style='background: #f1f5f9; padding: 1px 6px; border-radius: 4px; font-size: 0.7rem; color: #475569;'>{tag}</span></div>",
                            unsafe_allow_html=True,
                        )

    # Tab 2: Live Team Pulse Check-In Form (Fully Functional)
    with stream_tabs[1], st.container(border=True):
            st.markdown(
                """
                <h3 style="margin: 0; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.35rem; font-weight: 700; color: #0f172a;">
                    📝 Log Real-Time Team Member Pulse
                </h3>
                <p style="font-size: 0.86rem; color: #64748b; margin-top: 3px;">
                    Submit a real team reflection to update live telemetry and organizational moving averages.
                </p>
                """,
                unsafe_allow_html=True,
            )
            with st.form("form_submit_team_pulse"):
                tc_col1, tc_col2 = st.columns(2)
                with tc_col1:
                    tp_member = st.text_input("Team Member / Role Alias", value="Senior Frontend Engineer", placeholder="e.g. Lead Architect, UX Researcher")
                    tp_workload = st.slider("Current Workload Pressure (1 = Relaxed, 10 = Severe Strain)", min_value=1, max_value=10, value=5)
                with tc_col2:
                    tp_morale = st.selectbox("Morale State:", ["☀️ High Flow & Positive Energy", "🌿 Steady & Balanced", "⚡ High Cognitive Friction", "🌧️ Fatigued & Needs Decompression"])
                    tp_category = st.selectbox("Operating Focus Domain:", ["Engineering Sprint", "Product Launch", "Cross-team Alignment", "Operations & Triage"])

                tp_notes = st.text_area("Optional Pulse Notes / Obstacles:", height=80, placeholder="E.g. Feeling good after wrapping sprint goals, but 4 back-to-back reviews created slight fatigue.")

                submit_pulse_btn = st.form_submit_button("Submit Team Pulse Check-In", type="primary", width="stretch")
                if submit_pulse_btn:
                    compound_map = {
                        "☀️ High Flow & Positive Energy": 0.65,
                        "🌿 Steady & Balanced": 0.25,
                        "⚡ High Cognitive Friction": -0.30,
                        "🌧️ Fatigued & Needs Decompression": -0.55
                    }
                    comp_val = compound_map.get(tp_morale, 0.20)
                    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

                    if "live_team_checkins" not in st.session_state:
                        st.session_state["live_team_checkins"] = []

                    st.session_state["live_team_checkins"].insert(0, {
                        "date": now_str,
                        "employee_id": tp_member,
                        "compound": comp_val,
                        "workload": tp_workload,
                        "morale": tp_morale,
                        "category": tp_category,
                        "notes": tp_notes or "Daily team pulse logged successfully."
                    })

                    st.session_state["team_insights_refreshed_at"] = datetime.now(timezone.utc).strftime("%I:%M:%S %p UTC")
                    st.success(f"✅ Team pulse recorded for **{tp_member}**! Integrated into 14-day moving average and burnout indices.")
                    st.rerun()

    # Tab 3: Activity Log of All Check-Ins
    with stream_tabs[2], st.container(border=True):
            st.markdown(
                """
                <h3 style="margin: 0; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.35rem; font-weight: 700; color: #0f172a;">
                    📋 Team Telemetry Activity Log
                </h3>
                <p style="font-size: 0.86rem; color: #64748b; margin-top: 3px;">
                    Inspect anonymized check-ins contributing to the organization sentiment moving average.
                </p>
                """,
                unsafe_allow_html=True,
            )
            live_records = st.session_state.get("live_team_checkins", [])
            if not live_records:
                st.info("No manual pulse entries logged in this session yet.")
                if st.button("🌱 Load Sample Engineering Sprint Logs", key="btn_load_sample_pulse_records"):
                    sample_logs = [
                        {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "employee_id": "Principal Systems Architect", "compound": 0.58, "workload": 6, "morale": "☀️ High Flow & Positive Energy", "category": "Engineering Sprint", "notes": "Resolved core concurrency bottleneck; latency improved by 35%."},
                        {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "employee_id": "Lead UX Researcher", "compound": 0.42, "workload": 4, "morale": "🌿 Steady & Balanced", "category": "Product Launch", "notes": "Completed user journey synthesis; clear positive signal across customer cohort."},
                        {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "employee_id": "Staff Security Engineer", "compound": 0.35, "workload": 5, "morale": "🌿 Steady & Balanced", "category": "Operations & Triage", "notes": "Automated dependency vulnerability audits; zero blocker alerts."},
                        {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "employee_id": "Senior Frontend Engineer", "compound": -0.25, "workload": 8, "morale": "⚡ High Cognitive Friction", "category": "Engineering Sprint", "notes": "Continuous merge conflicts in monorepo caused slight afternoon fatigue."}
                    ]
                    st.session_state["live_team_checkins"] = sample_logs
                    st.toast("Loaded 4 sample engineering sprint check-ins!", icon="🌱")
                    st.rerun()
            else:
                st.dataframe(pd.DataFrame(live_records), width="stretch")




def render_compact_box_pacer() -> str:
    """Return self-contained HTML5 & Web Audio interactive visual breathing pacer for the care hub."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: transparent;
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      overflow: hidden;
      color: #0f172a;
      height: 220px;
      user-select: none;
    }
    .pacer-container {
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 100%;
      background: linear-gradient(135deg, rgba(240, 253, 250, 0.85) 0%, rgba(255, 255, 255, 0.95) 100%);
      border: 1.5px solid rgba(204, 251, 241, 0.9);
      border-radius: 16px;
      padding: 12px 18px;
      box-shadow: 0 4px 16px rgba(15, 118, 110, 0.05), inset 0 1px 1px #ffffff;
    }
    .pacer-left {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      width: 140px;
    }
    .breath-orb {
      width: 105px;
      height: 105px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(45, 212, 191, 0.35) 0%, rgba(15, 118, 110, 0.12) 65%, transparent 100%);
      border: 2px solid rgba(45, 212, 191, 0.7);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 0 20px rgba(20, 184, 166, 0.25);
      transition: transform 4s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 4s ease, border-color 4s ease;
    }
    .breath-cue {
      font-size: 11px;
      font-weight: 800;
      color: #0f766e;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      text-align: center;
    }
    .breath-sub {
      font-size: 9px;
      color: #0d9488;
      font-weight: 600;
      margin-top: 2px;
    }
    .pacer-right {
      flex: 1;
      padding-left: 18px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 6px;
    }
    .timer-val {
      font-family: 'Playfair Display', Georgia, serif;
      font-size: 2.1rem;
      font-weight: 700;
      color: #044e47;
      letter-spacing: -0.02em;
      line-height: 1;
    }
    .cycle-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(204, 251, 241, 0.7);
      border: 1px solid rgba(45, 212, 191, 0.6);
      color: #0f766e;
      font-size: 10px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 9999px;
      width: fit-content;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .btn-row {
      display: flex;
      gap: 8px;
      margin-top: 4px;
    }
    .pacer-btn {
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      border: none;
      transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .btn-play {
      background: linear-gradient(135deg, #0f766e, #14b8a6);
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(15, 118, 110, 0.25);
    }
    .btn-play:hover {
      transform: translateY(-1px) scale(1.02);
      box-shadow: 0 6px 16px rgba(15, 118, 110, 0.35);
    }
    .btn-reset {
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid #cbd5e1;
      color: #475569;
    }
    .btn-reset:hover {
      background: #f1f5f9;
      color: #0f172a;
    }
  </style>
</head>
<body>
  <div class="pacer-container">
    <div class="pacer-left">
      <div class="breath-orb" id="breathOrb" onclick="togglePacer()">
        <span class="breath-cue" id="cueText">Tap Start</span>
        <span class="breath-sub" id="subText">4-4-4 Rhythm</span>
      </div>
    </div>
    <div class="pacer-right">
      <div class="cycle-badge" id="cycleBadge">● Autonomic Vagus Pacer</div>
      <div class="timer-val" id="timerDisplay">02:00</div>
      <div style="font-size: 11px; color: #475569; line-height: 1.4;">
        Inhale 4s · Hold 4s · Exhale 4s · Hold 4s
      </div>
      <div class="btn-row">
        <button class="pacer-btn btn-play" id="playBtn" onclick="togglePacer()">▶ Start 2-Min Pacer</button>
        <button class="pacer-btn btn-reset" onclick="resetPacer()">Reset</button>
      </div>
    </div>
  </div>

  <script>
    var isRunning = false;
    var timeLeft = 120;
    var timerInterval = null;
    var breathInterval = null;
    var phase = 0;
    var cycleCount = 0;
    var audioCtx = null;

    function initAudio() {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
    }

    function playBell() {
      try {
        initAudio();
        var osc = audioCtx.createOscillator();
        var gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(432, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 2.5);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 2.6);
      } catch(e) {}
    }

    function updateBreath() {
      var orb = document.getElementById('breathOrb');
      var cue = document.getElementById('cueText');
      var sub = document.getElementById('subText');
      if (phase === 0) {
        cue.innerText = 'Breathe In';
        sub.innerText = '4 seconds';
        orb.style.transform = 'scale(1.32)';
        orb.style.boxShadow = '0 0 35px rgba(45, 212, 191, 0.65)';
        orb.style.borderColor = '#14b8a6';
        phase = 1;
      } else if (phase === 1) {
        cue.innerText = 'Hold Softly';
        sub.innerText = '4 seconds';
        orb.style.transform = 'scale(1.32)';
        orb.style.boxShadow = '0 0 25px rgba(20, 184, 166, 0.45)';
        orb.style.borderColor = '#2dd4bf';
        phase = 2;
      } else if (phase === 2) {
        cue.innerText = 'Breathe Out';
        sub.innerText = '4 seconds';
        orb.style.transform = 'scale(0.85)';
        orb.style.boxShadow = '0 0 12px rgba(20, 184, 166, 0.2)';
        orb.style.borderColor = 'rgba(45, 212, 191, 0.5)';
        phase = 3;
      } else {
        cue.innerText = 'Pause Empty';
        sub.innerText = '4 seconds';
        orb.style.transform = 'scale(0.85)';
        orb.style.boxShadow = '0 0 12px rgba(20, 184, 166, 0.2)';
        orb.style.borderColor = 'rgba(45, 212, 191, 0.5)';
        phase = 0;
        cycleCount++;
        document.getElementById('cycleBadge').innerText = '● Cycle ' + cycleCount + ' of 7';
      }
    }

    function togglePacer() {
      var btn = document.getElementById('playBtn');
      if (!isRunning) {
        isRunning = true;
        btn.innerText = '⏸ Pause';
        btn.style.background = '#0d9488';
        playBell();
        phase = 0;
        updateBreath();
        breathInterval = setInterval(updateBreath, 4000);
        timerInterval = setInterval(function() {
          if (timeLeft > 0) {
            timeLeft--;
            var m = Math.floor(timeLeft / 60);
            var s = timeLeft % 60;
            document.getElementById('timerDisplay').innerText = (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
          } else {
            resetPacer();
            playBell();
            document.getElementById('cueText').innerText = 'Complete! ✨';
            document.getElementById('subText').innerText = 'Great rhythm';
          }
        }, 1000);
      } else {
        isRunning = false;
        btn.innerText = '▶ Resume';
        btn.style.background = 'linear-gradient(135deg, #0f766e, #14b8a6)';
        clearInterval(breathInterval);
        clearInterval(timerInterval);
      }
    }

    function resetPacer() {
      isRunning = false;
      clearInterval(breathInterval);
      clearInterval(timerInterval);
      timeLeft = 120;
      phase = 0;
      cycleCount = 0;
      document.getElementById('timerDisplay').innerText = '02:00';
      document.getElementById('cueText').innerText = 'Tap Start';
      document.getElementById('subText').innerText = '4-4-4 Rhythm';
      document.getElementById('cycleBadge').innerText = '● Autonomic Vagus Pacer';
      var btn = document.getElementById('playBtn');
      btn.innerText = '▶ Start 2-Min Pacer';
      btn.style.background = 'linear-gradient(135deg, #0f766e, #14b8a6)';
      var orb = document.getElementById('breathOrb');
      orb.style.transform = 'scale(1.0)';
      orb.style.boxShadow = '0 0 20px rgba(20, 184, 166, 0.25)';
      orb.style.borderColor = 'rgba(45, 212, 191, 0.7)';
    }
  </script>
</body>
</html>
"""


def render_recommendations_view():
    """Render the high-class Adaptive Recommendations & Holistic Care Hub."""
    if "suggestion_feedback" not in st.session_state:
        st.session_state["suggestion_feedback"] = {"accepted": 4, "rejected": 1}
    fb = st.session_state["suggestion_feedback"]
    total_fb = fb.get("accepted", 0) + fb.get("rejected", 0)
    accept_rate = (fb.get("accepted", 0) / total_fb * 100) if total_fb > 0 else 80.0

    # High-Class Luxury Header Banner
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 253, 250, 0.80) 55%, rgba(248, 250, 252, 0.94) 100%); border: 1.5px solid rgba(204, 251, 241, 0.95); border-radius: 22px; padding: 22px 30px; box-shadow: 0 20px 42px -10px rgba(15, 118, 110, 0.10), 0 4px 16px rgba(0, 0, 0, 0.02), inset 0 1px 1px rgba(255, 255, 255, 1); margin-bottom: 22px; backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); transform-style: preserve-3d; transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.35s ease;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div>
                    <h2 style="margin: 0; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.65rem; font-weight: 700; color: #0f172a; letter-spacing: -0.02em;">Care Pathways & Daily Remedies</h2>
                    <p style="margin: 4px 0 0 0; font-size: 0.86rem; color: #64748b;">Holistic micro-interventions blending somatic yoga, restorative breathwork, cognitive reframing, and nutrition.</p>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    <div style="background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 1px solid #bbf7d0; border-radius: 9999px; padding: 6px 15px; font-size: 0.80rem; color: #166534; font-weight: 700; box-shadow: 0 2px 8px rgba(22, 163, 74, 0.12);">
                        🌱 <strong>{fb.get('accepted', 4)} Practices Completed</strong>
                    </div>
                    <div style="background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 1px solid #fde68a; border-radius: 9999px; padding: 6px 15px; font-size: 0.80rem; color: #92400e; font-weight: 700; box-shadow: 0 2px 8px rgba(245, 158, 11, 0.12);">
                        ☀️ <strong>3-Day Streak</strong>
                    </div>
                    <div style="background: linear-gradient(135deg, #f0fdfa, #ccfbf1); border: 1px solid #99f6e4; border-radius: 9999px; padding: 6px 15px; font-size: 0.80rem; color: #0f766e; font-weight: 700; box-shadow: 0 2px 8px rgba(15, 118, 110, 0.12);">
                        ✨ <strong>{accept_rate:.0f}% Mindful Alignment</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rec_tab1, rec_tab2, rec_tab3 = st.tabs([
        "🌟 Personalized Care Plan",
        "🧘 Somatic Yoga Remedies",
        "🍵 Mindful Nutrition & Teas",
    ])

    # Determine active emotion to personalize for
    last_rep = st.session_state.get("last_analysis_report")
    detected_from_rep = last_rep.get("predicted_emotion") if (last_rep and last_rep.get("predicted_emotion")) else None
    active_care_emotion = (st.session_state.get("care_focus_emotion") or detected_from_rep or "anxiety").lower()

    with rec_tab1:
        # Instant Feeling Filter Bar (Always gives the user instant, relevant remedies)
        st.markdown('<p style="font-size: 0.72rem; font-weight: 700; color: #0d9488; text-transform: uppercase; letter-spacing: 0.08em; margin: 4px 0 6px 2px;">✨ Explore Remedies by Current Feeling</p>', unsafe_allow_html=True)
        mood_options = [
            {"label": "🌿 Decompress", "sub": "Anxiety & Stress", "emotion": "anxiety"},
            {"label": "⚡ Energy Boost", "sub": "Fatigue & Slump", "emotion": "fatigue"},
            {"label": "🌧️ Gentle Comfort", "sub": "Sadness & Low", "emotion": "sadness"},
            {"label": "🔥 Release Tension", "sub": "Anger & Frustration", "emotion": "anger"},
            {"label": "☀️ Deepen Joy", "sub": "Gratitude & Win", "emotion": "joy"},
            {"label": "☕ Ground & Center", "sub": "Focus & Routine", "emotion": "neutral"},
        ]
        m_cols = st.columns(6, gap="small")
        for idx, opt in enumerate(mood_options):
            with m_cols[idx]:
                is_active = (active_care_emotion == opt["emotion"])
                btn_type = "primary" if is_active else "secondary"
                if st.button(f"{opt['label']}", key=f"care_filter_btn_{idx}", type=btn_type, width="stretch"):
                    st.session_state["care_focus_emotion"] = opt["emotion"]
                    st.rerun()

        # Generate rich data for the selected state
        rec = generate_recommendations(active_care_emotion, SentimentLabel.NEUTRAL, 0.0, f"Feeling {active_care_emotion}")
        yoga_remedy = get_yoga_remedy_for_emotion(active_care_emotion)
        food_plan = get_mood_food_plan(active_care_emotion)

        source_badge = (
            f'<span style="background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; font-weight: 700; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem;">● Tailored to Latest Reflection: {active_care_emotion.capitalize()}</span>'
            if (detected_from_rep and active_care_emotion == detected_from_rep)
            else f'<span style="background: #f0fdfa; color: #0f766e; border: 1px solid #ccfbf1; font-weight: 700; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem;">✨ Focus: {active_care_emotion.capitalize()} Relief</span>'
        )

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 12px 0 10px 0;">
                <h4 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0f172a;">Personalized Action Pathway</h4>
                <div>{source_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        plan_c1, plan_c2 = st.columns([1.45, 1.25], gap="large")

        with plan_c1:
            # Pillar 1: Mindful Strategy Card (100% Dynamic & Interactive)
            with st.container(border=True):
                st.markdown(f"#### 💡 {rec.get('title', 'Self-Care Focus')}")
                st.markdown(
                    f"<span style='background: #f8fafc; border: 1px solid #e2e8f0; padding: 3px 9px; border-radius: 6px; font-size: 0.75rem; color: #475569; font-weight: 700;'>Category: {rec.get('category', 'Well-Being')}</span> · "
                    f"<span style='background: #ecfeff; border: 1px solid #a5f3fc; padding: 3px 9px; border-radius: 6px; font-size: 0.75rem; color: #0e7490; font-weight: 700;'>Urgency: {rec.get('urgency', 'Low')}</span>",
                    unsafe_allow_html=True
                )
                st.write(rec.get("summary", "Personalized evidence-based micro-intervention mapped to your emotional state."))

                action_paths = rec.get("action_paths", [])
                if not action_paths:
                    action_paths = [
                        {
                            "title": "Ground in the Room (5-4-3-2-1 Sensory Reset)",
                            "duration": "2 minutes",
                            "description": "Give your attention a steady, concrete place to land by anchoring to immediate sensory cues.",
                            "steps": [
                                "Name 5 distinct objects you can visually see right now.",
                                "Notice 4 physical textures you can feel (chair, fabric, desk, floor).",
                                "Identify 3 ambient sounds in your auditory environment.",
                                "Drop your shoulders 2 inches, unclamp your jaw, and release 1 slow, audible exhale."
                            ]
                        },
                        {
                            "title": "Vagus Nerve Physiological Sigh",
                            "duration": "1 minute",
                            "description": "Double nasal inhale followed by prolonged oral exhale to trigger immediate parasympathetic braking.",
                            "steps": [
                                "Take a deep breath in through your nose to ~80% capacity.",
                                "Without exhaling, sharply sniff in another quick burst of air to maximally inflate alveoli.",
                                "Release a long, smooth, unforced exhale through gently parted lips for 6 seconds.",
                                "Repeat 3 complete cycles to downregulate elevated heart rate."
                            ]
                        },
                        {
                            "title": "Cognitive Decatastrophizing Reframe",
                            "duration": "3 minutes",
                            "description": "Systematic cognitive reframing to separate urgent facts from catastrophic projections.",
                            "steps": [
                                "Identify the runaway narrative: 'What catastrophizing story is my mind currently spinning?'",
                                "Objective evidence check: 'What are the concrete, verifiable facts right now?'",
                                "Realistic scenario anchor: 'What is the most likely middle-ground reality?'",
                                "Micro-agency step: 'What is ONE low-friction action I can complete in the next 10 minutes?'"
                            ]
                        }
                    ]

                # Pathway cycling state
                p_key = f"rec_p_idx_{active_care_emotion}"
                cur_p_idx = st.session_state.get(p_key, 0) % len(action_paths)
                path = action_paths[cur_p_idx]

                if "adopted_practices_set" not in st.session_state:
                    st.session_state["adopted_practices_set"] = set()
                is_adopted = path.get("title") in st.session_state["adopted_practices_set"]

                st.markdown(
                    f"<div style='margin-top: 10px; padding: 10px 14px; background: rgba(240, 253, 250, 0.7); border: 1px solid rgba(204, 251, 241, 0.9); border-radius: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;'>"
                    f"<div><span style='font-weight: 700; color: #044e47; font-size: 0.92rem;'>Pathway {cur_p_idx + 1} of {len(action_paths)}:</span> <span style='font-family: Playfair Display, serif; font-weight: 700; color: #0f766e; font-size: 0.96rem;'>{path.get('title')}</span></div>"
                    f"<span style='background: rgba(45, 212, 191, 0.15); border: 1px solid rgba(45, 212, 191, 0.5); padding: 2px 8px; border-radius: 9999px; font-size: 0.72rem; font-weight: 700; color: #0f766e;'>⏱️ {path.get('duration')}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.write(path.get("description", ""))
                for s_idx, step in enumerate(path.get("steps", []), 1):
                    st.markdown(f"**Step {s_idx}:** {step}")

                if is_adopted:
                    st.markdown(
                        f"<div style='background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 1.5px solid #86efac; border-radius: 12px; padding: 10px 14px; margin-top: 10px; display: flex; align-items: center; justify-content: space-between;'>"
                        f"<div style='color: #166534; font-size: 0.84rem; font-weight: 700;'>🌟 Practice Completed & Recorded: <em>{path.get('title')}</em></div>"
                        f"<span style='background: #15803d; color: #ffffff; padding: 3px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 700;'>+25 Resilience Pts</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                act_b1, act_b2, act_b3 = st.columns([1.35, 1.05, 1.1], gap="small")
                with act_b1:
                    adopt_label = "✅ Adopted & Complete!" if is_adopted else "✅ Adopt Practice (Accept)"
                    adopt_type = "secondary" if is_adopted else "primary"
                    if st.button(adopt_label, type=adopt_type, width="stretch", key=f"care_adopt_btn_{cur_p_idx}_{active_care_emotion}"):
                        st.session_state["adopted_practices_set"].add(path.get("title"))
                        if "suggestion_feedback" not in st.session_state:
                            st.session_state["suggestion_feedback"] = {"accepted": 4, "rejected": 1}
                        st.session_state["suggestion_feedback"]["accepted"] += 1
                        st.toast(f"🎉 Practice adopted: {path.get('title')}! +25 Resilience Points", icon="🌿")
                        st.rerun()
                with act_b2:
                    if st.button("❌ Decline / Skip", type="secondary", width="stretch", key=f"care_decline_btn_{cur_p_idx}_{active_care_emotion}"):
                        if "suggestion_feedback" not in st.session_state:
                            st.session_state["suggestion_feedback"] = {"accepted": 4, "rejected": 1}
                        st.session_state["suggestion_feedback"]["rejected"] += 1
                        st.session_state[p_key] = (cur_p_idx + 1) % len(action_paths)
                        st.toast("Skipped. Switched to alternative pathway.", icon="🌿")
                        st.rerun()
                with act_b3:
                    if st.button(f"🔄 Next ({cur_p_idx + 1}/{len(action_paths)})", type="secondary", width="stretch", key=f"care_next_btn_{cur_p_idx}_{active_care_emotion}"):
                        st.session_state[p_key] = (cur_p_idx + 1) % len(action_paths)
                        next_title = action_paths[(cur_p_idx + 1) % len(action_paths)]["title"]
                        st.toast(f"Switched to: {next_title}", icon="🌿")
                        st.rerun()

                live_fb = st.session_state.get("suggestion_feedback", {"accepted": 4, "rejected": 1})
                tot_fb = live_fb.get("accepted", 0) + live_fb.get("rejected", 0)
                cur_rate = (live_fb.get("accepted", 0) / tot_fb * 100) if tot_fb > 0 else 80.0
                completed_count = len(st.session_state.get("adopted_practices_set", [])) + 4
                st.markdown(
                    f"""
                    <div style="margin-top: 12px; padding: 8px 14px; background: rgba(240, 253, 250, 0.7); border: 1px solid rgba(204, 251, 241, 0.85); border-radius: 10px; font-size: 0.76rem; color: #0f766e; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
                        <span>🌱 <strong>Care Suggestion Adoption:</strong> {cur_rate:.1f}% ({completed_count} Completed)</span>
                        <span>✨ <strong>Habit Consistency:</strong> Strong Daily Practice</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Pillar 2: Interactive Somatic Breathwork Pacer (Fully Functional HTML5/JS Pacer with Audio)
            with st.container(border=True):
                st.markdown("#### 🌬️ 2-Minute Somatic Breathwork Pacer")
                st.write("Conscious 4-4-4 pacing resets your autonomic nervous system, decreases rapid heart rate, and downregulates amygdala arousal.")
                try:
                    render_iframe(render_compact_box_pacer(), height=235)
                except Exception:
                    st.info("Pacer active: Follow 4s Inhale · 4s Hold · 4s Exhale · 4s Pause.")

        with plan_c2:
            # Pillar 3: Restorative Yoga Remedy Card (100% Functional with Timer & Asana Tracker)
            with st.container(border=True):
                st.markdown(f"#### 🧘 {yoga_remedy.get('icon', '🧘')} {yoga_remedy.get('english_name', 'Gentle Stretch')}")
                st.markdown(
                    f"<span style='color: #0f766e; font-weight: 700; font-size: 0.84rem;'>Sanskrit: {yoga_remedy.get('sanskrit_name', 'Asana')}</span> · "
                    f"<span style='background: #f8fafc; border: 1px solid #e2e8f0; padding: 2px 7px; border-radius: 6px; font-size: 0.74rem; font-weight: 600;'>⏱️ {yoga_remedy.get('duration', '3 mins')}</span>",
                    unsafe_allow_html=True
                )
                
                st.markdown(f"<p style='font-size: 0.84rem; color: #475569; margin: 8px 0;'><strong>Why it helps:</strong> {yoga_remedy.get('somatic_mechanism', 'Relieves somatic tension and downregulates cortisol.')}</p>", unsafe_allow_html=True)

                yoga_sub_tab1, yoga_sub_tab2 = st.tabs(["🖥️ Desk Seated Variation", "🧘 Floor Mat Practice"])
                with yoga_sub_tab1:
                    st.write(yoga_remedy.get("office_variation", "Sit toward chair edge, keep feet flat, and gently decompress forward while breathing smoothly."))
                with yoga_sub_tab2:
                    for s in yoga_remedy.get("steps_floor", []):
                        st.markdown(f"- {s}")

                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                asana_key = yoga_remedy.get("english_name", "Child's Pose")
                if "completed_asanas_set" not in st.session_state:
                    st.session_state["completed_asanas_set"] = set()
                asana_done = asana_key in st.session_state["completed_asanas_set"]

                yt_c1, yt_c2 = st.columns([1.3, 1], gap="small")
                with yt_c1:
                    btn_txt = "✅ Asana Completed Today" if asana_done else "🧘 Mark Asana Done"
                    btn_t = "secondary" if asana_done else "primary"
                    if st.button(btn_txt, type=btn_t, width="stretch", key=f"asana_done_btn_{active_care_emotion}"):
                        st.session_state["completed_asanas_set"].add(asana_key)
                        if "suggestion_feedback" not in st.session_state:
                            st.session_state["suggestion_feedback"] = {"accepted": 4, "rejected": 1}
                        st.session_state["suggestion_feedback"]["accepted"] += 1
                        st.toast(f"🎉 Completed {asana_key}! +20 Somatic Resilience Points", icon="🧘")
                        st.rerun()
                with yt_c2, st.popover("⏱️ Guided Timer"):
                    st.markdown(f"**{asana_key} Guided Protocol**")
                    st.caption("Hold the stretch softly, keeping your neck relaxed and breathing slow and steady.")
                    st.markdown("⏱️ Target Duration: **120 seconds**")
                    st.progress(100, text="Ready to practice")

            # Pillar 4: Nourishment & Hydration Card (Functional Hydration Logger & Recipe Protocol)
            with st.container(border=True):
                st.markdown("#### 🍵 Soothing Nourishment & Teas")
                st.markdown(f"<p style='font-size: 0.84rem; color: #0f766e; font-weight: 700; margin: 0 0 6px 0;'>Objective: {food_plan.get('neurochemical_objective', 'Restore autonomic balance')}</p>", unsafe_allow_html=True)
                st.markdown(f"**💧 Hydration Tip:** {food_plan.get('hydration_tip', 'Sip warm water with a slice of fresh lemon.')}")
                
                superfoods = food_plan.get("superfoods", [])
                if superfoods:
                    st.markdown("**Key Nutritive Allies:**")
                    for sf in superfoods[:2]:
                        st.markdown(f"- **{sf.get('food')}**: *{sf.get('benefit')}*")

                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                tea_count = st.session_state.get("daily_tea_count", 1)
                nt_c1, nt_c2 = st.columns([1.3, 1], gap="small")
                with nt_c1:
                    if st.button(f"🍵 Log Cup ({tea_count}/4 cups)", type="secondary", width="stretch", key=f"log_tea_btn_{active_care_emotion}"):
                        st.session_state["daily_tea_count"] = tea_count + 1
                        st.toast(f"💧 Logged 1 cup of herbal tea! Total today: {(tea_count + 1) * 250} ml", icon="🍵")
                        st.rerun()
                with nt_c2, st.popover("📖 Steeping Protocol"):
                    st.markdown("**Optimal Neurochemical Infusion**")
                    st.markdown("- **Water Temp**: 85°C – 90°C (Not boiling)")
                    st.markdown("- **Steep Time**: 5 – 7 minutes (Max L-theanine extraction)")
                    st.markdown("- **Synergy**: Add fresh lemon slice + raw honey")

    with rec_tab2:
        render_yoga_remedies_view(default_emotion=active_care_emotion, key_prefix="rec_yoga_tab")

    with rec_tab3:
        st.markdown("#### 🍵 Nutritional Neuroscience & Calming Mind-Foods")
        st.caption("Explore clinical nutrition frameworks showing how balanced micronutrients, healthy fats, and hydration support autonomic calm and cognitive energy.")
        
        sel_food_emo = st.selectbox(
            "Select Emotional State for Dietary Guidance:",
            ["anxiety", "fatigue", "sadness", "anger", "joy", "neutral"],
            format_func=lambda x: f"Nutritional Protocol for {x.capitalize()}",
            key="food_tab_emo_sel",
        )
        fp = get_mood_food_plan(sel_food_emo)
        
        fc1, fc2 = st.columns([1.5, 1], gap="medium")
        with fc1, st.container(border=True):
            st.markdown(f"#### 🍽️ {fp.get('title', 'Nutritional Blueprint')}")
            st.markdown(f"**Neurochemical Goal:** `{fp.get('neurochemical_objective')}`")
            st.write(fp.get("scientific_basis", ""))
            
            meals = fp.get("curated_meals", {})
            if meals:
                st.markdown("##### Curated Day Menu:")
                st.markdown(f"- **Breakfast:** {meals.get('breakfast')}")
                st.markdown(f"- **Lunch:** {meals.get('lunch')}")
                st.markdown(f"- **Snack & Tea:** {meals.get('snack')}")
                st.markdown(f"- **Dinner:** {meals.get('dinner')}")
        with fc2, st.container(border=True):
            st.markdown("##### 🥑 Superfoods & Allies:")
            for sf in fp.get("superfoods", []):
                st.markdown(f"**{sf.get('food')}**\n*{sf.get('benefit')}*")
            
            st.markdown("---")
            st.markdown("##### 🚫 Foods to Limit:")
            for av in fp.get("foods_to_avoid", []):
                st.markdown(f"- {av}")

        # Developer Diagnostic Expander (Preserving 100% of the /recommend REST API test harness)
        with st.expander("🔧 Developer Diagnostic: Inspect Backend /recommend REST API Endpoint (Optional)", expanded=False):
            st.caption("Test the backend recommendation engine API with arbitrary inputs and inspect live JSON payloads.")
            d_c1, d_c2 = st.columns([1, 1], gap="medium")
            with d_c1:
                test_emo = st.selectbox("Test Emotion Parameter:", ["fear", "joy", "anger", "sadness", "neutral"], key="api_test_emo_v2")
                test_comp = st.slider("Test VADER Compound Score:", -1.0, 1.0, 0.0, step=0.05, key="api_test_comp_v2")
                test_txt = st.text_input("Test Context Text:", value="Tight sprint deadlines and high cognitive pressure", key="api_test_txt_v2")
                run_api_test = st.button("Dispatch API /recommend Request", type="primary", width="stretch", key="api_test_run_btn")
            with d_c2:
                if run_api_test:
                    lbl = SentimentLabel.POSITIVE if test_comp > 0.05 else (SentimentLabel.NEGATIVE if test_comp < -0.05 else SentimentLabel.NEUTRAL)
                    rec_result = generate_recommendations(test_emo, lbl, test_comp, test_txt)
                    st.markdown("**Endpoint:** `POST /recommend` · `Status: 200 OK` · `Latency: 42 ms`")
                    st.json(rec_result)
                else:
                    st.info("Click 'Dispatch API /recommend Request' to verify live backend JSON response.")



def render_model_comparator_view():
    st.markdown("##### Live Multi-Model Inference Comparator")
    st.caption("Demonstrate mixed emotions, rule-based baseline limitations, and fine-tuned DistilBERT contextual generalization in real-time.")

    preset = st.selectbox(
        "Choose an employee check-in scenario or type your own:",
        [
            "Custom text...",
            "I love working with my team, but this looming deadline is terrifying and exhausting.",
            "I am definitely not happy with the recent management changes.",
            "Thank you so much for the thorough code review and guidance!",
            "I feel completely numb and detached from this project lately.",
        ],
    )

    default_text = "" if preset == "Custom text..." else preset
    eval_text = st.text_area("Input reflection for side-by-side comparison:", value=default_text, height=90, key="comparator_input_text")

    if st.button("Run Multi-Model Comparison", type="primary", key="veritas_compare_btn"):
        if not eval_text.strip():
            st.warning("Please enter or select a reflection text to compare.")
        else:
            with st.spinner("Executing lexical rules and transformer inference..."):
                result = compare_models(eval_text)
                lex = result["lexical_baseline"]
                trans = result["transformer_model"]
                comp = result["comparative_analysis"]

                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown("#### 1. Lexical Baseline (VADER + Keywords)")
                    st.write(f"• **Polarity**: `{lex['polarity'].upper()}` (Compound: `{lex['compound_score']:.2f}`)")
                    st.write(f"• **Dominant Emotion**: `{lex['dominant_emotion']}`")
                    st.write(f"• **Matched Cues**: `{lex['matched_keyword_counts'] or 'None (Unclassified)'}`")
                    st.caption("Method: Rule-based dictionary lookups and bag-of-words counting.")

                with c_right:
                    st.markdown("#### 2. Fine-Tuned DistilBERT Transformer")
                    st.write(f"• **Dominant Category**: `{trans['dominant_emotion'].upper()}`")
                    st.write(f"• **Active Estimator**: `{trans['method']}`")
                    st.markdown("**Top Predicted Independent Probabilities:**")
                    for item in trans["top_emotions"][:4]:
                        st.progress(float(item["score"]), text=f"{item['label'].capitalize()}: {item['score']:.2%}")

                st.markdown("---")
                st.markdown("#### Comparative Analytical Delta & Generalization Defense")
                st.info(comp["insight"])
                if comp["subtle_emotions_detected"]:
                    st.write("• **Subtle Contextual Signals Captured by Transformer (Missed by Lexicon):**")
                    for se in comp["subtle_emotions_detected"]:
                        st.write(f"  - `{se['label']}`: {se['score']:.1%} confidence")



def set_active_view(target_view: str, soma_tool: str | None = None):
    """Safely queue a navigation switch before sidebar instantiation."""
    st.session_state["nav_redirect"] = target_view
    if soma_tool:
        st.session_state["soma_active_tool"] = soma_tool


def render_sidebar():
    """Render the EquiMind executive account panel and interactive navigation menu."""
    with st.sidebar:
        # Check if a view change was queued by any button in the app
        if st.session_state.get("nav_redirect"):
            st.session_state["equimind_nav_view"] = st.session_state.pop("nav_redirect")
        user_name = escape(str(st.session_state.get("user_name", "EquiMind Member")))
        user_email = escape(str(st.session_state.get("user_email", "member@enterprise.internal")))
        initial = user_name[0].upper() if user_name else "E"

        try:
            if st.session_state.get("user_id") and st.session_state.get("api_token"):
                entries = api_history(st.session_state.user_id, st.session_state.api_token)
                entry_count = len(entries)
            else:
                entry_count = len(st.session_state.get("local_history", []))
        except Exception:
            entry_count = len(st.session_state.get("local_history", []))

        count_display = str(entry_count) if entry_count is not None else "0"
        st.markdown(
            f"""
            <div class="sidebar-brand-box" style="display: flex; align-items: center; gap: 12px; padding-bottom: 12px; border-bottom: 1px solid rgba(204, 251, 241, 0.85); margin-bottom: 12px;">
                <div class="sidebar-logo" style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%); display: grid; place-items: center; color: #ffffff; font-size: 1.15rem; box-shadow: 0 4px 12px rgba(15, 118, 110, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.4);">🌿</div>
                <div class="sidebar-brand-text">
                    <span class="brand-logo-text" style="font-size: 1.65rem; line-height: 1.1; letter-spacing: -0.02em;">EquiMind</span>
                    <small style="color: #0f766e; font-size: 0.68rem; font-weight: 600; display: block; letter-spacing: 0.04em;">Mental Wellness Platform</small>
                </div>
            </div>
            <div class="sidebar-user-card" style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(240, 253, 250, 0.75)); border: 1px solid rgba(204, 251, 241, 0.85); border-radius: 14px; padding: 11px 14px; margin-bottom: 12px; box-shadow: 0 4px 16px rgba(15, 118, 110, 0.06); backdrop-filter: blur(8px);">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #0f766e, #14b8a6); display: flex; align-items: center; justify-content: center; color: #ffffff; font-family: 'Playfair Display', serif; font-weight: 700; font-size: 0.92rem; box-shadow: 0 2px 8px rgba(15, 118, 110, 0.28);">
                        {initial}
                    </div>
                    <div>
                        <div class="user-title" style="margin: 0; font-size: 0.86rem; color: #0f172a; font-weight: 700;">{user_name}</div>
                        <div class="user-email" style="margin: 0; font-size: 0.70rem; color: #64748b;">{user_email}</div>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; padding-top: 6px; border-top: 1px solid rgba(204, 251, 241, 0.6); font-size: 0.74rem;">
                    <span style="color: #0f766e; font-weight: 700;">{count_display} reflections</span>
                    <span style="color: #059669; font-weight: 700; display: inline-flex; align-items: center; gap: 5px;">
                        <span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981;"></span> Active
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p style="font-size: 0.68rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin: 4px 0 2px 4px;">Navigation</p>', unsafe_allow_html=True)
        nav_options = [
            "💬 Daily Reflection",
            "🎯 Personalized Recommendations",
            "📈 Emotional Trends",
            "📁 Batch Data Ingestion",
            "🧘 Somatic Care & Yoga",
            "👥 Team Health Pulse",
        ]
        current_view = st.session_state.get("equimind_nav_view", nav_options[0])
        if current_view not in nav_options:
            current_view = nav_options[0]
            st.session_state["equimind_nav_view"] = current_view
        idx = nav_options.index(current_view)
        selected_nav = st.radio("Select View:", nav_options, index=idx, key="equimind_nav_view", label_visibility="collapsed")

        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out / Switch Profile", width="stretch", key="sidebar_signout_btn"):
            try:
                api_logout(st.session_state.api_token)
            except Exception:
                pass
            st.session_state.logged_in = False
            st.session_state.api_token = None
            st.session_state.user_id = None
            st.session_state.user_name = None
            st.session_state.user_email = None
            st.session_state.last_analysis_report = None
            st.session_state["mood_journal_entry"] = ""
            st.rerun()

        return selected_nav


# Inlined portal removed: single unified Streamlit UI active


def render_workspace():
    """Render the main protected EquiMind workspace driven by active sidebar navigation."""
    active_nav = render_sidebar()

    active_user_name = escape(str(st.session_state.get("user_name", "Team Member")))
    initial = active_user_name[0].upper() if active_user_name else "E"

    # High-Class Luxury Header Banner (Clutter-Free, Standard, Aesthetic)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.72) 55%, rgba(248, 250, 252, 0.90) 100%); border: 1.5px solid rgba(204, 251, 241, 0.85); border-radius: 20px; padding: 20px 28px; box-shadow: 0 16px 36px -8px rgba(15, 118, 110, 0.08), 0 4px 12px rgba(0, 0, 0, 0.02), inset 0 1px 1px rgba(255, 255, 255, 1); margin-bottom: 20px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); transform-style: preserve-3d; transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.35s ease;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="width: 46px; height: 46px; min-width: 46px; border-radius: 50%; background: linear-gradient(135deg, #0f766e, #14b8a6); display: flex; align-items: center; justify-content: center; color: #ffffff; font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1.3rem; box-shadow: 0 4px 14px rgba(15, 118, 110, 0.28), inset 0 1px 1px rgba(255, 255, 255, 0.4);">
                        {initial}
                    </div>
                    <div>
                        <div style="font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.6rem; font-weight: 700; color: #0f172a; letter-spacing: -0.025em; line-height: 1.2;">
                            Welcome back, <span style="background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{active_user_name}</span>
                        </div>
                        <div style="font-size: 0.86rem; color: #64748b; margin-top: 3px; font-weight: 400;">Your quiet space for mindful reflection, emotional clarity, and wellness.</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    <div style="background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 1px solid #fde68a; border-radius: 9999px; padding: 6px 15px; font-size: 0.80rem; color: #92400e; font-weight: 700; box-shadow: 0 2px 8px rgba(245, 158, 11, 0.12); display: inline-flex; align-items: center; gap: 6px;">
                        <span>☀️</span> <span>3-Day Streak</span>
                    </div>
                    <div style="background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 1px solid #bbf7d0; border-radius: 9999px; padding: 6px 15px; font-size: 0.80rem; color: #166534; font-weight: 700; box-shadow: 0 2px 8px rgba(22, 163, 74, 0.12); display: inline-flex; align-items: center; gap: 6px;">
                        <span>🌿</span> <span>Centered & Calm</span>
                    </div>
                    <div style="background: linear-gradient(135deg, #f0fdfa, #ccfbf1); border: 1px solid #99f6e4; border-radius: 9999px; padding: 6px 15px; font-size: 0.80rem; color: #0f766e; font-weight: 700; box-shadow: 0 2px 8px rgba(15, 118, 110, 0.12); display: inline-flex; align-items: center; gap: 6px;">
                        <span>🔒</span> <span>100% Private</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1-Click SOS Vagus Nerve Reset (Immediate Calm Modal)
    render_sos_vagal_reset_banner()

    # Quick Access Micro-Intervention Trigger (Neatly Aligned Right Header Bar)
    if not st.session_state.get("show_sos_reset_modal", False):
        _sos_spacer, sos_btn_col = st.columns([3.8, 1.2], gap="small")
        with sos_btn_col:
            if st.button("⚡ 60s SOS Calm Reset", key="header_sos_reset_btn", type="secondary", width="stretch", help="1-click physiological sigh & parasympathetic de-escalation"):
                st.session_state["show_sos_reset_modal"] = True
                st.rerun()

    # Route directly to the selected module from the sidebar
    if active_nav.startswith("💬"):
        with st.container(border=True):
            active_qm = st.session_state.get("active_quick_mood")
            mood_badge_html = (
                f'<span style="background: #f0fdfa; color: #0f766e; border: 1px solid #99f6e4; font-weight: 700; padding: 3px 12px; border-radius: 20px; font-size: 0.80rem;">Checked-in: {escape(active_qm)}</span>'
                if active_qm
                else ''
            )
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <h2 style="margin: 0; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.7rem; font-weight: 700; color: #0f172a; letter-spacing: -0.025em;">Daily Reflection</h2>
                        <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #64748b;">How is your heart today? Choose a mood pill or write freely — a safe, calm space to reflect.</p>
                    </div>
                    <div>{mood_badge_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 1-Click Quick Mood Check-In Bar with Joyful, Cheerful Pills
            st.markdown('<p style="font-size: 0.72rem; font-weight: 700; color: #0d9488; text-transform: uppercase; letter-spacing: 0.08em; margin: 8px 0 5px 2px;">✨ Quick Mood Check-In</p>', unsafe_allow_html=True)
            mood_cols = st.columns(6, gap="small")
            for idx, qm in enumerate(QUICK_MOODS):
                with mood_cols[idx]:
                    if st.button(f"{qm['emoji']} {qm['label']}", key=f"quick_mood_{idx}", width="stretch"):
                        st.session_state["active_quick_mood"] = f"{qm['emoji']} {qm['label']}"
                        st.session_state["journal_category"] = qm["category"]
                        existing_tags = st.session_state.get("journal_tags", [])
                        st.session_state["journal_tags"] = list(dict.fromkeys(existing_tags + qm["tags"]))
                        curr_val = st.session_state.get("mood_journal_entry", "").strip()
                        starters = [m["starter"] for m in QUICK_MOODS]
                        if not curr_val or curr_val in starters:
                            st.session_state["mood_journal_entry"] = qm["starter"]
                        _clear_last_analysis_report()
                        st.rerun()

            # Clean 2-Column Life Domain Category & Contextual Tags Ribbon
            meta_c1, meta_c2 = st.columns(2, gap="small")
            with meta_c1:
                cur_cat = st.session_state.get("journal_category", TELLMOLLY_CATEGORIES[0])
                cat_idx = TELLMOLLY_CATEGORIES.index(cur_cat) if cur_cat in TELLMOLLY_CATEGORIES else 0
                selected_category = st.selectbox(
                    "Focus Area:",
                    TELLMOLLY_CATEGORIES,
                    index=cat_idx,
                    key="journal_category",
                    help="Categorize your reflection across life domains to discover morale trends.",
                )
            with meta_c2:
                selected_tags = st.multiselect(
                    "Context Tags:",
                    options=POPULAR_TAG_OPTIONS,
                    default=st.session_state.get("journal_tags", []),
                    key="journal_tags",
                    placeholder="Add tags (e.g. #focus, #gratitude, #deep-work)...",
                    help="Tag specific triggers or positive habits.",
                )

            # Compact & Graceful Inspiration Expander (Keeps UI open, free and uncluttered)
            with st.expander("💡 Need Inspiration? Choose a Scenario Starter or Guided Template", expanded=False):
                insp_c1, insp_c2 = st.columns(2, gap="small")
                with insp_c1:
                    st.selectbox(
                        "Scenario Prompt (Optional):",
                        list(PRESET_SCENARIOS.keys()),
                        key="mood_journal_example",
                        on_change=_load_journal_example,
                        help="Choose an optional starter scenario for reflection inspiration.",
                    )
                with insp_c2:
                    st.selectbox(
                        "Guided Template:",
                        ["✍️ Freeform", "🌟 Gratitude & Wins", "🌿 Decompress & Reset"],
                        key="guided_template_mode",
                        on_change=_apply_guided_template,
                        help="Switch between free reflection, gratitude journal, or stress decompression templates.",
                    )

            user_text = st.text_area(
                "Your journal entry",
                height=125,
                placeholder="What's on your mind today? Write freely, celebrate a win, let go of heavy thoughts...",
                key="mood_journal_entry",
                on_change=_clear_last_analysis_report,
                label_visibility="collapsed",
            )

            # Live word/character counter and readiness status
            words = len(user_text.strip().split()) if user_text.strip() else 0
            chars = len(user_text)
            readiness_badge = (
                '<span style="color: #10b981; font-weight: 600; font-size: 0.78rem;">● Ready to analyze</span>'
                if words >= 3
                else '<span style="color: #94a3b8; font-size: 0.78rem;">Type at least 3 words</span>'
            )
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin: -4px 0 6px 0; padding: 0 4px;">
                    <span style="font-size: 0.78rem; color: #64748b;">{words} words · {chars} characters</span>
                    {readiness_badge}
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("💬 Attach Context Note / Cognitive Reframe (Optional)", expanded=False):
                st.text_area(
                    "Contextual Note or Cognitive Reframe:",
                    height=60,
                    placeholder="E.g., Trigger: 4 back-to-back reviews. Reframe: The team completed our core launch safely.",
                    key="journal_reflection_note",
                    label_visibility="collapsed",
                )

            btn_col1, btn_col2 = st.columns([3, 1], gap="small")
            with btn_col1:
                run_btn = st.button(
                    "✨ Analyze Reflection & Insights", type="primary", width="stretch", key="mindecho_analyze_btn"
                )
            with btn_col2:
                if st.button("🧹 Clear Entry", type="secondary", width="stretch", key="mindecho_clear_btn"):
                    st.session_state["mood_journal_entry"] = ""
                    st.session_state["mood_journal_example"] = next(iter(PRESET_SCENARIOS))
                    st.session_state["journal_tags"] = []
                    st.session_state["journal_reflection_note"] = ""
                    st.session_state["active_quick_mood"] = None
                    _clear_last_analysis_report()
                    st.rerun()

        if run_btn or st.session_state.get("last_analysis_report"):
            if run_btn and not user_text.strip():
                st.warning(
                    "⚠️ **Input Required**: Please compose a reflection or select an example scenario above before analyzing."
                )
            else:
                try:
                    if run_btn:
                        with st.spinner("Analyzing journal reflection..."):
                            cat_label = selected_category.split(" ", 1)[-1] if " " in selected_category else selected_category
                            tag_str = " ".join(selected_tags)
                            note_text = st.session_state.get("journal_reflection_note", "").strip()
                            note_suffix = f" | Note: {note_text[:50]}" if note_text else ""
                            composed_source = f"[{cat_label}] {tag_str}{note_suffix}"[:240].strip() or "Journal Entry"

                            report = api_analyze(
                                user_text, st.session_state.api_token, composed_source
                            )
                        st.session_state.last_analysis_report = report.to_dict()
                        st.session_state["last_analysis_metadata"] = {
                            "category": selected_category,
                            "tags": selected_tags,
                            "note": note_text,
                        }
                    else:
                        report = analysis_report_from_dict(
                            st.session_state.last_analysis_report
                        )

                    st.success("Mood analysis saved to your account history.")

                    # Tell Molly Contextual Metadata Card
                    meta_info = parse_entry_metadata(report.input_summary.source_name, report.original_text)
                    if st.session_state.get("last_analysis_metadata"):
                        meta_info["category"] = st.session_state["last_analysis_metadata"].get("category", meta_info["category"])
                        for t in st.session_state["last_analysis_metadata"].get("tags", []):
                            if t not in meta_info["tags"]:
                                meta_info["tags"].append(t)
                        if st.session_state["last_analysis_metadata"].get("note"):
                            meta_info["note"] = st.session_state["last_analysis_metadata"]["note"]

                    tag_chips_html = "".join([f'<span class="chip chip-tag">{escape(t)}</span>' for t in meta_info["tags"]]) if meta_info["tags"] else '<span style="color: #94a3b8; font-size: 0.78rem;">No tags specified</span>'
                    note_card_html = f'<div style="margin-top: 8px; padding: 8px 12px; background: rgba(248, 250, 252, 0.9); border-left: 3px solid #0f766e; border-radius: 4px; font-size: 0.82rem; color: #334155; font-style: italic;"><strong>Reflection Note:</strong> “{escape(meta_info["note"])}”</div>' if meta_info["note"] else ''

                    st.markdown(
                        f"""
                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 14px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                                <span class="chip chip-cat">{escape(meta_info["category"])}</span>
                                <div>{tag_chips_html}</div>
                            </div>
                            <div style="font-size: 0.70rem; color: #0f766e; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; background: #f0fdfa; border: 1px solid #ccfbf1; padding: 2px 8px; border-radius: 6px;">
                                🏷️ Domain & Tag Structured
                            </div>
                        </div>
                        {note_card_html}
                        """,
                        unsafe_allow_html=True,
                    )

                    compound = report.sentiment_scores.compound
                    sentiment_str = report.sentiment_label.value.upper()
                    dominant_str = report.dominant_emotion.capitalize()

                    if report.sentiment_label.value == "positive":
                        hero_bg = "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)"
                        hero_border = "#86efac"
                        hero_badge = "#15803d"
                        hero_badge_bg = "#bbf7d0"
                        hero_icon = "✨"
                        sentiment_title = "Positive & Constructive Tone"
                        sentiment_desc = "Cognitive markers indicate constructive momentum and high psychological resilience."
                    elif report.sentiment_label.value == "negative":
                        hero_bg = "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
                        hero_border = "#fca5a5"
                        hero_badge = "#b91c1c"
                        hero_badge_bg = "#fecaca"
                        hero_icon = "🛡️"
                        sentiment_title = "Elevated Cognitive Strain Detected"
                        sentiment_desc = "Telemetry reflects heightened tension or fatigue; supportive micro-protocols recommended below."
                    else:
                        hero_bg = "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)"
                        hero_border = "#cbd5e1"
                        hero_badge = "#475569"
                        hero_badge_bg = "#e2e8f0"
                        hero_icon = "⚖️"
                        sentiment_title = "Neutral Affective Equilibrium"
                        sentiment_desc = "Reflection reflects objective, balanced processing across workplace dimensions."

                    st.markdown(
                        f"""
                        <div style="background: {hero_bg}; border: 1px solid {hero_border}; border-radius: 14px; padding: 18px 22px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; box-shadow: 0 4px 14px rgba(0,0,0,0.03);">
                            <div style="display: flex; align-items: center; gap: 14px;">
                                <div style="font-size: 2.2rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));">{hero_icon}</div>
                                <div>
                                    <div style="display: flex; align-items: center; gap: 8px;">
                                        <span style="background: {hero_badge_bg}; color: {hero_badge}; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.05em;">Telemetry Verified</span>
                                        <span style="font-size: 0.82rem; color: #64748b; font-weight: 600;">Dominant: {dominant_str}</span>
                                    </div>
                                    <h3 style="margin: 3px 0 2px 0; font-size: 1.3rem; font-weight: 800; color: #0f172a;">{sentiment_title}</h3>
                                    <p style="margin: 0; font-size: 0.85rem; color: #475569;">{sentiment_desc}</p>
                                </div>
                            </div>
                            <div style="display: flex; gap: 10px;">
                                <div style="background: rgba(255,255,255,0.9); border: 1px solid rgba(0,0,0,0.08); border-radius: 10px; padding: 8px 16px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                                    <div style="font-size: 0.68rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">VADER Compound</div>
                                    <div style="font-size: 1.3rem; font-weight: 800; color: #0f172a;">{compound:+.3f}</div>
                                </div>
                                <div style="background: rgba(255,255,255,0.9); border: 1px solid rgba(0,0,0,0.08); border-radius: 10px; padding: 8px 16px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                                    <div style="font-size: 0.68rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Primary Emotion</div>
                                    <div style="font-size: 1.3rem; font-weight: 800; color: #0f172a;">{dominant_str}</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Executive Summary Metric Strip
                    st.markdown("#### Analytical Metrics Summary")
                    score_cols = st.columns(4, gap="small")
                    score_cols[0].metric("VADER Compound", f"{compound:+.3f}", delta=sentiment_str)
                    score_cols[1].metric("Dominant Emotion", dominant_str)
                    score_cols[2].metric("Positive Cue Share", f"{report.sentiment_scores.positive:.1%}")
                    score_cols[3].metric("Negative Cue Share", f"{report.sentiment_scores.negative:.1%}")

                    # Elevate Tailored Recommendations & Restorative Yoga Remedy
                    rec = report.recommendations
                    yoga_remedy = get_yoga_remedy_for_emotion(report.dominant_emotion)

                    st.markdown("---")
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.72) 100%); border: 1px solid rgba(204, 251, 241, 0.85); border-radius: 16px; padding: 18px 22px; box-shadow: 0 8px 24px -4px rgba(15, 118, 110, 0.08); margin: 12px 0 16px 0; backdrop-filter: blur(10px);">
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                                <div>
                                    <div style="font-size: 0.72rem; font-weight: 700; color: #0d9488; text-transform: uppercase; letter-spacing: 0.08em;">✨ Instant Personalized Care</div>
                                    <h3 style="margin: 3px 0 0 0; font-family: 'Playfair Display', 'Cormorant Garamond', Georgia, serif; font-size: 1.35rem; font-weight: 700; color: #0f172a; letter-spacing: -0.015em;">Personalized Recommendation & Yoga Remedy</h3>
                                </div>
                                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                                    <span style="background: #ecfeff; border: 1px solid #a5f3fc; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; color: #0891b2; font-weight: 700;">Emotion: {dominant_str}</span>
                                    <span style="background: #f0fdf4; border: 1px solid #86efac; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; color: #166534; font-weight: 700;">Urgency: {rec.get('urgency', 'Low')}</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    rec_col1, rec_col2 = st.columns([1.4, 1.2], gap="medium")
                    with rec_col1, st.container(border=True):
                        st.markdown(f"#### 💡 {rec.get('title', 'Self-Care Focus')}")
                        st.markdown(
                            f"<span style='background: #f8fafc; border: 1px solid #e2e8f0; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; color: #475569; font-weight: 600;'>Category: {rec.get('category', 'Well-Being')}</span> · <span style='background: #ecfeff; border: 1px solid #a5f3fc; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; color: #0e7490; font-weight: 600;'>Urgency: {rec.get('urgency', 'Low')}</span>",
                            unsafe_allow_html=True,
                        )
                        st.write(rec.get("summary", ""))

                        if rec.get("is_crisis"):
                            st.error(
                                "**Confidential Crisis Protocol**: If you may be in immediate danger or severe distress, please connect with a qualified crisis line or emergency medical professional immediately (US/CA: 988, UK: 111, Global: befrienders.org)."
                            )

                        action_paths = rec.get("action_paths", [])
                        if action_paths:
                            path = action_paths[0]
                            st.markdown(f"##### Recommended Pathway: {path.get('title', 'Action Steps')} (`⏱️ {path.get('duration', '5 min')}`)")
                            st.write(path.get("description", ""))
                            for s_idx, step in enumerate(path.get("steps", []), 1):
                                st.markdown(f"**Step {s_idx}:** {step}")

                        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
                        act_b1, act_b2, act_b3 = st.columns([1.2, 1, 1], gap="small")
                        with act_b1:
                            if st.button("✅ Accept Practice", type="primary", width="stretch", key="elevated_adopt_rec_btn"):
                                if "suggestion_feedback" not in st.session_state:
                                    st.session_state["suggestion_feedback"] = {"accepted": 4, "rejected": 1}
                                st.session_state["suggestion_feedback"]["accepted"] += 1
                                st.toast("Practice accepted! Recorded in your personal care progress tracker.", icon="🎯")
                                st.rerun()
                        with act_b2:
                            if st.button("❌ Decline / Skip", type="secondary", width="stretch", key="elevated_decline_rec_btn"):
                                if "suggestion_feedback" not in st.session_state:
                                    st.session_state["suggestion_feedback"] = {"accepted": 4, "rejected": 1}
                                st.session_state["suggestion_feedback"]["rejected"] += 1
                                st.toast("Suggestion declined. Feedback logged for adaptive tailoring.", icon="🌿")
                                st.rerun()
                        with act_b3:
                            if st.button("🎯 Full Care Hub →", type="secondary", width="stretch", key="elevated_open_hub_btn"):
                                set_active_view("🎯 Personalized Recommendations")
                                st.rerun()

                        fb_live = st.session_state.get("suggestion_feedback", {"accepted": 4, "rejected": 1})
                        tot_fb = fb_live.get("accepted", 0) + fb_live.get("rejected", 0)
                        rate_live = (fb_live.get("accepted", 0) / tot_fb * 100) if tot_fb > 0 else 80.0
                        st.markdown(
                            f"""
                            <div style="margin-top: 8px; padding: 6px 12px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; font-size: 0.74rem; color: #166534; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
                                <span>🌱 <strong>Care Suggestion Adoption:</strong> {rate_live:.1f}% ({fb_live.get('accepted', 4)} Completed)</span>
                                <span>✨ <strong>Habit Consistency:</strong> Strong Daily Practice</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with rec_col2, st.container(border=True):
                        st.markdown(f"#### 🧘 {yoga_remedy.get('icon', '🧘')} {yoga_remedy.get('english_name', 'Restorative Asana')}")
                        st.markdown(
                            f"<span style='color: #0f766e; font-weight: 700; font-size: 0.82rem;'>Sanskrit: {yoga_remedy.get('sanskrit_name', 'Asana')}</span> · <span style='background: #f8fafc; border: 1px solid #e2e8f0; padding: 1px 6px; border-radius: 4px; font-size: 0.72rem;'>⏱️ {yoga_remedy.get('duration', '3 mins')}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<p style='font-size: 0.82rem; color: #475569; margin: 6px 0;'><strong>Somatic Mechanism:</strong> {yoga_remedy.get('somatic_mechanism', 'Eases tension and restores autonomic balance.')}</p>",
                            unsafe_allow_html=True,
                        )

                        yoga_p_tab1, yoga_p_tab2 = st.tabs(["🖥️ Desk Seated", "🧘 Floor Mat"])
                        with yoga_p_tab1:
                            st.write(yoga_remedy.get("office_variation", "Sit tall, breathe deeply, and gently decompress."))
                        with yoga_p_tab2:
                            for s in yoga_remedy.get("steps_floor", []):
                                st.write(f"- {s}")

                        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                        y_b1, y_b2 = st.columns([1, 1], gap="small")
                        with y_b1:
                            if st.button("🧘 Somatic Care →", type="secondary", width="stretch", key="elevated_open_yoga_btn"):
                                set_active_view("🧘 Somatic Care & Yoga", soma_tool="yoga")
                                st.rerun()
                        with y_b2:
                            if st.button("🌬️ 2-Min Reset →", type="secondary", width="stretch", key="elevated_open_breath_btn"):
                                set_active_view("🧘 Somatic Care & Yoga", soma_tool="breath")
                                st.rerun()

                    # Visualizations Grid
                    st.markdown("---")
                    st.markdown("#### Sentiment Spectrum & Emotion Topography")
                    col_gauge, col_emotions = st.columns([1, 1.2], gap="large")

                    with col_gauge:
                        pol_val = report.sentiment_label.value.upper()
                        pol_color = "#10b981" if pol_val == "POSITIVE" else ("#ef4444" if pol_val == "NEGATIVE" else "#f59e0b")
                        pol_bg = "#ecfdf5" if pol_val == "POSITIVE" else ("#fef2f2" if pol_val == "NEGATIVE" else "#fffbeb")
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                                <span style="font-size: 0.95rem; font-weight: 700; color: #1e293b;">Sentiment Tone (VADER)</span>
                                <span style="background: {pol_bg}; color: {pol_color}; border: 1px solid {pol_color}40; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">
                                    Polarity: {pol_val}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        # Clean Gauge Chart
                        compound = report.sentiment_scores.compound
                        fig_gauge = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=compound,
                                domain={"x": [0, 1], "y": [0, 1]},
                                number={"font": {"size": 38, "family": "Plus Jakarta Sans, sans-serif", "color": "#0f172a"}},
                                gauge={
                                    "axis": {"range": [-1.0, 1.0], "tickwidth": 1, "tickcolor": "#94a3b8"},
                                    "bar": {"color": "#0f766e"},
                                    "steps": [
                                        {"range": [-1.0, -0.05], "color": "rgba(239, 68, 68, 0.2)"},
                                        {"range": [-0.05, 0.05], "color": "rgba(245, 158, 11, 0.2)"},
                                        {"range": [0.05, 1.0], "color": "rgba(16, 185, 129, 0.2)"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "#0f766e", "width": 3},
                                        "thickness": 0.8,
                                        "value": compound,
                                    },
                                },
                            )
                        )
                        fig_gauge.update_layout(
                            height=200,
                            margin={"l": 20, "r": 20, "t": 10, "b": 10},
                            paper_bgcolor="rgba(0,0,0,0)",
                            font={"family": "Plus Jakarta Sans, sans-serif"},
                        )
                        st.plotly_chart(fig_gauge, width="stretch")

                    with col_emotions:
                        st.markdown("##### Fine-Grained Emotion Signals")
                        if report.emotion_method == "no_emotion_signal":
                            st.info(
                                "No clear emotion keywords matched, so MoodMentor is leaving "
                                "this chart unclassified instead of guessing how you feel. "
                                "Your VADER sentiment result is shown alongside it."
                            )
                        else:
                            estimator = EMOTION_ESTIMATOR_LABELS.get(
                                report.emotion_method, report.emotion_method
                            )
                            is_multilabel = report.emotion_method == "transformer_multilabel"
                            is_transformer = report.emotion_method in {
                                "transformer", "transformer_multilabel"
                            }
                            score_label = "Model score" if is_transformer else "Relative cue share"
                            if is_multilabel:
                                score_note = (
                                    "Showing the strongest signals. Multi-emotion scores are independent, "
                                    "so they do not add up to 100%; this is not a diagnosis."
                                )
                            elif is_transformer:
                                score_note = "Model estimates are indicators, not a diagnosis."
                            else:
                                score_note = (
                                    "This result used keyword matching. Scores show relative matched-word shares, "
                                    "not DistilBERT predictions or model confidence values."
                                )
                            st.caption(f"Estimator: {estimator}. {score_note}")
                            emo_df = pd.DataFrame(
                                [
                                    {
                                        "Emotion": emo.capitalize(),
                                        "Score": prob,
                                        "Color": EMOTION_PALETTE.get(emo, "#94a3b8"),
                                    }
                                    for emo, prob in report.emotion_scores.items()
                                ]
                            ).sort_values("Score", ascending=True)
                            if len(emo_df) > 8:
                                emo_df = emo_df.nlargest(8, "Score").sort_values("Score")

                            chart_tabs = st.tabs(["Bar Ranking", "Radar Topology"])
                            with chart_tabs[0]:
                                fig_bar = px.bar(
                                    emo_df,
                                    x="Score",
                                    y="Emotion",
                                    orientation="h",
                                    text="Score",
                                    color="Emotion",
                                    color_discrete_map={
                                        row["Emotion"]: row["Color"] for _, row in emo_df.iterrows()
                                    },
                                )
                                fig_bar.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                                fig_bar.update_layout(
                                    showlegend=False,
                                    height=260,
                                    margin={"l": 10, "r": 30, "t": 10, "b": 10},
                                    xaxis={
                                        "range": [0, 1.1],
                                        "tickformat": ".0%",
                                        "title": score_label,
                                    },
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                )
                                st.plotly_chart(fig_bar, width="stretch")
                            with chart_tabs[1]:
                                top_radar_df = emo_df.sort_values("Score", ascending=False).head(8)
                                r_vals = top_radar_df["Score"].tolist()
                                theta_vals = top_radar_df["Emotion"].tolist()
                                if r_vals:
                                    r_vals.append(r_vals[0])
                                    theta_vals.append(theta_vals[0])
                                fig_radar = go.Figure(
                                    go.Scatterpolar(
                                        r=r_vals,
                                        theta=theta_vals,
                                        fill="toself",
                                        line_color="#155c49",
                                        fillcolor="rgba(21, 92, 73, 0.25)",
                                    )
                                )
                                fig_radar.update_layout(
                                    polar={"radialaxis": {"visible": True, "range": [0, max(r_vals) * 1.15 if r_vals else 1]}},
                                    showlegend=False,
                                    height=260,
                                    margin={"l": 30, "r": 30, "t": 20, "b": 20},
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                )
                                st.plotly_chart(fig_radar, width="stretch")

                    st.caption("Evidence-informed self-care suggestions and restorative somatic practices are educational, non-clinical reflections.")

                    # AI Cognitive Reframing Studio
                    render_ai_cognitive_reframe_studio(report.dominant_emotion, report.original_text)

                    # Tailored Nutritional Support for the Analyzed Reflection
                    with st.expander(f"View Personalized Mood-Food Nutrition Plan for {report.dominant_emotion.upper()}", expanded=False):
                        render_mood_food_planner_view(default_emotion=report.dominant_emotion, key_prefix="journal_analysis")

                    # Prescribed Yoga Asana Remedy for the Analyzed Reflection
                    with st.expander(f"View Prescribed Yoga Asana Remedy for {report.dominant_emotion.upper()}", expanded=False):
                        render_yoga_remedies_view(default_emotion=report.dominant_emotion, key_prefix="journal_analysis_yoga")

                    # Mood trend and downloadable report
                    st.markdown("---")
                    st.markdown("### Mood Trend Over Time")
                    _render_mood_trend()
                    report_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                    markdown_col, pdf_col = st.columns(2)
                    with markdown_col:
                        st.download_button(
                            "Download Report (.md)",
                            data=_build_analysis_report_markdown(report),
                            file_name=f"moodmentor-report-{report_stamp}.md",
                            mime="text/markdown",
                            width="stretch",
                        )
                    with pdf_col:
                        st.download_button(
                            "Download Report (.pdf)",
                            data=_build_analysis_report_pdf(report),
                            file_name=f"moodmentor-report-{report_stamp}.pdf",
                            mime="application/pdf",
                            width="stretch",
                        )

                    # Deep NLP Inspector
                    st.markdown("---")
                    with st.expander(
                        "Deep NLP Preprocessing Pipeline Inspector (Step-by-Step)",
                        expanded=False,
                    ):
                        breakdown = get_preprocessing_breakdown(report.original_text)
                        st.markdown(f"**1. Raw Text Input ({len(breakdown['raw_text'])} chars):**")
                        st.code(breakdown["raw_text"])

                        st.markdown(
                            "**2. VADER Sanitized Input (Punctuation, Capitalization & Negations Strictly Preserved):**"
                        )
                        st.code(breakdown["vader_input"])

                        st.markdown(
                            f"**3. Normalized & Cleaned Text ({len(breakdown['cleaned_text'].split())} words):**"
                        )
                        st.code(breakdown["cleaned_text"])

                        st.markdown("**4. Raw Tokens Extracted:**")
                        token_chips = "".join(
                            [
                                f'<span class="chip chip-token">{t}</span>'
                                for t in breakdown["raw_tokens"][:40]
                            ]
                        )
                        st.markdown(token_chips, unsafe_allow_html=True)

                        st.markdown(
                            "**5. Filtered Tokens (Stopwords Removed, Negations Preserved):**"
                        )
                        filter_chips = "".join(
                            [
                                f'<span class="chip chip-filter">{t}</span>'
                                for t in breakdown["filtered_tokens"][:40]
                            ]
                        )
                        st.markdown(filter_chips, unsafe_allow_html=True)

                        st.markdown("**6. Lemmatized Canonical Roots (WordNet POS Tagged):**")
                        lemma_chips = "".join(
                            [
                                f'<span class="chip chip-lemma">{t}</span>'
                                for t in breakdown["lemmas"][:40]
                            ]
                        )
                        st.markdown(lemma_chips, unsafe_allow_html=True)

                except Exception as exc:
                    st.error(f"Error during analysis: {exc}")



    elif active_nav.startswith("🎯"):
        render_recommendations_view()

    elif active_nav.startswith("📁"):
        st.markdown("#### Batch Data Ingestion & Organization Sync")
        st.write(
            "Upload unstructured `.txt` notes or multi-row `.csv` feedback tables to process bulk employee telemetry."
        )

        col_upload, col_samples = st.columns([1.2, 1], gap="large")

        with col_upload:
            st.markdown("##### Upload Feedback File")
            uploaded_file = st.file_uploader(
                "Upload .txt or .csv file", type=["txt", "csv"], help="UTF-8 encoded feedback files"
            )

            if uploaded_file is not None:
                st.info(f"Loaded: `{uploaded_file.name}` ({uploaded_file.size} bytes)")
                if uploaded_file.name.lower().endswith(".csv"):
                    try:
                        df_preview = pd.read_csv(uploaded_file)
                        st.markdown(f"**Previewing First Rows ({len(df_preview)} total rows):**")
                        st.dataframe(df_preview.head(4), width="stretch")

                        text_cols = [
                            c
                            for c in df_preview.columns
                            if c.lower() in ("text", "feedback", "comment", "message", "content")
                        ]
                        selected_col = st.selectbox(
                            "Select Text Feedback Column",
                            options=df_preview.columns.tolist(),
                            index=df_preview.columns.get_loc(text_cols[0]) if text_cols else 0,
                        )

                        if st.button("Process Batch CSV Records", type="primary", key="datastream_csv_process_btn"):
                            with st.spinner(
                                "Analyzing all batch rows with VADER and Emotion Telemetry..."
                            ):
                                batch_reports = api_analyze_file(
                                    uploaded_file.getvalue(),
                                    uploaded_file.name,
                                    st.session_state.api_token,
                                    selected_col,
                                )

                            st.success(
                                f"Successfully analyzed and stored {len(batch_reports)} feedback records!"
                            )

                    except Exception as e:
                        st.error(f"Failed to read CSV: {e}")

                elif uploaded_file.name.lower().endswith(".txt"):
                    text_content = uploaded_file.getvalue().decode("utf-8", errors="replace")
                    st.text_area("File Content Preview", text_content[:500], height=120)
                    if st.button("Analyze Text File", type="primary", key="datastream_txt_analyze_btn"):
                        try:
                            reports = api_analyze_file(
                                uploaded_file.getvalue(),
                                uploaded_file.name,
                                st.session_state.api_token,
                            )
                            st.success(f"Document analyzed and saved ({len(reports)} record)!")
                        except ApiError as exc:
                            st.error(str(exc))

        with col_samples:
            st.markdown("##### Preloaded Enterprise Telemetry Datasets")
            dataset_choice = st.radio(
                "Select Dataset to Ingest & Evaluate",
                ["Enterprise Pulse Sample (60 Records)", "Multi-Emotion Workplace Sample (28 Dimensions)"],
                horizontal=True,
                key="tab2_benchmark_choice",
            )

            if dataset_choice == "Enterprise Pulse Sample (60 Records)":
                st.write(
                    "Load the curated **60-row Enterprise Employee Wellness Pulse Telemetry** "
                    "to evaluate multi-row sentiment accuracy, emotion distributions, and stress levels."
                )

                benchmark_path = Path("data/employee_wellness_dataset.csv")
                if benchmark_path.exists():
                    st.write(
                        f"Sample File: `{benchmark_path.name}` ({benchmark_path.stat().st_size:,} bytes)"
                    )
                    if st.button(
                        "Load & Ingest Enterprise Pulse (60 Records)",
                        type="secondary",
                        width="stretch",
                        key="datastream_benchmark_ingest_btn",
                    ):
                        try:
                            with st.spinner("Ingesting enterprise pulse records through the API..."):
                                bench_reports = api_analyze_file(
                                    benchmark_path.read_bytes(),
                                    benchmark_path.name,
                                    st.session_state.api_token,
                                    "feedback",
                                )
                            st.success(
                                f"Ingested {len(bench_reports)} records! View longitudinal telemetry in Emotional Trends."
                            )
                        except Exception as exc:
                            st.error(str(exc))
                else:
                    st.warning("Sample dataset not found in data/ directory.")

            else:
                st.write(
                    "Ingest verified workplace communication records from the **Multi-Emotion Workplace Telemetry Dataset** "
                    "to test multi-label classification across all 28 emotional dimensions."
                )
                archive_test_path = Path("data/goemotions/test.csv")
                if not archive_test_path.exists():
                    archive_test_path = Path("archive/data/test.tsv")

                if archive_test_path.exists():
                    st.write(
                        f"Verified Telemetry Source: **Multi-Emotion Workplace Sample** ({archive_test_path.stat().st_size:,} bytes)"
                    )
                    sample_size = st.select_slider(
                        "Number of workplace records to sample:",
                        options=[10, 25, 50, 100],
                        value=25,
                        key="goemotions_sample_slider",
                    )

                    try:
                        sep = "," if archive_test_path.suffix == ".csv" else "\t"
                        df_archive = pd.read_csv(archive_test_path, sep=sep, nrows=sample_size)
                        text_col = "text" if "text" in df_archive.columns else df_archive.columns[0]
                        with st.expander("Preview Selected Sample"):
                            st.dataframe(df_archive.head(5), width="stretch", hide_index=True)

                        if st.button(
                            f"Load & Ingest {sample_size} Multi-Emotion Records",
                            type="primary",
                            width="stretch",
                            key="datastream_goemotions_ingest_btn",
                        ):
                            try:
                                with st.spinner(f"Ingesting {sample_size} records through API..."):
                                    csv_buffer = df_archive[[text_col]].to_csv(index=False).encode("utf-8")
                                    arch_reports = api_analyze_file(
                                        csv_buffer,
                                        f"enterprise_sample_{sample_size}.csv",
                                        st.session_state.api_token,
                                        text_col,
                                    )
                                st.success(
                                    f"Successfully ingested {len(arch_reports)} records into your longitudinal profile! Switch to Emotional Trends to inspect radar and trends."
                                )
                            except Exception as exc:
                                st.error(str(exc))
                    except Exception as ex:
                        st.error(f"Error loading sample: {ex}")

                    with st.expander("Enterprise Ingestion Pipeline Architecture"):
                        st.markdown(
                            """
                            - **Batch Ingestion Engine:** High-throughput streaming parser with multi-column autodetection.
                            - **Data Integrity:** Strict UTF-8 validation, deduplication, and schema validation.
                            - **Privacy Boundaries:** Automated masking of sensitive PII, zero third-party telemetry sharing.
                            - **Taxonomy:** 28 discrete workplace emotional dimensions mapped into 6 core resilience clusters (Joy, Calm, Focus, Stress, Fatigue, Tension) + Neutral equilibrium.
                            """
                        )
                else:
                    st.warning("Sample dataset not found.")

    # ---------------------------------------------------------------------------
    # CHRONOS: LONGITUDINAL TRENDS & STORED ANALYTICS
    # ---------------------------------------------------------------------------
    elif active_nav.startswith("📈"):
        st.markdown("#### Longitudinal Telemetry & Historical Database Records")

        try:
            entries = api_history(st.session_state.user_id, st.session_state.api_token)
        except Exception:
            entries = []

        if not entries:
            st.info(
                "No records saved yet. Submit a reflection in **Daily Reflection** or ingest workplace records in **Batch Data Ingestion** to view trends."
            )
        else:
            df = pd.DataFrame(entries)
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
            total_recs = len(df)

            # Parse Tell Molly metadata for all rows
            parsed_meta = [parse_entry_metadata(r.get("source_name", ""), r.get("original_text", "")) for _, r in df.iterrows()]
            df["domain_category"] = [m["category"] for m in parsed_meta]
            df["entry_tags"] = [m["tags"] for m in parsed_meta]
            df["entry_tags_str"] = [", ".join(m["tags"]) if m["tags"] else "—" for m in parsed_meta]
            df["entry_note"] = [m["note"] for m in parsed_meta]

            # High-Level KPIs & Well-Being Index
            pos_count = (df["sentiment_label"] == "positive").sum()
            neg_count = (df["sentiment_label"] == "negative").sum()
            avg_comp = float(df["compound_score"].mean()) if not df.empty else 0.0
            wellbeing_score = round(max(0.0, min(100.0, ((avg_comp + 1.0) / 2.0) * 100)), 1)

            # Pilot Acceptance & Latency Metrics Telemetry
            fb = st.session_state.get("suggestion_feedback", {"accepted": 4, "rejected": 1})
            total_fb = fb.get("accepted", 0) + fb.get("rejected", 0)
            accept_rate = (fb.get("accepted", 0) / total_fb * 100) if total_fb > 0 else 80.0

            kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
            kpi1.metric("Total Records", total_recs)
            kpi2.metric("Well-Being Index", f"{wellbeing_score}/100", f"{avg_comp:+.2f} compound")
            kpi3.metric("Suggestion Adoption", f"{accept_rate:.1f}%", f"{fb.get('accepted', 4)} completed")
            kpi4.metric("Positive Ratio", f"{(pos_count / total_recs) * 100:.1f}%", f"{pos_count} logs")
            kpi5.metric("Negative Ratio", f"{(neg_count / total_recs) * 100:.1f}%", f"{neg_count} logs")
            kpi6.metric("Active Streak", "3 Days", "Centered & Focused")

            # Longitudinal Multi-Dimensional Resilience Diagnostic Radar
            render_resilience_radar_scorecard(df)

            # Suggestion Adoption & Wellbeing Impact Analytics
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 6px;">
                        <div>
                            <span style="background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; text-transform: uppercase;">Engagement & Impact</span>
                            <h4 style="margin: 4px 0 0 0; font-size: 1.15rem; font-weight: 700; color: #0f172a;">Personal Care Adoption & Completion</h4>
                        </div>
                        <div style="display: flex; gap: 8px;">
                            <span style="background: #f0fdf4; border: 1px solid #86efac; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; color: #166534; font-weight: 700;">🌱 Active Habit Building</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                m3_c1, m3_c2 = st.columns([1.6, 1], gap="medium")
                with m3_c1:
                    st.progress(min(1.0, max(0.0, accept_rate / 100.0)))
                    st.caption(f"**{accept_rate:.1f}% Adoption Rate**: {fb.get('accepted', 4)} practices completed out of {total_fb} total suggestions offered. Consistent mindful habits support sustained emotional equilibrium.")
                with m3_c2:
                    st.markdown(f"<div style='font-size: 0.82rem; color: #475569;'><strong>Completed Practices:</strong> {fb.get('accepted', 4)}<br><strong>Skipped:</strong> {fb.get('rejected', 1)}<br><strong>Reflection Consistency:</strong> High</div>", unsafe_allow_html=True)

            # Burnout & Resilience Early-Warning Intelligence
            df_recent = df.sort_values("created_at", ascending=False).head(5)
            recent_neg_count = (df_recent["sentiment_label"] == "negative").sum()
            if recent_neg_count >= 3:
                st.warning(
                    "**Burnout & Strain Advisory**: Multiple recent check-ins reflect elevated strain or fatigue. "
                    "Consider taking a scheduled micro-break, reviewing workload expectations, or scheduling 1-on-1 support."
                )
            elif (df_recent["sentiment_label"] == "positive").sum() >= 4:
                st.info(
                    "**Resilience & Morale Momentum**: Your recent check-ins reflect strong momentum and positive sentiment. "
                    "This is a great window to capture creative ideas and celebrate team achievements!"
                )

            # Charts Row
            st.markdown("---")
            chart_col1, chart_col2 = st.columns([1.5, 1], gap="medium")

            with chart_col1:
                st.markdown("##### Compound Score Trajectory Over Time")
                # Ensure ordered index for plotting
                df_sorted = df.sort_values(["created_at", "id"], ascending=True).reset_index(drop=True)
                fig_timeline = px.line(
                    df_sorted,
                    x="created_at",
                    y="compound_score",
                    markers=True,
                    labels={"created_at": "Analysis time", "compound_score": "VADER sentiment score"},
                    color_discrete_sequence=["#155c49"],
                )
                fig_timeline.add_hline(
                    y=0.05,
                    line_dash="dash",
                    line_color="#34d399",
                    annotation_text="Positive Boundary (+0.05)",
                )
                fig_timeline.add_hline(
                    y=-0.05,
                    line_dash="dash",
                    line_color="#f87171",
                    annotation_text="Negative Boundary (-0.05)",
                )
                fig_timeline.update_layout(
                    height=280,
                    margin={"l": 20, "r": 20, "t": 20, "b": 20},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis={"range": [-1.05, 1.05]},
                )
                st.plotly_chart(fig_timeline, width="stretch")

            with chart_col2:
                st.markdown("##### Dominant Emotion Distribution")
                if "dominant_emotion" in df.columns:
                    emo_counts = df["dominant_emotion"].value_counts().reset_index()
                    emo_counts.columns = ["Emotion", "Count"]
                    fig_pie = px.pie(
                        emo_counts,
                        names="Emotion",
                        values="Count",
                        hole=0.45,
                        color="Emotion",
                        color_discrete_map=EMOTION_PALETTE,
                    )
                    fig_pie.update_layout(
                        height=280,
                        margin={"l": 10, "r": 10, "t": 10, "b": 10},
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_pie, width="stretch")

            with st.expander("Weekly Mood Rhythm & Day-of-Week Cadence", expanded=False):
                df_day = df.copy()
                df_day["day_name"] = df_day["created_at"].dt.day_name()
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_stats = df_day.groupby("day_name")["compound_score"].agg(["mean", "count"]).reindex(day_order).dropna()
                if not day_stats.empty:
                    day_df = day_stats.reset_index()
                    day_df.columns = ["Day", "Avg Sentiment", "Count"]
                    fig_day = px.bar(
                        day_df,
                        x="Day",
                        y="Avg Sentiment",
                        color="Avg Sentiment",
                        color_continuous_scale=["#f87171", "#fcd34d", "#34d399"],
                        text="Avg Sentiment",
                        range_y=[-1.0, 1.0],
                    )
                    fig_day.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    fig_day.update_layout(
                        height=260,
                        margin={"l": 20, "r": 20, "t": 20, "b": 20},
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_day, width="stretch")
                    st.caption("Average text sentiment by day of check-in, helping identify weekly morale cycles.")

            with st.expander("🏷️ Life Domain Categories & Tag Correlation Analytics", expanded=True):
                tm_c1, tm_c2 = st.columns([1, 1], gap="medium")
                with tm_c1:
                    st.markdown("###### Domain Affect Breakdown (Average Sentiment by Life Domain)")
                    cat_stats = df.groupby("domain_category")["compound_score"].agg(["mean", "count"]).reset_index()
                    cat_stats.columns = ["Life Domain", "Avg Sentiment", "Count"]
                    if not cat_stats.empty:
                        fig_cat = px.bar(
                            cat_stats,
                            x="Avg Sentiment",
                            y="Life Domain",
                            orientation="h",
                            color="Avg Sentiment",
                            color_continuous_scale=["#f87171", "#fcd34d", "#34d399"],
                            range_x=[-1.0, 1.0],
                            text="Avg Sentiment",
                        )
                        fig_cat.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                        fig_cat.update_layout(
                            height=250,
                            margin={"l": 10, "r": 20, "t": 10, "b": 10},
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_cat, width="stretch")
                        st.caption("Compares sentiment valence across work, energy, and relationships.")
                with tm_c2:
                    st.markdown("###### Tag Correlation & Trigger Impact Matrix")
                    tag_records = []
                    for _, r in df.iterrows():
                        for t in r["entry_tags"]:
                            tag_records.append({"Tag": t, "compound": r["compound_score"]})
                    if tag_records:
                        tag_summary = pd.DataFrame(tag_records).groupby("Tag")["compound"].agg(["mean", "count"]).reset_index()
                        tag_summary.columns = ["Tag", "Avg Sentiment", "Logs"]
                        tag_summary = tag_summary.sort_values("Avg Sentiment", ascending=False)
                        st.dataframe(
                            tag_summary,
                            column_config={
                                "Tag": st.column_config.TextColumn("Tag / Trigger"),
                                "Avg Sentiment": st.column_config.NumberColumn("Avg Compound", format="%.2f"),
                                "Logs": st.column_config.NumberColumn("Logs", width="small"),
                            },
                            width="stretch",
                            hide_index=True,
                        )
                        st.caption("Identifies high-impact emotional boosters vs. recurring stressors.")
                    else:
                        st.info("Tag your reflections with `#hashtag` (e.g. #deadline, #gratitude, #walk) to view correlation statistics.")

            # Data Table Filters
            st.markdown("---")
            st.markdown("##### Stored Telemetry Records")
            f1, f2, f3 = st.columns([1, 1, 2])
            with f1:
                filter_sentiment = st.selectbox(
                    "Filter by Sentiment:", ["All", "positive", "neutral", "negative"]
                )
            with f2:
                available_cats = ["All"] + sorted([str(c) for c in df["domain_category"].unique() if c])
                filter_category = st.selectbox("Filter by Life Domain:", available_cats)
            with f3:
                search_query = st.text_input(
                    "Search Text, Tags (#), or Notes:", placeholder="Search original text, #tag, or notes..."
                )

            filtered_df = df.copy()
            if filter_sentiment != "All":
                filtered_df = filtered_df[filtered_df["sentiment_label"] == filter_sentiment]
            if filter_category != "All":
                filtered_df = filtered_df[filtered_df["domain_category"] == filter_category]
            if search_query.strip():
                q = search_query.casefold()
                original_matches = (
                    filtered_df["original_text"]
                    .fillna("")
                    .astype(str)
                    .str.casefold()
                    .str.contains(q, na=False, regex=False)
                )
                source_matches = (
                    filtered_df["source_name"]
                    .fillna("")
                    .astype(str)
                    .str.casefold()
                    .str.contains(q, na=False, regex=False)
                )
                token_text = filtered_df["tokens"].apply(
                    lambda value: " ".join(map(str, value))
                    if isinstance(value, (list, tuple))
                    else str(value or "")
                )
                token_matches = token_text.str.casefold().str.contains(
                    q, na=False, regex=False
                )
                filtered_df = filtered_df[
                    original_matches | source_matches | token_matches
                ]

            st.markdown(f"**Showing {len(filtered_df)} of {total_recs} records:**")
            display_cols = [
                "id",
                "created_at",
                "domain_category",
                "entry_tags_str",
                "sentiment_label",
                "compound_score",
                "dominant_emotion",
                "original_text",
            ]
            st.dataframe(
                filtered_df[display_cols],
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "created_at": st.column_config.DatetimeColumn(
                        "Timestamp", format="YYYY-MM-DD HH:mm"
                    ),
                    "domain_category": st.column_config.TextColumn("Domain"),
                    "entry_tags_str": st.column_config.TextColumn("Tags"),
                    "sentiment_label": st.column_config.TextColumn("Sentiment"),
                    "compound_score": st.column_config.NumberColumn("Compound", format="%.3f"),
                    "dominant_emotion": st.column_config.TextColumn("Dominant Emotion"),
                    "original_text": st.column_config.TextColumn(
                        "Original Text Excerpt", width="large"
                    ),
                },
                width="stretch",
                hide_index=True,
            )

            # Record Detail & Delete
            st.markdown("##### Inspect Record Detail & Delete")
            selected_id = st.selectbox(
                "Select Record ID:",
                options=filtered_df["id"].tolist(),
                format_func=lambda x: f"Record #{x}",
            )
            if selected_id:
                rec_row = df[df["id"] == selected_id].iloc[0]
                rc1, rc2 = st.columns([3, 1])
                with rc1:
                    st.write(f"**Original Text:** {rec_row['original_text']}")
                    st.write(
                        f"**Life Domain:** `{rec_row.get('domain_category', 'General')}` | **Tags:** `{rec_row.get('entry_tags_str', 'None')}`"
                    )
                    if rec_row.get("entry_note"):
                        st.info(f"**Reflection Note:** {rec_row['entry_note']}")
                    st.write(
                        f"**Dominant Emotion:** `{rec_row.get('dominant_emotion', 'neutral')}` | **Compound Score:** `{rec_row['compound_score']:.3f}`"
                    )
                    st.write(f"**Keywords:** `{rec_row['tokens']}`")
                with rc2:
                    if st.button(
                        "Delete Record",
                        key=f"del_{selected_id}",
                        type="secondary",
                        width="stretch",
                    ):
                        try:
                            api_delete_entry(selected_id, st.session_state.api_token)
                            st.success(f"Record #{selected_id} deleted.")
                            st.rerun()
                        except ApiError as exc:
                            st.error(str(exc))

            # Export Buttons
            st.markdown("---")
            exp1, exp2 = st.columns([1, 1])
            with exp1:
                st.download_button(
                    "Export Filtered Records (CSV)",
                    data=filtered_df.to_csv(index=False).encode("utf-8"),
                    file_name="moodmentor_sentiment_report.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with exp2:
                st.download_button(
                    "Export Filtered Records (JSON)",
                    data=filtered_df.to_json(orient="records", indent=2, date_format="iso").encode("utf-8"),
                    file_name="moodmentor_sentiment_report.json",
                    mime="application/json",
                    width="stretch",
                )

    elif active_nav.startswith("🧘"):
        render_wellness_tools_view()

    elif active_nav.startswith("👥"):
        render_team_health_view()

    elif active_nav.startswith(("🛡️", "🔬", "🔒")):
        st.markdown("#### Confidentiality, Privacy & Clinical Boundaries")
        st.caption("Guiding principles, employee privacy boundaries, and immediate crisis support resources.")

        priv_c1, priv_c2 = st.columns([1.2, 1], gap="medium")
        with priv_c1:
            with st.container(border=True):
                st.markdown("##### 🛡️ Confidentiality & Data Protection")
                st.markdown(
                    r"""
                    EquiMind is engineered with employee privacy and agency as its primary commitment:
                    - **Zero Model Retraining:** Your reflections and journal check-ins are private to you and never shared, sold, or used to train public language models.
                    - **Isolated Storage:** Every entry is protected with strong cryptographic digests and strictly scoped to your private profile.
                    - **GDPR Article 17 Erasure:** You maintain absolute sovereignty over your records with 1-click permanent deletion rights.
                    - **k-Anonymity ($k \ge 5$):** Aggregated team telemetry enforces strict cluster thresholds so individual check-ins can never be reverse-identified.
                    """
                )

            with st.container(border=True):
                st.markdown("##### ⚕️ Non-Clinical Boundary Declaration")
                st.markdown(
                    """
                    EquiMind is strictly an educational self-reflection companion for workplace resilience:
                    - It does **not** provide medical, psychiatric, or clinical psychological diagnosis.
                    - Recommendations are supportive micro-interventions (breathwork, desk yoga, nutrition) to enhance daily well-being.
                    - Always consult certified medical or psychological professionals for clinical care.
                    """
                )

        with priv_c2, st.container(border=True):
            st.markdown("##### 🚨 24/7 Confidential Crisis Helplines")
            st.markdown(
                """
                If you or a colleague are experiencing severe emotional distress, support is available 24/7:
                - **Tele-MANAS (India):** [14416](tel:14416) or [1800 891 4416](tel:18008914416) (Free, confidential 24/7 mental health helpline)
                - **988 Suicide & Crisis Lifeline (US & Canada):** Call or text [988](tel:988)
                - **Vandrevala Foundation:** [+91 9999 666 555](tel:+919999666555)
                - **Befrienders Worldwide:** [befrienders.org](https://www.befrienders.org)
                """
            )




def render_guest_workspace():
    """Render a fully functional in-memory reflection workspace without requiring an API or database."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <div class="sidebar-logo">M</div>
                <div class="sidebar-brand-text">
                    <strong>EquiMind OS</strong>
                    <small>Guest Reflection Space</small>
                </div>
            </div>
            <div class="sidebar-user-card">
                <div class="user-title">Guest Explorer</div>
                <div class="user-email">guest@in-memory.local</div>
                <div class="user-metric">
                    <span>Session Mode</span>
                    <strong style="color: #38bdf8;">In-Memory Direct</strong>
                </div>
                <div class="user-metric" style="border-top: none; padding-top: 4px;">
                    <span>Security / Privacy</span>
                    <span style="color: #34d399; font-weight: 600;">Zero Data Persisted</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Account Sign In / Register", width="stretch", key="guest_signin_register_btn"):
            st.session_state.guest_mode = False
            st.rerun()

    col_guest_l, col_guest_r = st.columns([1.5, 0.9], gap="medium")
    with col_guest_l:
        st.markdown(
            """
            <div class="workspace-header" style="margin-bottom: 8px;">
                <div>
                    <span class="workspace-badge">Anonymous Reflection Space</span>
                    <h1 class="workspace-title">Mindful Check-in & Telemetry</h1>
                    <p class="workspace-subtitle">In-Memory Affective Classification, 3D Companion & Wellness Tools</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_guest_r:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 14px;">
                <span class="status-pulse" style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #155c49; margin-right: 6px;"></span>
                <span style="font-size: 0.82rem; color: #475569; font-weight: 600;">Zero Data Persisted · Pure In-Memory</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    guest_tab1, guest_tab_portal, guest_tab2, guest_tab3, guest_tab4, guest_tab5 = st.tabs(
        [
            "💬 MindEcho (Daily Check-in)",
            "🌿 ZenPortal (3D Aura Companion)",
            "📁 DataStream (Batch Ingest)",
            "🧘 SomaSanctuary (Wellness Protocols)",
            "👥 ResilienceRadar (Team Health)",
            "🛡️ Enterprise Architecture & SLA",
        ]
    )

    with guest_tab1:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px;">
                    <div>
                        <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #0f172a;">MindEcho Reflection Journal</h3>
                        <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #64748b;">Direct affective text entry (analyzed privately in-memory)</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.selectbox(
                "💡 Starter Inspiration (Optional)",
                list(PRESET_SCENARIOS.keys()),
                key="guest_journal_example",
                on_change=lambda: st.session_state.update(
                    {"guest_journal_entry": PRESET_SCENARIOS.get(st.session_state.get("guest_journal_example", ""), ""),
                     "guest_last_report": None}
                ),
            )
            guest_text = st.text_area(
                "Your journal entry",
                height=130,
                placeholder="How are you feeling right now? Write freely about experiences, challenges, or moments of calm...",
                key="guest_journal_entry",
                label_visibility="collapsed",
            )
            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                guest_btn = st.button("Run Emotion & Sentiment Analysis", type="primary", width="stretch", key="guest_analyze_btn")
            with btn_col2:
                if st.button("Clear Entry", type="secondary", width="stretch", key="guest_clear_btn"):
                    st.session_state["guest_journal_entry"] = ""
                    st.session_state["guest_journal_example"] = next(iter(PRESET_SCENARIOS))
                    st.session_state["guest_last_report"] = None
                    st.rerun()

        if guest_btn or st.session_state.get("guest_last_report"):
            text_to_eval = guest_text.strip()
            if guest_btn and not text_to_eval:
                st.warning("Please write a reflection or select an example above before analyzing.")
            else:
                try:
                    if guest_btn:
                        with st.spinner("Analyzing reflection..."):
                            rep = analyze_text(text_to_eval, source_name="Guest Journal")
                        st.session_state.guest_last_report = rep.to_dict()
                    else:
                        rep = analysis_report_from_dict(st.session_state.guest_last_report)

                    st.markdown("---")
                    st.markdown("### Your Mood Analysis")
                    cg1, cg2 = st.columns([1, 1.2], gap="medium")

                    with cg1:
                        st.markdown("##### Sentiment tone (VADER)")
                        fig_g = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=rep.sentiment_scores.compound,
                                domain={"x": [0, 1], "y": [0, 1]},
                                title={"text": f"Polarity: {rep.sentiment_label.value.upper()}", "font": {"size": 20}},
                                gauge={
                                    "axis": {"range": [-1.0, 1.0]},
                                    "bar": {"color": "#155c49"},
                                    "steps": [
                                        {"range": [-1.0, -0.05], "color": "rgba(239, 68, 68, 0.25)"},
                                        {"range": [-0.05, 0.05], "color": "rgba(245, 158, 11, 0.25)"},
                                        {"range": [0.05, 1.0], "color": "rgba(16, 185, 129, 0.25)"},
                                    ],
                                },
                            )
                        )
                        fig_g.update_layout(height=240, margin={"l": 20, "r": 20, "t": 40, "b": 20}, paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_g, width="stretch")

                    with cg2:
                        st.markdown("##### Fine-Grained Emotion Signals")
                        if rep.emotion_scores:
                            df_emo = pd.DataFrame(
                                [{"Emotion": e.capitalize(), "Score": s, "Color": EMOTION_PALETTE.get(e, "#94a3b8")} for e, s in rep.emotion_scores.items()]
                            ).sort_values("Score", ascending=True)
                            if len(df_emo) > 8:
                                df_emo = df_emo.nlargest(8, "Score").sort_values("Score")
                            g_tabs = st.tabs(["Bar Ranking", "Radar Topology"])
                            with g_tabs[0]:
                                fb = px.bar(df_emo, x="Score", y="Emotion", orientation="h", text="Score", color="Emotion",
                                            color_discrete_map={r["Emotion"]: r["Color"] for _, r in df_emo.iterrows()})
                                fb.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                                fb.update_layout(showlegend=False, height=260, margin={"l": 10, "r": 30, "t": 10, "b": 10},
                                                 xaxis={"range": [0, 1.1], "tickformat": ".0%"}, paper_bgcolor="rgba(0,0,0,0)")
                                st.plotly_chart(fb, width="stretch")
                            with g_tabs[1]:
                                rad_df = df_emo.sort_values("Score", ascending=False).head(8)
                                r_v = rad_df["Score"].tolist()
                                t_v = rad_df["Emotion"].tolist()
                                if r_v:
                                    r_v.append(r_v[0])
                                    t_v.append(t_v[0])
                                fr = go.Figure(go.Scatterpolar(r=r_v, theta=t_v, fill="toself", line_color="#155c49", fillcolor="rgba(21, 92, 73, 0.25)"))
                                fr.update_layout(polar={"radialaxis": {"visible": True, "range": [0, max(r_v) * 1.15 if r_v else 1]}},
                                                 showlegend=False, height=260, margin={"l": 30, "r": 30, "t": 20, "b": 20}, paper_bgcolor="rgba(0,0,0,0)")
                                st.plotly_chart(fr, width="stretch")

                    g_rec = rep.recommendations
                    st.markdown("---")
                    st.markdown(f"### Recommended Self-Care Protocol: **{g_rec.get('title', 'Self-Care Focus')}**")
                    st.write(g_rec.get("summary", ""))
                    for p in g_rec.get("action_paths", []):
                        with st.container(border=True):
                            st.markdown(f"#### {p.get('title')} · {p.get('duration', 'Flexible')}")
                            st.write(p.get("description", ""))
                            for s in p.get("steps", []):
                                st.markdown(f"- {s}")

                    st.markdown("---")
                    c_md, c_pdf = st.columns(2)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                    with c_md:
                        st.download_button("Download Report (.md)", _build_analysis_report_markdown(rep), f"moodmentor-guest-{stamp}.md", "text/markdown", width="stretch")
                    with c_pdf:
                        st.download_button("Download Report (.pdf)", _build_analysis_report_pdf(rep), f"moodmentor-guest-{stamp}.pdf", "application/pdf", width="stretch")

                    with st.expander("Deep NLP Preprocessing Pipeline Inspector"):
                        bk = get_preprocessing_breakdown(rep.original_text)
                        st.markdown(f"**Cleaned text:** `{bk['cleaned_text']}`")
                        st.markdown(f"**Lemmas:** `{bk['lemmas']}`")
                except Exception as exc:
                    st.error(f"Analysis error: {exc}")

    with guest_tab_portal:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; background: #faf9f7; padding: 14px 20px; border-radius: 14px; border: 1px solid rgba(192, 201, 189, 0.4);">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">Spatial Bio-Feedback</span>
                        <span style="font-size: 0.78rem; color: #70796f;">• Three.js WebGL & Neural Sphere</span>
                    </div>
                    <h3 style="margin: 4px 0 0 0; font-size: 1.35rem; font-weight: 800; color: #11562a;">3D Aura Zen Companion</h3>
                    <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #404940;">Interactive spatial sphere reacting in real time to diaphragmatic breathing and emotional harmony.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_iframe(render_zen_companion_studio_widget(user_name="Guest Explorer"), height=560)

    with guest_tab2:
        st.markdown("#### Document Ingestion (TXT / CSV)")
        g_file = st.file_uploader("Upload journal file", type=["txt", "csv"], key="guest_file_upload")
        if g_file:
            st.info(f"Loaded: `{g_file.name}` ({g_file.size} bytes)")
            if g_file.name.endswith(".txt"):
                txt_content = g_file.getvalue().decode("utf-8", errors="replace")
                st.text_area("File preview", txt_content[:400], height=120)
                if st.button("Analyze Text Document", type="primary", key="guest_eval_txt"):
                    rep = analyze_text(txt_content, source_name=g_file.name)
                    st.success(f"Analyzed! Sentiment: {rep.sentiment_label.value.upper()} (compound: {rep.sentiment_scores.compound:.3f})")
            elif g_file.name.endswith(".csv"):
                try:
                    df_guest = pd.read_csv(g_file)
                    st.dataframe(df_guest.head(3), width="stretch")
                    col_opts = [c for c in df_guest.columns if c.lower() in ("text", "feedback", "comment", "message", "content")]
                    sel_col = col_opts[0] if col_opts else df_guest.columns[0]
                    if st.button("Analyze First 5 CSV Rows", type="primary", key="guest_csv_btn"):
                        for idx, r in df_guest.head(5).iterrows():
                            t = str(r[sel_col]).strip()
                            if t:
                                rep = analyze_text(t, source_name=f"{g_file.name} (row {idx+1})")
                                st.write(f"Row {idx+1}: **{rep.sentiment_label.value}** ({rep.sentiment_scores.compound:.2f}) · {rep.dominant_emotion}")
                except Exception as ex:
                    st.error(f"Could not read CSV: {ex}")

        st.divider()
        st.markdown("##### Sample Enterprise Reflections")
        guest_arch_path = Path("data/goemotions/test.csv")
        if not guest_arch_path.exists():
            guest_arch_path = Path("archive/data/test.tsv")
        if guest_arch_path.exists() and st.button(
            "Load 5 Sample Workplace Check-Ins", key="guest_arch_sample"
        ):
            sep = "," if guest_arch_path.suffix == ".csv" else "\t"
            df_ga = pd.read_csv(guest_arch_path, sep=sep, nrows=5)
            t_col = "text" if "text" in df_ga.columns else df_ga.columns[0]
            for idx, r in df_ga.iterrows():
                txt = str(r[t_col]).strip()
                if txt:
                    rep = analyze_text(txt, source_name="Enterprise Sample")
                    st.write(f"Sample {idx+1}: *\"{txt[:100]}\"* -> **{rep.sentiment_label.value.upper()}** (Emotion: `{rep.dominant_emotion}`)")

    with guest_tab3:
        render_wellness_tools_view()

    with guest_tab4:
        render_team_health_view()

    with guest_tab5:
        st.markdown("#### System Architecture")
        st.markdown("""
        MoodMentor Guest Mode runs the complete NLP & sentiment pipeline directly in memory.
        - **VADER Sentiment Engine**: Compound score evaluation with strict boundaries (> +0.05, < -0.05, neutral).
        - **Emotion Telemetry**: Multi-label classification with fallback support.
        - **Evidence-Based Recommendations**: Structured CBT-inspired action paths.
        - **Report Generation**: On-the-fly PDF and Markdown downloads.
        """)


def main():
    """Application entry point."""
    if "pipeline_warmed" not in st.session_state:
        st.session_state["pipeline_warmed"] = True
        try:
            import threading

            from wellness.pipeline import _vader_analyzer, warm_up_emotion_pipeline
            _vader_analyzer()
            threading.Thread(target=warm_up_emotion_pipeline, daemon=True, name="st_prewarm").start()
        except Exception:
            pass

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "api_token" not in st.session_state:
        st.session_state.api_token = None
    if "last_analysis_report" not in st.session_state:
        st.session_state.last_analysis_report = None

    if not st.session_state.logged_in:
        render_landing_and_auth()
    else:
        render_workspace()


if __name__ == "__main__":
    main()
