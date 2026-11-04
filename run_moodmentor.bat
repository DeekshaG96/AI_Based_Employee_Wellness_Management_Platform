@echo off
title MoodMentor Unified Runner
echo =========================================================
echo Starting MoodMentor Platform (FastAPI + Streamlit)...
echo =========================================================
cd /d "%~dp0"

if exist ".venv-training\Scripts\python.exe" (
    ".venv-training\Scripts\python.exe" run_moodmentor.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_moodmentor.py
) else (
    python run_moodmentor.py
)

pause
