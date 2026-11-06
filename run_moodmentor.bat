@echo off
setlocal
cd /d "%~dp0"
if exist .venv-training\Scripts\python.exe (
    set "PYTHON_EXE=.venv-training\Scripts\python.exe"
) else if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)
"%PYTHON_EXE%" run_moodmentor.py %*

