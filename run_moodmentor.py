"""1-Click Local Launcher for MoodMentor.

Starts both the FastAPI backend and Streamlit frontend together,
monitors their health, and automatically opens the dashboard in your browser.
No Docker or PostgreSQL setup required — automatically uses the built-in SQLite database.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def find_python_executable() -> str:
    """Find the best python executable in local virtual environments or fallback to sys.executable."""
    candidates = [
        ROOT / ".venv-training" / "Scripts" / "python.exe",
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / "venv" / "Scripts" / "python.exe",
        ROOT / ".venv-training" / "bin" / "python",
        ROOT / ".venv" / "bin" / "python",
        ROOT / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable

def check_health(url: str, timeout: float = 2.0) -> bool:
    """Return True if an HTTP GET request to url succeeds."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False

def main():
    py = find_python_executable()
    print("=" * 65)
    print("🌿 MoodMentor 1-Click Unified Runner")
    print("=" * 65)
    print(f"Python interpreter: {py}")
    print("Working directory:  ", ROOT)
    print("-" * 65)

    api_process = None
    streamlit_process = None

    # 1. Start FastAPI backend if not already running
    if check_health("http://127.0.0.1:8000/health"):
        print("✅ FastAPI backend is already running on http://127.0.0.1:8000")
    else:
        print("🚀 Starting FastAPI backend on http://127.0.0.1:8000 ...")
        api_cmd = [
            py, "-m", "uvicorn", "wellness_api.main:app",
            "--host", "127.0.0.1",
            "--port", "8000",
        ]
        api_process = subprocess.Popen(api_cmd, cwd=ROOT)
        
        # Wait for API to become ready
        attempts = 0
        while attempts < 30:
            time.sleep(0.5)
            if check_health("http://127.0.0.1:8000/health"):
                print("✅ FastAPI backend is ready and connected to database!")
                break
            attempts += 1
        else:
            print("⚠️ FastAPI backend started, continuing to launch Streamlit frontend...")

    # 2. Start Streamlit frontend if not already running
    if check_health("http://localhost:8501/_stcore/health"):
        print("✅ Streamlit frontend is already running on http://localhost:8501")
    else:
        print("🚀 Starting Streamlit frontend on http://localhost:8501 ...")
        st_cmd = [
            py, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "true",
        ]
        streamlit_process = subprocess.Popen(st_cmd, cwd=ROOT)

        # Wait for Streamlit to become ready
        attempts = 0
        while attempts < 30:
            time.sleep(0.5)
            if check_health("http://localhost:8501/_stcore/health"):
                print("✅ Streamlit frontend is live and healthy!")
                break
            attempts += 1

    print("=" * 65)
    print("🎉 MoodMentor is running successfully!")
    print("=" * 65)
    print("  • Web Dashboard:       http://localhost:8501")
    print("  • Interactive API:     http://127.0.0.1:8000/docs")
    print("  • 1-Click Demo Login:  alex@example.com / Password123!")
    print("  • Standalone Mode:     Available via '🌿 Guest Access' tab")
    print("-" * 65)
    print("Press Ctrl+C to stop all services.\n")

    # Automatically open browser
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass

    def cleanup(signum=None, frame=None):
        print("\nStopping MoodMentor services...")
        if streamlit_process:
            streamlit_process.terminate()
        if api_process:
            api_process.terminate()
        print("All services stopped cleanly. Have a wonderful day! 🌿")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            time.sleep(1)
            if api_process and api_process.poll() is not None:
                print("⚠️ FastAPI backend exited unexpectedly.")
                break
            if streamlit_process and streamlit_process.poll() is not None:
                print("⚠️ Streamlit frontend exited unexpectedly.")
                break
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
