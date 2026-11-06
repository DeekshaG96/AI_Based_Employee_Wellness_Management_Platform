"""MoodMentor runner script.

Launches both the FastAPI backend and Streamlit frontend in a unified,
gracefully managed development/evaluation session.
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


def _check_endpoint(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False


def main() -> int:
    root = Path(__file__).resolve().parent
    os.chdir(root)

    print("=" * 68)
    print("  MoodMentor - AI-Based Employee Wellness Platform")
    print("  AI Wellness Platform - Evaluation & Service Runner")
    print("=" * 68)

    model_dir = root / "models" / "goemotions-distilbert"
    if model_dir.is_dir() and (model_dir / "model.safetensors").exists():
        print("  [ML Engine]  Fine-tuned DistilBERT GoEmotions Model: READY (Local Offline)")
    else:
        print("  [ML Engine]  Fine-tuned Model: Fallback / Lexical mode")

    python_exe = sys.executable

    # Check if API is already running
    api_proc = None
    if _check_endpoint("http://127.0.0.1:8000/health", timeout=1.0):
        print("  [FastAPI]    Service already running at http://127.0.0.1:8000")
    else:
        print("  [FastAPI]    Starting service at http://127.0.0.1:8000 ...")
        api_proc = subprocess.Popen(
            [python_exe, "-m", "uvicorn", "wellness_api.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=str(root),
        )
        for _ in range(30):
            if _check_endpoint("http://127.0.0.1:8000/health", timeout=1.0):
                print("  [FastAPI]    Service health check: OK")
                break
            time.sleep(0.5)
        else:
            print("  [WARNING] FastAPI health check timed out; continuing...")

    # Check if UI is already running
    ui_proc = None
    if _check_endpoint("http://localhost:8501/_stcore/health", timeout=1.0):
        print("  [Streamlit]  Service already running at http://localhost:8501")
    else:
        print("  [Streamlit]  Starting dashboard at http://localhost:8501 ...")
        ui_env = os.environ.copy()
        ui_env["API_URL"] = "http://127.0.0.1:8000"
        ui_proc = subprocess.Popen(
            [python_exe, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
            cwd=str(root),
            env=ui_env,
        )
        for _ in range(30):
            if _check_endpoint("http://localhost:8501/_stcore/health", timeout=1.0):
                print("  [Streamlit]  Service health check: OK")
                break
            time.sleep(0.5)

    print("-" * 68)
    print("  MoodMentor is active and ready:")
    print("  - Streamlit Dashboard: http://localhost:8501")
    print("  - FastAPI Docs (Swagger): http://127.0.0.1:8000/docs")
    print("  - Health Endpoint: http://127.0.0.1:8000/health")
    print("  Press Ctrl+C to stop all services.")
    print("-" * 68)

    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
            if api_proc and api_proc.poll() is not None:
                print("FastAPI process exited.")
                break
            if ui_proc and ui_proc.poll() is not None:
                print("Streamlit process exited.")
                break
    except KeyboardInterrupt:
        print("\nStopping MoodMentor services...")
    finally:
        for proc in (ui_proc, api_proc):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("MoodMentor shut down cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

