# 🧠 MoodMentor: AI-Powered Employee Wellness Platform

[![CI](https://github.com/DeekshaG96/AI_Based_Employee_Wellness_Management_Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/DeekshaG96/AI_Based_Employee_Wellness_Management_Platform/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Infosys Springboard Virtual Internship**  
> **Student:** Deeksha Ganesh ([@DeekshaG96](https://github.com/DeekshaG96)) • **Report:** [Milestone 1 Documentation PDF](docs/Milestone1_Documentation.pdf)

**MoodMentor** is an intelligent workplace mental health and emotion telemetry platform. It analyzes emotional tone from employee feedback, journal entries, 1-on-1 notes, and pulse surveys using a standardized NLP pipeline, precision VADER sentiment scoring, fine-grained Transformer emotion classification, and persistent SQLite relational storage.

---

## 🌟 Key Features

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Authentication & Persistence** | SQLite3 + PBKDF2 | Secure landing page with 1-click Demo access, user registration/login, salted cryptographic password hashing (120,000 iterations), and relational mood history tracking. |
| **NLP Preprocessing** | NLTK + RegEx | Contraction expansion, URL/HTML/email sanitization, tokenization, stop-word filtering with **negation preservation**, POS tagging, and WordNet lemmatization. |
| **Sentiment Analysis** | NLTK VADER | Rule-based polarity engine evaluated on sanitized text (preserving capitalization, punctuation, and intensifiers) with **strict boundary logic** (`> +0.05` Pos, `< -0.05` Neg, `[-0.05, +0.05]` Neu). |
| **Emotion Intelligence** | Hugging Face Transformers | Fine-grained classification across 7 primary emotional states (`Joy`, `Sadness`, `Fear`, `Anger`, `Surprise`, `Disgust`, `Neutral`) with an interactive Plotly horizontal distribution chart. |
| **Well-Being Recommendations** | Context Engine | Empathetic, actionable self-care interventions (breathing exercises, cognitive cool-down, task triage, positive reinforcement) with automated crisis screening. |
| **Batch Ingestion & Benchmark** | Pandas + Plotly | Multi-row `.csv` and `.txt` ingestion with table preview, 1-click **Kaggle 60-row workplace wellness dataset**, and longitudinal polarity trajectory spline charts. |

---

## 🏛️ System Architecture

```text
[Input Feedback / File Upload]
              │
              ▼
    [NLP Preprocessing] ──(Sanitized Text: Caps/Punctuation Intact)──> [VADER Polarity Engine]
              │                                                              │
              ├─(Tokens / Lemmatization)─> [Keywords Breakdown]              ▼
              │                                                      [Strict Thresholds]
              ▼                                                      (> +0.05, < -0.05, Neu)
    [Hugging Face Transformer]                                               │
   (DistilRoBERTa Emotion Model)                                             ▼
              │                                                  [Recommendation Engine]
              └───────(7 Probability Scores)────────────────────> (Actionable Interventions)
                                                                             │
                                                                             ▼
                                                                [SQLite Relational Storage]
                                                               (users & mood_entries tables)
```

---

## 📋 Evaluation Faults Resolution Verification

| Fault Identified | Evaluation Requirement | Architectural Resolution in Code | Status |
| :--- | :--- | :--- | :---: |
| **1. NLP Preprocessing** | Retain raw text with capitalization & punctuation for VADER; expand contractions; keep sentiment negations. | `wellness/preprocessing.py`: `sanitize_text_for_vader()` preserves casing/punctuation; `expand_contractions()` handles contractions; stop-word filtering strictly excludes negations (`no`, `not`, `never`, `cannot`). | ✅ **Verified** |
| **2. VADER Thresholds** | Explicit, mutually exclusive conditional blocks: `> +0.05` (Pos), `< -0.05` (Neg), `[-0.05, +0.05]` (Neu). | `wellness/pipeline.py`: Strict conditional classification eliminating boundary overlap or gaps. | ✅ **Verified** |
| **3. UI & Transparency** | Input validation warning on empty submissions; dedicated file uploader with previews; modular structure. | `app.py`: `st.warning` on empty text; dedicated 4-tab workflow with CSV preview, Plotly trend timeline, and SQLite persistence. | ✅ **Verified** |

---

## 🚀 Quickstart

### 1. Clone Repository
```bash
git clone https://github.com/DeekshaG96/AI_Based_Employee_Wellness_Management_Platform.git
cd AI_Based_Employee_Wellness_Management_Platform
```

### 2. Setup Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e ".[dev,ui]"
pip install -r requirements.txt

# Download required NLTK corpora
python -m wellness.cli --download-nltk-data
```

### 3. Launch Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🖥️ Command-Line Interface (CLI)

The package provides a built-in CLI for terminal analysis and batch pipelines:

```bash
# Analyze direct text
python -m wellness.cli --text "I am feeling proud of our successful team launch!"

# Output machine-readable JSON
python -m wellness.cli --text "Terrible communication today." --json

# Ingest file
python -m wellness.cli --file sample_data/sample_survey.csv --text-column feedback
```

---

## 🧪 Automated Testing & Code Quality

The repository is covered by 21 automated tests covering text preprocessing, strict threshold boundaries, emotion classification, recommendation mapping, file ingestion, and database persistence.

```bash
# Run pytest test suite (100% passing)
pytest -v

# Run Ruff linter (100% clean)
ruff check .
```

---

## 📂 Project Structure

```text
AI_Based_Employee_Wellness_Management_Platform/
├── app.py                         # Modern 4-tab Streamlit web application & UI
├── requirements.txt               # Pinned production dependencies
├── pyproject.toml                 # Package configuration, dependencies & tool configs
├── Dockerfile                     # Container deployment definition
├── render.yaml                    # Cloud deployment blueprint
├── wellness/                      # Core Python package
│   ├── __init__.py                # Public API exports
│   ├── cli.py                     # Command-line interface entry point
│   ├── errors.py                  # Domain-specific exception hierarchy
│   ├── ingestion.py               # File (.txt, .csv) & text validation
│   ├── pipeline.py                # VADER scoring, DistilRoBERTa emotion & recommendations
│   ├── preprocessing.py           # Sanitization, contraction expansion & lemmatization
│   ├── report.py                  # Typed data models (AnalysisReport, SentimentScores)
│   └── storage.py                 # SQLite database & salted PBKDF2 authentication
├── data/
│   └── employee_wellness_dataset.csv  # Curated 60-row Kaggle workplace wellness benchmark
├── docs/                          # Official Milestone 1 documentation & screenshots
│   ├── Milestone1_Documentation.md
│   ├── Milestone1_Documentation.pdf
│   └── screenshots/
└── tests/                         # Automated unit & integration tests
    ├── test_pipeline.py           # 17 NLP, VADER, Emotion & Recommendation tests
    └── test_storage.py            # 4 SQLite persistence & authentication tests
```

---

## 📄 License & Contributing

Distributed under the [MIT License](LICENSE). Review [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) for contribution guidelines and security vulnerability reporting.
