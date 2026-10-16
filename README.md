# 🌱 MoodMentor: AI-Powered Emotional Tone Analyzer & Employee Wellness Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-15%20passed-success.svg)](tests/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Infosys Springboard](https://img.shields.io/badge/Infosys%20Springboard-Batch--4-orange.svg)](https://springboard.infosys.com/)

> **Infosys Springboard Virtual Internship — Milestone 1**  
> **Project Title:** AI-Powered Emotional Tone Analyzer and Well-Being Recommendation System  
> **Student:** Deeksha Ganesh ([@DeekshaG96](https://github.com/DeekshaG96))  
> **Official Milestone 1 Documentation:** [docs/Milestone1_Documentation.pdf](docs/Milestone1_Documentation.pdf) | [Markdown Version](docs/Milestone1_Documentation.md)

---

## 📌 Milestone 1 Overview

**Milestone 1** builds the core Natural Language Processing (NLP) and sentiment classification foundation for the MoodMentor wellness platform. It ingests workplace communication (journal entries, 1-on-1 meeting notes, pulse survey responses, and chat messages), executes a standardized 5-stage NLP preprocessing pipeline, computes baseline VADER compound polarity, logs persistent records in an SQLite database, and presents insights through an interactive 3-tab Streamlit dashboard.

### 🌟 Key Deliverables
- ✅ **Multi-Source Ingestion:** Direct text input, UTF-8 `.txt` file uploads, and multi-row `.csv` survey exports with intelligent column auto-detection (`feedback`, `text`, `comment`, `message`, `content`).
- ✅ **Sequential NLP Preprocessing:** Lowercasing, regex cleaning (URLs/symbols/digits), NLTK Punkt tokenization, English stop-word filtering, POS tagging, and WordNet lemmatization.
- ✅ **VADER Sentiment Engine:** Computes calibrated compound polarity ($-1.000$ to $+1.000$) alongside positive, neutral, and negative distributions.
- ✅ **Relational Persistence:** SQLite database storing user accounts with salted PBKDF2 password derivation and `mood_entries` tracking longitudinal sentiment telemetry.
- ✅ **Modern 3-Tab Streamlit UI:** Dedicated workflows for Direct Text Feedback, File Ingestion, and Stored Records & Analytics.
- ✅ **Test Suite:** 15 automated `pytest` test cases and clean `ruff` code formatting.

---

## 📸 Application Screenshots

### 1. Landing Portal & Authentication
Secure employee portal supporting registration, salted PBKDF2 authentication, and persistent SQLite database connection.
![Landing Portal](docs/screenshots/fig4_landing_auth.png)

### 2. Direct Feedback Analysis & NLP Preprocessing Pipeline
Real-time sentiment scoring featuring positive polarity (+0.857), 4-metric score cards, and an interactive step-by-step NLTK preprocessing breakdown with lemmatized keyword chips.
![Direct Feedback Analysis](docs/screenshots/fig1_direct_feedback_nlp.png)

### 3. Batch File Ingestion (.txt / .csv)
File uploader accepting UTF-8 `.txt` files and `.csv` survey exports with dynamic tabular preview and automatic column selection.
![File Ingestion](docs/screenshots/fig2_file_ingestion.png)

### 4. Stored Records & Analytics Dashboard
Persistent SQLite `mood_entries` table view displaying aggregate KPI metrics (Total Records, % Positive, % Neutral, % Negative, Avg Compound), sentiment filtering, and keyword search.
![Stored Records Dashboard](docs/screenshots/fig3_stored_records_analytics.png)

---

## 📂 Project Structure

```text
AI_Based_Employee_Wellness_Management_Platform/
├── app.py                         # Streamlit frontend & 3-tab interface
├── pyproject.toml                 # Package metadata & dependencies
├── Dockerfile                     # Production container deployment definition
├── render.yaml                    # Render Cloud Blueprint configuration
├── wellness/
│   ├── __init__.py                # Package exports
│   ├── cli.py                     # Command-line interface entry point
│   ├── errors.py                  # Custom domain exception classes
│   ├── ingestion.py               # Input validation & TXT/CSV file parsing
│   ├── pipeline.py                # Public analysis API coordinating NLP & VADER
│   ├── preprocessing.py           # Cleaning, tokenization, stop-words, lemmatization
│   ├── report.py                  # Typed data models & JSON serialization
│   └── storage.py                 # SQLite database setup, auth, & mood persistence
├── tests/
│   ├── test_pipeline.py           # Ingestion, preprocessing, and VADER tests
│   └── test_storage.py            # SQLite setup, auth, and mood persistence tests
├── sample_data/
│   ├── sample_feedback.txt        # Sample text file for ingestion testing
│   └── sample_survey.csv          # Sample CSV survey for batch testing
├── docs/
│   ├── Milestone1_Documentation.md# Comprehensive project documentation
│   ├── Milestone1_Documentation.pdf# Official Infosys Springboard PDF report
│   └── screenshots/               # Output execution screenshots
└── build_milestone1_pdf.py        # PDF compilation script (ReportLab)
```

---

## 🚀 Quickstart & Setup

### 1. Local Environment Setup
Requires Python 3.10 or higher:

```bash
# Clone the repository
git clone https://github.com/DeekshaG96/AI_Based_Employee_Wellness_Management_Platform.git
cd AI_Based_Employee_Wellness_Management_Platform

# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # On Windows (.venv/bin/activate on Linux/macOS)

# Install package with UI and development dependencies
pip install -e ".[dev,ui]"

# Download required NLTK corpuses and models
python -m wellness.cli --download-nltk-data
```

### 2. Launch the Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 💻 CLI Usage

The package exposes a command-line interface `wellness-analyze`:

```bash
# Direct text analysis
wellness-analyze --text "I feel very supported and motivated at work."
wellness-analyze --text "The workload is exhausting." --json

# File analysis
wellness-analyze --file sample_data/sample_feedback.txt
wellness-analyze --file sample_data/sample_survey.csv --text-column feedback --json
```

---

## 🧪 Testing & Code Quality

Run the test suite and code quality checks:

```bash
# Run 15 unit and integration tests
pytest

# Run Ruff linter
ruff check .
```

---

## 🐳 Docker Deployment

To build and run the platform inside a container:

```bash
docker build -t employee-wellness-platform .
docker run --rm -p 8501:8501 employee-wellness-platform
```

Open `http://localhost:8501`. Container health check is exposed at `/_stcore/health`.

---

## 🗺️ Roadmap & Future Scope

- **Milestone 1 (Complete):** Text Ingestion, NLP Preprocessing, VADER Sentiment Analysis, SQLite Persistence, Streamlit Dashboard.
- **Milestone 2:** Deep Emotion Classification using fine-tuned BERT and DistilBERT transformer models (Joy, Sadness, Anger, Fear, Surprise, Exhaustion).
- **Milestone 3:** AI-driven personalized wellness recommendation engine, longitudinal trend tracking, and FastAPI REST endpoints.
- **Milestone 4:** Production packaging, multi-tenant RBAC, CLI enhancements, and enterprise deployment.
