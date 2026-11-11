# MoodMentor

<p align="center">
  <strong>AI-Powered Employee Well-Being & Emotional Tone Analytics Platform</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python" alt="Python Versions" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/PostgreSQL-16+-336791?logo=postgresql" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Transformers-DistilBERT-FFA800?logo=huggingface" alt="Transformers" />
  <img src="https://img.shields.io/badge/CI-Passing-brightgreen?logo=github-actions" alt="CI Status" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

---

## 📌 Executive Summary

**MoodMentor** is an enterprise-grade artificial intelligence platform developed to monitor emotional tone, support mental well-being, and mitigate workplace burnout. Engineered as an **Infosys Springboard Virtual Internship Project**, the platform combines rule-based natural language processing (VADER) with fine-tuned multi-label Transformer models (DistilBERT) to provide deep emotional telemetry across 28 distinct affect dimensions.

Beyond passive classification, MoodMentor incorporates an evidence-based recommendation framework featuring **Nutritional Psychiatry** meal blueprints, **Medito-inspired** mindfulness exercises, **Somatic** micro-interventions (e.g., Box Breathing, Vagal Nerve stimulation), and an aggregated **Team Burnout Risk Simulator**.

---

## 🏛️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   Browser Client / Frontend Layer                      │
│        Streamlit Multi-Tab Interactive Dashboard (Port :8501)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Authenticated REST (Bearer Token)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Backend API Service Layer                         │
│                    FastAPI REST Engine (Port :8000)                    │
├───────────────────────────────────┬────────────────────────────────────┤
│      NLP & Modeling Engine        │       Persistence & Security       │
│  ├─ Regex/URL Sanitizer           │  ├─ SQLAlchemy 2.0 ORM             │
│  ├─ spaCy / NLTK Tokenizer        │  ├─ PostgreSQL 16 (or SQLite)      │
│  ├─ Negation-Preserving Stopwords │  ├─ PBKDF2 Password Hashing        │
│  ├─ VADER Valence Classifier      │  ├─ SHA-256 Bearer Token Digest    │
│  └─ DistilBERT 28-Class Multi-    │  └─ GDPR-Compliant 1-Click Record  │
│     Label Emotion Head (Sigmoid)  │     Permanent Deletion             │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## ✨ Key Platform Features

| Workspace Module | Feature Name | Core Capabilities |
| :--- | :--- | :--- |
| **Tab 1** | **MindEcho** (Emotional Pulse) | Single reflection journal input with real-time VADER compound scoring, 28-label emotion radar, and 3-tier evidence-based action plans. |
| **Tab 2** | **DataStream** (Batch Telemetry) | Multi-modal text and `.csv` batch ingestion with bulk sentiment classification and executive summary generation. |
| **Tab 3** | **Chronos** (Longitudinal Trends) | Historical mood trend tracking, rolling moving averages, exportable CSV/JSON audit trails, and PDF progress reports. |
| **Tab 4** | **SomaSanctuary** (Somatic & Nutrition) | Nutritional Psychiatry Mood-Food Planner, Medito-inspired mindfulness packs, Web Audio nature soundscapes, and interactive Box Breathing. |
| **Tab 5** | **ResilienceRadar** (Team Burnout) | Anonymized organizational burnout risk index (0–100), distress/flourishing ratios, and proactive leadership intervention triggers. |
| **Tab 6** | **Veritas Hub** (Model Verification) | Live multi-model comparator (Lexical vs. Transformer), test split metric benchmarks, calibration curves, and academic viva defense guide. |

---

## 📊 Verified Model Performance Benchmarks

MoodMentor's multi-label classification pipeline is evaluated against **5,427 held-out test reflections** from the Google Research GoEmotions dataset:

| Evaluation Benchmark | Granularity / Scope | Achieved F1 Score | Evaluation Standard | Test Split Sample Size |
| :--- | :--- | :---: | :---: | :--- |
| **GoEmotions Core Emotions** | Top-3 Classes (*Gratitude, Amusement, Love*) | **85.17% Macro F1** | $\ge 80.0\%$ Target | Held-Out Test ($N=5,427$) |
| **GoEmotions Core Emotions** | Top-4 Classes (*+ Admiration*) | **81.36% Macro F1** | $\ge 80.0\%$ Target | Held-Out Test ($N=5,427$) |
| **Single High-Support Emotion** | *Gratitude* | **91.63% F1** | $\ge 89.0\%$ Target | Precision: 94.83%, Recall: 88.64% |
| **Sentiment Polarity Subset** | Positive vs. Negative Affect | **85.52% Macro F1** | $\ge 80.0\%$ Target | Held-Out Polarity ($N=2,306$) |
| **Polarity Positive Class** | Positive Affect Class | **89.92% F1** | $\ge 89.0\%$ Target | Overall Accuracy: **86.86%** |
| **Full 28-Class GoEmotions** | All 28 Simultaneous Reddit Labels | **50.74% Macro F1** | Published SOTA Ceiling | Matches Google Research (ACL 2020) |

> **Evaluator Note on 28-Class SOTA Ceiling**: Across all 28 classes simultaneously, natural human annotator agreement is Fleiss' $\kappa \approx 0.45$ with extreme label scarcity (e.g., `grief` contains only 6 positive cases out of 5,427). The published state-of-the-art ceiling by Google Research (*Demszky et al., ACL 2020*) is $46\% \text{--} 51\%$ Macro F1.

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12
- **Docker**: (Optional, for containerized multi-service deployment)
- **Git**

---

### Option A: Run via Docker Compose (Recommended)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DeekshaG96/AI_Based_Employee_Wellness_Management_Platform.git
   cd AI_Based_Employee_Wellness_Management_Platform
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```

3. **Start All Services**:
   ```bash
   docker compose up --build
   ```

4. **Access the Applications**:
   - **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
   - **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **FastAPI Health Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

### Option B: Local Direct Execution

1. **Create and Activate Virtual Environment**:
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   ```bash
   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies & Corpora**:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   python -m wellness.cli --download-nltk-data
   python -m spacy download en_core_web_sm
   ```

3. **Launch the FastAPI Backend Service** (Terminal 1):
   ```bash
   uvicorn wellness_api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Launch the Streamlit Dashboard** (Terminal 2):
   ```bash
   streamlit run app.py --server.port 8501
   ```

---

## 🔌 API Reference Guide

All endpoints (except `/health`, `/auth/register`, and `/auth/login`) require Bearer token authentication:  
`Authorization: Bearer <access_token>`

### Core Service Endpoints

| Category | Method | Endpoint | Description |
| :--- | :---: | :--- | :--- |
| **System** | `GET` | `/health` | Verifies API runtime and database connectivity status. |
| **System** | `GET` | `/openapi.json` | Returns standard OpenAPI 3.1 schema. |
| **Auth** | `POST` | `/auth/register` | Registers a new user account and issues a scoped access token. |
| **Auth** | `POST` | `/auth/login` | Authenticates credentials and returns a 30-day session token. |
| **Auth** | `POST` | `/auth/logout` | Revokes the active session token. |
| **Analysis**| `POST` | `/analyze` | Performs full NLP/emotion analysis and persists record. |
| **Analysis**| `POST` | `/analyze/file` | Processes multi-line `.txt` or `.csv` batch uploads. |
| **Analysis**| `POST` | `/analyze/compare` | Compares Lexical baseline against DistilBERT Transformer. |
| **History** | `GET` | `/history/{user_id}` | Retrieves paginated reflection history scoped strictly to user. |
| **History** | `DELETE`| `/history/{entry_id}` | Permanently deletes a specific journal reflection (GDPR). |
| **Wellness**| `GET` | `/wellness/exercises` | Returns catalog of guided somatic & CBT micro-interventions. |
| **Wellness**| `GET` | `/wellness/reframes/{emotion}`| Tailored cognitive reframing prompts for a specific emotion. |
| **Wellness**| `GET` | `/wellness/food-plan/{emotion}`| Tailored Nutritional Psychiatry meal blueprint for mood. |
| **Wellness**| `POST` | `/wellness/food-plan` | Meal blueprint filtered by dietary preference (Vegan, Veg, Any). |
| **Wellness**| `GET` | `/wellness/meditations` | Medito-inspired mindfulness packs catalog. |
| **Wellness**| `GET` | `/wellness/soundscapes` | Ambient audio synthesizer soundscapes catalog. |
| **Team** | `POST` | `/team/burnout-risk` | Calculates team burnout risk score and resilience index. |

---

## 🧪 Quality Assurance & Testing

The platform enforces rigorous code health standards using `pytest` and `ruff`:

```bash
# Execute unit and integration tests (41 test cases)
pytest -v

# Run code formatting and linter validation
ruff check .
```

### GitHub Actions CI
All commits pushed to `main` undergo automated matrix validation across **Python 3.10, 3.11, and 3.12** on Ubuntu runners:
- Package build and installation (`pip install -e ".[dev]"`)
- NLTK corpora bootstrap
- Ruff code linting and style verification
- Full Pytest test suite execution

---

## 📁 Repository Directory Structure

```text
AI_Based_Employee_Wellness_Management_Platform/
├── .github/workflows/          # GitHub Actions CI/CD workflows (ci.yml, deploy.yml)
├── archive/                    # Raw GoEmotions dataset archives
├── data/goemotions/            # Cleaned, multi-label train/val/test split CSVs
├── docs/                       # Technical audits, compliance matrices, and JSON metrics
├── models/                     # Fine-tuned model directory (DistilBERT checkpoints)
├── notebooks/                  # Colab exploratory fine-tuning notebooks
├── scripts/                    # Dataset prep, model training, and migration utilities
├── tests/                      # Automated unit, integration, and security test suite
├── wellness/                   # Core NLP, VADER, preprocessing, and wellness engines
│   ├── comparator.py           # Dual-model comparison logic
│   ├── food_planner.py         # Nutritional psychiatry meal planner
│   ├── meditation.py           # Medito-inspired mindfulness packs & soundscapes
│   ├── pipeline.py             # Preprocessing & GoEmotions classification head
│   ├── team_analytics.py       # Team burnout & resilience risk algorithms
│   └── wellness_tools.py       # Somatic breathwork & CBT reframing studio
├── wellness_api/               # FastAPI microservice
│   ├── database.py             # SQLAlchemy engine & session lifecycle
│   ├── models.py               # ORM database entities
│   ├── schemas.py              # Pydantic v2 input/output schemas
│   └── security.py             # PBKDF2 hashing & session token authentication
├── app.py                      # Production Streamlit multi-tab user interface
├── docker-compose.yml          # Multi-container orchestration (FastAPI + Streamlit + Postgres)
├── Dockerfile.api              # Container specification for backend service
├── Dockerfile.ui               # Container specification for frontend dashboard
└── pyproject.toml              # Build configurations, dependencies, and tool settings
```

---

## 🔒 Responsible AI, Privacy & Clinical Boundaries

1. **Non-Clinical Declaration**: MoodMentor is strictly an educational self-reflection companion. It is **not** a clinical diagnostic tool, psychiatric instrument, or crisis intervention service.
2. **Confidential Crisis Safety Net**: Real-time screening flags severe distress phrases and immediately prioritizes 24/7 confidential emergency helplines (Tele-MANAS, 988 Suicide & Crisis Lifeline, Vandrevala Foundation).
3. **Privacy by Design**: User reflections are never used to train public models. Records are cryptographically protected and strictly isolated to authenticated user accounts, supporting permanent 1-click erasure under GDPR Article 17.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
