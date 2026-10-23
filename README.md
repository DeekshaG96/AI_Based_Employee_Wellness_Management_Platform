# MoodMentor

**AI-Powered Emotional Tone Analyzer and Well-Being Recommendation System**

MoodMentor is an Infosys Springboard internship project that analyzes written journal reflections, presents sentiment and emotion signals, and suggests general well-being actions. Each check-in offers three tailored ways forward, such as a quick reset, a practical next step, or reaching out for support. Users can explore one option at a time; all options are saved with the analysis and included in the downloadable report. The Streamlit dashboard uses an authenticated FastAPI service; PostgreSQL stores user accounts, analysis history, and recommendations.

## Architecture

```text
Streamlit dashboard :8501
        │ authenticated HTTP requests
        ▼
FastAPI service :8000 ── NLTK preprocessing + VADER + Hugging Face emotion model
        │
        ▼
PostgreSQL :5432 (host port :5433 in local Compose)
```

The API owns authentication, analysis, and persistence. The dashboard does not open the database directly. Analysis requests run the NLP pipeline; the service stores each result and its recommendation in one transaction. Preprocessing uses spaCy's `en_core_web_sm` for tokenization and lemmatization when installed, with NLTK POS tagging and WordNet lemmatization as a fallback. The emotion classifier uses `j-hartmann/emotion-english-distilroberta-base` by default. Each result identifies its estimator. If the model cannot load, a simple lexical fallback keeps the demo responsive; those fallback scores are heuristic indicators, not transformer confidence scores.

## Run the full application with Docker Compose

1. Install Docker Desktop and start its engine.
2. Copy `.env.example` to `.env` and replace `POSTGRES_PASSWORD` with a local password. The `DATABASE_URL` line uses the same value.
3. From this folder, start the services:

   ```powershell
   docker compose up --build
   ```

4. Open the dashboard at [http://localhost:8501](http://localhost:8501) and sign in or create an account.
5. In **Mood Journal**, write a reflection and select **Analyze my mood**. The result includes a mood trend chart and a downloadable Markdown report; longer-term history is available in **Longitudinal Trends & Analytics**.
6. Explore the API and its interactive documentation at [http://localhost:8000/docs](http://localhost:8000/docs). Its health endpoint is [http://localhost:8000/health](http://localhost:8000/health).

The first build installs the NLP dependencies and downloads NLTK resources. The first emotion analysis may also download the Hugging Face model, so it can take longer than later requests. Compose binds the web services and database to loopback for local development.

## Run the API and dashboard directly

Use a Python 3.10+ environment and a reachable PostgreSQL server. Install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords wordnet omw-1.4 punkt punkt_tab averaged_perceptron_tagger_eng vader_lexicon
```

Set `DATABASE_URL` to your PostgreSQL connection string before starting the API, then run the two processes in separate terminals:

```powershell
$env:DATABASE_URL = 'postgresql+psycopg://moodmentor:your-password@localhost:5432/moodmentor'
uvicorn wellness_api.main:app --reload --port 8000
```

```powershell
$env:API_URL = 'http://127.0.0.1:8000'
streamlit run app.py
```

The API creates the tables on startup. `requirements.txt` installs the complete local development environment, including Transformers and PyTorch. The container builds use smaller, separate API and UI dependency sets.

## API overview

Authenticated endpoints require `Authorization: Bearer <access_token>`. Register and login responses return a token; the API stores only its SHA-256 digest and expires sessions after 30 days.

- `GET /health`: Check API and database connectivity.
- `POST /auth/register`: Create an account and session.
- `POST /auth/login`: Verify credentials and create a session.
- `GET /auth/me`: Return the signed-in account.
- `POST /auth/logout`: Revoke the current session.
- `POST /analyze`: Analyze and save a text entry.
- `POST /analyze/file`: Analyze and save a `.txt` file or CSV rows.
- `GET /history/{user_id}`: Return only the signed-in user's paginated history.
- `DELETE /history/{entry_id}`: Delete one of the signed-in user's records.

Example analysis request:

```json
{
  "text": "I feel proud of the progress our team made this week.",
  "source_name": "Journal Entry"
}
```

The response includes the VADER scores, polarity label, token and preprocessing information, emotion scores, recommendation, record ID, and timestamp. CSV uploads accept a `text_column` query parameter; without it, MoodMentor looks for a common header such as `text`, `feedback`, `comment`, `message`, or `content`. Uploads are limited to 2 MB.

## PostgreSQL data model

- `users`: account name, unique email, password hash, and creation time.
- `auth_sessions`: user ID, hashed bearer token, creation time, and expiry.
- `mood_logs`: owner, source, input and cleaned text, tokens, VADER values, emotion values, and timestamp.
- `recommendations`: one linked recommendation per mood log, including actions and urgency.

The schema is defined in `wellness_api/models.py`; tables are initialized by the API. The legacy SQLite functions remain in `wellness/storage.py` for existing tests and import support. The Streamlit application no longer uses them.

## Import existing SQLite history

The migration script preserves account IDs, password hashes, log IDs, timestamps, scores, emotion data, and recommendations where available. It refuses to write into a destination that already has users or mood logs.

1. Start the local database only: `docker compose up -d db`.
2. Install `requirements-dev.txt` in the host Python environment.
3. Point the migration command at the Compose database's host port and preview the row counts:

   ```powershell
   $env:DATABASE_URL = 'postgresql+psycopg://moodmentor:your-password@localhost:5433/moodmentor'
   python -m scripts.migrate_sqlite_to_postgres --source .\wellness.db --dry-run
   ```

4. If the source and destination counts are expected, run the same command without `--dry-run`.
5. Start the API and dashboard with `docker compose up --build`.

Keep a backup of `wellness.db` before migration. The script reads the SQLite source and does not delete or modify it. Texts imported from older versions may not have all current preprocessing fields; the migration derives those fields from the stored tokens and source text.

## Test and lint

Install `requirements-dev.txt`, then run:

```powershell
pytest
ruff check .
```

The API tests use an isolated SQLite database and stub the heavy model call. The existing pipeline and storage tests continue to cover the legacy NLP and SQLite utilities.

## Project layout

```text
app.py                         Streamlit dashboard
wellness/                      NLTK, VADER, emotion pipeline, reports, legacy SQLite support
wellness/api_client.py         Authenticated dashboard-to-API client
wellness_api/                  FastAPI routes, schemas, security, and SQLAlchemy models
scripts/migrate_sqlite_to_postgres.py
tests/                         Pipeline, legacy storage, and API contract tests
Dockerfile.api                 FastAPI and NLP runtime
Dockerfile.ui                  Streamlit runtime
docker-compose.yml             PostgreSQL, API, dashboard, and model cache
docs/PROJECT_REPORT.md         Mentor-facing report and presentation outline
```

## Responsible-use notes

MoodMentor is a learning and self-reflection tool, not a clinical assessment, diagnosis, or emergency service. Text classifiers can misread context, sarcasm, dialect, and individual communication styles. Do not use these outputs to make employment, performance, or other high-impact decisions. Journal text can contain sensitive personal information: use consent, restricted access, a retention policy, encrypted transport, and production-grade secret management before any real deployment. The Compose credentials are for local development only.
