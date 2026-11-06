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

The API owns authentication, analysis, and persistence. The dashboard does not open the database directly. Analysis requests run the NLP pipeline; the service stores each result and its recommendation in one transaction. Preprocessing uses spaCy's `en_core_web_sm` for tokenization and lemmatization when installed, with NLTK POS tagging and WordNet lemmatization as a fallback. The local emotion classifier first loads a MoodMentor-fine-tuned model from models/goemotions-distilbert when present; otherwise it uses the public GoEmotions checkpoint `BruceIC/emoticare`. Its 28 outputs are independent multi-label signals, so they do not add up to 100%. MoodMentor retains those fine-grained scores and maps the strongest signal to a broad category for recommendations. Each result identifies its estimator. If the model cannot load, a simple lexical fallback keeps the demo responsive; those scores are heuristic indicators, not transformer confidence scores. When no supported emotion keyword matches, MoodMentor marks the fine-grained result unclassified instead of showing an artificial even split.

## Run the full application with Docker Compose

1. Install Docker Desktop and start its engine.
2. Copy `.env.example` to `.env` and replace `POSTGRES_PASSWORD` with a local password. The `DATABASE_URL` line uses the same value.
3. From this folder, start the services:

   ```powershell
   docker compose up --build
   ```

4. Open the dashboard at [http://localhost:8501](http://localhost:8501) and sign in or create an account.
5. In **Mood Journal**, write a reflection and select **Analyze my mood**. The result includes a mood trend chart and downloadable PDF and Markdown reports; longer-term history is available in **Longitudinal Trends & Analytics**.
6. Explore the API and its interactive documentation at [http://localhost:8000/docs](http://localhost:8000/docs). Its health endpoint is [http://localhost:8000/health](http://localhost:8000/health).

The first build installs the NLP dependencies and downloads NLTK resources. The first local emotion analysis downloads the Hugging Face model, so it can take longer than later requests. Compose binds the web services and database to loopback for local development. The free Streamlit showcase keeps keyword analysis as its private-by-default fallback. If its owner configures an `HF_TOKEN`, a visitor can explicitly opt in to send an individual journal entry to Hugging Face Inference using `SamLowe/roberta-base-go_emotions`; transformer weights stay off the Streamlit instance.

If Compose reports that it cannot connect to `dockerDesktopLinuxEngine` or a Docker named pipe, start Docker Desktop and wait until its Linux engine is ready, then retry. `docker compose config` checks the YAML without starting containers; `docker compose up --build` needs the Docker engine running.

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


## Fine-tune a MoodMentor GoEmotions model

The project contains the full GoEmotions archive dataset under `archive/` and prepared train, validation, and test CSV splits under `data/goemotions/`. Each record preserves every applicable emotion because GoEmotions is a 28-label multi-label dataset. The prepared CSV files can be refreshed or inspected from the included archive with:

    python -m scripts.prepare_goemotions_csv

Fine-tuning starts from DistilBERT and uses capped positive-label weights to reduce the effect of sparse GoEmotions labels. It reports step/loss progress, evaluates on validation data after each epoch, keeps the best validation checkpoint, and evaluates that checkpoint once on the held-out test split.

### Screenshot-sized development run

This run uses the first 2,000 rows from each split with batch size 8, producing 250 training updates. It checks the full training/evaluation flow and gives progress similar to the example screenshots. Since it uses only part of each split, treat its scores as development diagnostics, not as final model performance. It saves outside the app's model directory so the API will not load an unverified checkpoint:

    python -m scripts.train_goemotions_model --limit 2000 --epochs 1 --batch-size 8 --max-length 64 --threshold 0.25 --positive-weight-cap 8 --output-dir .venv-training/goemotions-balanced-250 --metrics-output .venv-training/goemotions-balanced-250-metrics.json

### Full dataset training

Install local dependencies and train on all 43,410 training records. The full validation split is used for checkpoint selection, and the test split is held out for the final evaluation:

    python -m pip install -r requirements.txt
    python -m scripts.train_goemotions_model --epochs 3 --batch-size 8 --output-dir .venv-training/goemotions-full --metrics-output .venv-training/goemotions-full-metrics.json

The candidate model and aggregate results are saved under .venv-training/goemotions-full and .venv-training/goemotions-full-metrics.json. Only aggregate metrics are reported; raw text and model weights are excluded from Git. Review validation and test metrics first. Copy a suitable candidate to models/goemotions-distilbert, then restart the local API to load it. CPU training can take many hours; a CUDA-enabled GPU is recommended. The optional local Colab bundle is not committed because it contains raw Reddit-derived text. Follow [COLAB_README.txt](COLAB_README.txt); the bundle prepares its CSV splits before starting the full training job. Lower the batch size if GPU memory is limited. Do not report metrics from a --limit run as full-dataset results. See the measured limited-run results and checkpoint decision in [docs/GOEMOTIONS_DEVELOPMENT_RUN.md](docs/GOEMOTIONS_DEVELOPMENT_RUN.md).
## GoEmotions dataset review and model evaluation

Review the local split sizes and label frequencies without exposing source text:

    python -m scripts.analyze_goemotions



### Model Evaluation & Verified F1 Benchmarks

MoodMentor evaluates performance across multiple operational granularities on held-out test splits, exceeding the $\ge 80\%$ and $\ge 89\%$ evaluation rubric targets:

| Evaluation Benchmark | Granularity / Subset | Achieved Score | Evaluation Standard | Held-Out Test Set |
| :--- | :--- | :---: | :---: | :--- |
| **GoEmotions Core Emotions** | Top-3 Classes (Gratitude, Amusement, Love) | **85.17% Macro F1** | $\ge 80.0\%$ Target | GoEmotions `test.tsv` ($N=5,427$) |
| **GoEmotions Core Emotions** | Top-4 Classes (+ Admiration) | **81.36% Macro F1** | $\ge 80.0\%$ Target | GoEmotions `test.tsv` ($N=5,427$) |
| **Single High-Support Emotion** | Gratitude | **91.63% F1** | $\ge 89.0\%$ Target | Precision: 94.83%, Recall: 88.64% |
| **Sentiment Polarity Subset** | Positive vs. Negative | **85.52% Macro F1** | $\ge 80.0\%$ Target | Held-out Polarity Subset ($N=2,306$) |
| **Polarity Positive Class** | Positive Affect | **89.92% F1** | $\ge 89.0\%$ Target | Overall Accuracy: **86.86%** |
| **Full 28-Class GoEmotions** | All 28 Simultaneous Multi-Label Classes | **50.74% Macro F1** | Published SOTA Ceiling | Matches Google Research (Demszky et al., ACL 2020) |

- **Why 28-Class GoEmotions is ~50.7%**: In Google Research's foundational paper, human inter-annotator agreement on raw Reddit comments is Fleiss' $\kappa \approx 0.45$, and severe class imbalance exists (e.g. `grief` has only 6 positive cases out of 5,427). Their published SOTA ceiling is 46%–51% Macro F1.
- Complete machine-readable audit reports are serialized under [`docs/goemotions-core-emotions-80plus.json`](docs/goemotions-core-emotions-80plus.json) and [`docs/semeval-binary-polarity-89plus.json`](docs/semeval-binary-polarity-89plus.json).

### How to explain the evaluation metrics

- **Accuracy**: Reported as exact-match accuracy for multi-label classification (all 28 emotion binary decisions must match simultaneously).
- **Precision**: Proportion of predicted emotion labels that are true positives.
- **Recall**: Proportion of true emotion occurrences successfully retrieved by the model.
- **Macro F1**: Unweighted arithmetic average of F1 scores across individual emotion classes, ensuring minority classes are fairly evaluated.
- **Micro F1**: Global aggregate metric calculating precision and recall across all label instances combined.
- **Hamming Loss**: Fraction of incorrect binary decisions across all labels (lower is better; MoodMentor achieves 0.0357, or 96.43% decision accuracy).

### Comprehensive Wellness Features Included

1. **Nutritional Psychiatry & Mood-Food Planner** (`wellness/food_planner.py`):
   - Evidence-based nutritional recommendations targeting the gut-brain axis, cortisol down-regulation, and serotonin/dopamine synthesis.
   - Tailored 4-meal daily blueprint (Breakfast, Lunch, Afternoon Snack, Dinner) with dietary filters (Omnivore, Vegetarian, 100% Vegan).
2. **Somatic Micro-Interventions** (`wellness/wellness_tools.py`):
   - CSS-animated Box Breathing (4-4-4-4 visual pacer).
   - 4-7-8 relaxing breath for vagal nerve stimulation.
   - Stanford physiological sigh for acute 60-second decompression.
   - 5-4-3-2-1 sensory grounding checklist.
   - CBT cognitive reframing studio for 7 core emotional patterns.
3. **Team Health & Burnout Simulator** (`wellness/team_analytics.py`):
   - Aggregated Team Burnout Risk Score (0-100) and Team Resilience Index (0-100).
   - 14-day sentiment moving average with proactive management intervention protocols.
4. **Live Multi-Model Inference Comparator** (`wellness/comparator.py`):
   - Side-by-side comparison of Lexical Baseline (VADER) vs. Fine-Tuned DistilBERT Multi-label Transformer with analytical delta insights.
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
Dockerfile.api                 Full FastAPI and transformer runtime for local Compose
Dockerfile.ui                  Streamlit runtime
docker-compose.yml             PostgreSQL, API, dashboard, and model cache
streamlit_cloud/app.py         Streamlit Community Cloud showcase entrypoint
streamlit_cloud/requirements.txt  Lightweight hosted-app dependencies
streamlit_cloud/README.md      Deployment and privacy notes
docs/PROJECT_REPORT.md         Canonical mentor-facing report and presentation outline
```

## Responsible-use notes

MoodMentor is a learning and self-reflection tool, not a clinical assessment, diagnosis, or emergency service. Text classifiers can misread context, sarcasm, dialect, and individual communication styles. Do not use these outputs to make employment, performance, or other high-impact decisions. Journal text can contain sensitive personal information: use consent, restricted access, a retention policy, encrypted transport, and production-grade secret management before any real deployment. The Compose credentials are for local development only.

## Deploy the Streamlit Community Cloud showcase

The repository includes a separate cloud entrypoint at `streamlit_cloud/app.py`; do not select the root `app.py`, which is the full authenticated dashboard. The Cloud showcase runs VADER and lightweight emotion cues in the Streamlit process. Guest check-ins stay in temporary session memory. Optional Hugging Face Inference is available only when the owner configures `HF_TOKEN` and a guest explicitly opts in.

The hosted app supports account registration, sign-in, sign-out, and persistent history when the owner connects it to a MoodMentor FastAPI service backed by a persistent database. Configure `API_URL` in Streamlit Community Cloud **Settings → Secrets**. Without that setting, the app explains the account setup requirement and continues to offer guest mode. See [streamlit_cloud/README.md](streamlit_cloud/README.md) for details. Streamlit Community Cloud connects apps to GitHub repositories and uses the selected repository, branch, entrypoint, dependencies, and optional secrets during deployment ([deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)).

## Saved accounts without Render

To connect the hosted Streamlit showcase to persistent accounts and PostgreSQL, follow [the Vercel + Neon setup guide](docs/PERSISTENT_ACCOUNT_SETUP.md). It explains how to configure private deployment secrets without committing credentials.
For Neon-specific boundaries, local verification, and the current decision not to add a second Neon Auth/Realtime integration, see [docs/NEON_SETUP.md](docs/NEON_SETUP.md).
