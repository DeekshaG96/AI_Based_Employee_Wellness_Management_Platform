# MoodMentor: Project Report and Mentor Presentation

## Abstract

MoodMentor is a web application for exploring how natural-language processing can summarize the emotional tone of short written reflections and provide general well-being suggestions. It combines text preprocessing, VADER sentiment scoring, a 28-label multi-label GoEmotions DistilBERT classifier, a recommendation ruleset, authenticated REST endpoints, PostgreSQL persistence, and an interactive Streamlit dashboard. The result is intended for learning and personal reflection; it is not a mental-health assessment or an employee evaluation system.

## Problem statement

Written feedback and personal reflections contain qualitative signals that are difficult to review consistently at scale. A user may also want a simple way to reflect on sentiment over time. The project addresses this as a prototype by making a transparent text-analysis workflow available through a dashboard and by retaining each user's own history for review.

The prototype does not claim to infer a person's true emotional state. Text can be ambiguous, context-dependent, sarcastic, or culturally specific, and automated labels can be wrong.

## Objectives

1. Accept free text and small TXT/CSV uploads.
2. Preserve punctuation, casing, and negation for VADER while creating a normalized token stream for inspection.
3. Return sentiment scores and independent fine-grained emotion signals, mapping the strongest signal to a broad category for recommendations; use a clearly labeled fallback when the model is unavailable.
4. Provide practical, non-clinical well-being actions based on simple rules.
5. Store account-scoped analysis records and recommendations in PostgreSQL.
6. Present current results and personal history through a usable dashboard.

## Solution architecture

```text
Browser
  │
  ▼
Streamlit UI ── authenticated HTTP / bearer token ──► FastAPI
                                                        │
                           ┌────────────────────────────┴─────────────┐
                           ▼                                          ▼
                 NLTK + VADER + Hugging Face                 SQLAlchemy 2
                                                                      │
                                                                      ▼
                                                                  PostgreSQL
```

The UI handles presentation and sends requests to FastAPI. The API validates inputs, checks session ownership, runs the analysis pipeline, and persists results. SQLAlchemy maps the users, sessions, mood logs, and recommendations. The API is the only application layer that writes to the database.

## Implementation modules

- `wellness/preprocessing.py`: URL/HTML/email cleanup, contraction expansion, negation-preserving preprocessing, and spaCy tokenization/lemmatization with an NLTK POS/WordNet fallback.
- `wellness/pipeline.py`: VADER scores and polarity labels; lazy inference with the pretrained GoEmotions DistilBERT model; broad-family mapping for recommendation rules; keyword fallback for offline and low-resource hosting. When no supported cue matches, the result is marked unclassified.
- `wellness/report.py`: Typed analysis result and JSON conversion.
- `wellness_api/main.py`: health, authentication, analysis, file-upload, history, and delete routes.
- `wellness_api/schemas.py`: API payload and response validation.
- `wellness_api/security.py`: salted PBKDF2 password hashing, bearer-session issuance, digest-only token storage, expiry, and authentication.
- `wellness_api/models.py`: PostgreSQL entities and relationships.
- `wellness/api_client.py`: Streamlit's HTTP client and API-to-dashboard result conversion.
- `app.py`: sign-in, direct analysis, file ingestion, charts, history, exports, and detail view.
- `scripts/migrate_sqlite_to_postgres.py`: guarded import of legacy SQLite accounts and analysis logs.

## Data model

- **users** owns accounts and relates each account to its sessions and mood logs.
- **auth_sessions** stores a SHA-256 token digest and expiration time. Raw bearer tokens are returned only at sign-in.
- **mood_logs** stores source metadata, original and cleaned text, tokens, sentiment values, emotion scores, and timestamp.
- **recommendations** stores the guidance generated for a mood log, including category, urgency, actions, and crisis flag.

Each mood log belongs to a user. History queries require a valid bearer token and reject a requested user ID that does not match the authenticated account. Delete requests likewise constrain both record and owner.

## Analysis workflow

1. The user submits text or uploads a TXT/CSV file in Streamlit.
2. The dashboard sends it to an authenticated FastAPI endpoint.
3. Input validation rejects empty or oversized content. CSV processing detects a likely feedback column or uses the supplied column name.
4. The NLP pipeline keeps a sentiment-safe text form, preprocesses a token form, and computes VADER's negative, neutral, positive, and compound values.
5. The pretrained Hugging Face model produces 28 independent GoEmotions scores. MoodMentor keeps these fine-grained signals and maps the strongest one to a broad category for recommendations. If the model is unavailable, a deterministic keyword fallback is used and treated as a heuristic. When no supported cue matches, the system reports an unclassified result rather than fabricating an even distribution. Each saved result records which estimator produced it.
6. A small ruleset selects a general well-being recommendation.
7. The API commits the mood log and its recommendation. The response returns the result for charts and user review.
8. History and exports use the authenticated user's stored records.

## Dataset and model evaluation

The archived GoEmotions simplified agree-2 dataset has 54,263 examples (43,410 train, 5,426 development, and 5,427 test) and 28 labels. The profile found no malformed records, empty text, repeated IDs within a split, or invalid label IDs. The archive contains Reddit-derived text, is not committed to Git, and is not representative of employee journals.

The GoEmotions DistilBERT checkpoint was evaluated on all 5,427 held-out test entries with independent sigmoid scores and a 0.5 decision threshold. It reached micro precision 0.8618, micro recall 0.3321, micro F1 0.4795, macro F1 0.2980, and exact-match accuracy 0.3243. This conservative threshold misses many labels, especially rare classes. These are benchmark results for the public dataset, not an accuracy claim for workplace writing. Threshold calibration should use the development split before any consequential use. Detailed label metrics and runtime versions are saved in `docs/goemotions-distilbert-evaluation.json`; see `docs/GOEMOTIONS_DATASET_REVIEW.md` for the privacy-preserving profile and limitations.

## Local deployment

The intended demonstration runs PostgreSQL, FastAPI, and Streamlit with Docker Compose. The API listens on host port 8000 and Streamlit on 8501; PostgreSQL is bound to local host port 5433. API documentation is served at `/docs`. See the repository README for commands, local development, dependency installation, and SQLite migration instructions.

## Verification plan

The project includes tests for preprocessing, VADER behavior, legacy SQLite utilities, authentication, API validation, account-scoped history, analysis persistence, and deletion. API tests replace the large model call with a predictable report so that endpoint behavior can be checked without downloading model weights. Before a mentor demo, run the test suite and lint check, then manually verify:

1. API health reports a connected database.
2. Account sign-in and registration succeed.
3. A journal entry returns sentiment, emotion, recommendations, a downloadable report, and an updated mood trend.
4. TXT and CSV uploads persist their results.
5. History charts and exports update after analysis.
6. One user's token cannot read or delete another user's logs.
7. The application explains when it cannot reach the API.

## Limitations and responsible use

- The emotion model is not validated as a diagnostic or clinical tool.
- Rule-based recommendations are general prompts, not therapy or medical advice.
- A heuristic fallback does not have calibrated probabilities.
- The public GoEmotions labels come from Reddit comments, not workplace writing; language variety, sarcasm, mixed emotions, and missing context can affect predictions. The held-out result also shows low recall on rare emotion classes.
- Stored journal text is sensitive. A real deployment needs user consent, deletion and retention rules, access auditing, HTTPS, managed secrets, database backups, and a privacy review.
- Do not use outputs for hiring, performance review, discipline, or other high-impact employment decisions.

## Suggested presentation (7 slides)

1. **Title and motivation** — MoodMentor and the challenge of reflecting on unstructured feedback.
2. **Problem and scope** — the prototype's user need, intended audience, and boundaries.
3. **Architecture** — Streamlit, FastAPI, NLP pipeline, SQLAlchemy, and PostgreSQL.
4. **Analysis pipeline** — preprocessing, VADER, the multi-label DistilBERT GoEmotions model, broad-category recommendations, and fallback; explain that model scores are independent.
5. **Data and API design** — account-scoped history, relational schema, and key endpoints.
6. **Live walkthrough and verification** — sign in, analyze, inspect charts, upload a CSV, and show ownership checks.
7. **Limitations and next steps** — evaluate on consented labeled data, improve calibration and accessibility, and add production privacy/security controls.

## Short mentor summary

> MoodMentor is a full-stack NLP prototype that processes short written reflections with NLTK preprocessing, VADER sentiment analysis, and a pretrained multi-label DistilBERT model trained on GoEmotions. A FastAPI service validates requests, authenticates users, stores analysis history and recommendations in PostgreSQL, and powers a Streamlit dashboard with interactive trends. The outputs are informational and intended for personal reflection, not clinical or employment decisions.

## Hosted review deployment

The Render Blueprint defines a Streamlit web service, a FastAPI web service, and a PostgreSQL database. The API receives its database connection string through a Render service reference, while the dashboard receives the API's public URL dynamically. Both containers listen on Render's assigned port. No passwords or database URLs are committed to the repository.

The free hosted review uses a lean API image without PyTorch or Transformers. In that environment, sentiment uses VADER and emotion estimates use the documented lexical heuristic. The full Hugging Face classifier is available through the local Docker Compose setup. The public free web services can sleep after inactivity, and Render's free PostgreSQL databases expire 30 days after creation. The hosted version is suitable for review and demonstration only; it is not a durable or production deployment.
