# AI-Based Employee Wellness Management Platform

Milestone 1 provides a reusable Python API and CLI for employee feedback ingestion,
NLTK preprocessing, and VADER sentiment analysis. Future recommendations and web/API
milestones are intentionally outside this implementation.

## Setup

Python 3.10+ is required. Create and activate a virtual environment, then install the
package and development tools:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m wellness.cli --download-nltk-data
```

To use the frontend, install the UI extra as well:

```bash
python -m pip install -e ".[dev,ui]"
streamlit run app.py
```

The Streamlit page follows the application flow **Landing page -> Login or
Registration -> Connect Database -> Input -> Sentiment Analysis**. Registration and
login use a local SQLite database (`wellness.db`) with salted PBKDF2 password hashes.
The database is a lightweight milestone boundary and can later be replaced by a
managed database without changing the sentiment pipeline. The authenticated analyzer
supports direct text, TXT upload, CSV upload with an optional text-column override,
human-readable rendering, score cards, and JSON report download. Invalid or empty
inputs are shown as explicit errors rather than producing an empty report.

The download command installs `stopwords`, `wordnet`, `punkt`, `punkt_tab`,
`averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng`, and `vader_lexicon`.
The API reports a clear setup error if any required data is missing.

## Usage

Direct text:

```bash
wellness-analyze --text "I feel supported and motivated at work."
wellness-analyze --text "The workload is exhausting." --json
```

Files:

```bash
wellness-analyze --file feedback.txt
wellness-analyze --file feedback.csv --text-column feedback --json
```

TXT files must be UTF-8 and non-empty. CSV files must have a header and a non-empty
text column. `text`, `feedback`, `comment`, `message`, and `content` are detected
automatically; use `--text-column` for another name. Non-empty values from all rows
are analyzed together. Missing, malformed, unsupported, or empty inputs return an
explicit error and non-zero exit status.

The report includes input summary, original/cleaned/processed text, final tokens,
negative/neutral/positive/compound VADER scores, a positive/negative/neutral label,
and useful counts. The Python API returns an `AnalysisReport`:

```python
from wellness import analyze_text, analyze_file

report = analyze_text("My manager listens and supports the team.")
print(report.sentiment_label.value)
print(report.to_dict())
```

## Tests and checks

```bash
pytest
ruff check .
```

## Deployment

The repository includes a production-ready `Dockerfile` and `render.yaml`. To run
the same container locally:

```bash
docker build -t employee-wellness-platform .
docker run --rm -p 8501:8501 employee-wellness-platform
```

Open <http://localhost:8501>. The image installs all NLTK datasets during the build
and exposes Streamlit's `/_stcore/health` endpoint for platform health checks.

For Render, create a new **Blueprint** from this repository; Render will detect
`render.yaml`, build the Docker image, and publish the service. The free plan has
ephemeral storage, so the local `wellness.db` account database is suitable for
demonstration only. Use a managed persistent database before production use.
For any Docker host, set the `PORT` environment variable to the port supplied by
the host.
