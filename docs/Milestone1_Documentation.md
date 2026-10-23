# Infosys Springboard Virtual Internship

## Project Name: AI-Powered Emotional Tone Analyzer and Well-Being Recommendation System (MoodMentor)

**Student Name:** Deeksha Ganesh  
**GitHub Account:** [DeekshaG96](https://github.com/DeekshaG96)  
**Repository:** [AI_Based_Employee_Wellness_Management_Platform](https://github.com/DeekshaG96/AI_Based_Employee_Wellness_Management_Platform)  
**Date:** 16 October 2026  
**Milestone 1:** Text Ingestion, Preprocessing & Baseline Sentiment Analysis  

---

## 1. Introduction

The **AI-Powered Emotional Tone Analyzer and Well-Being Recommendation System (MoodMentor)** is an intelligent web-based platform designed to analyze emotional signals and sentiment from workplace written communication—including employee journal entries, 1-on-1 notes, pulse survey feedback, and chat messages.

Milestone 1 establishes the core Natural Language Processing (NLP) foundation of the MoodMentor architecture:
- Robust multi-source text ingestion (direct text input, UTF-8 `.txt` documents, and multi-row `.csv` files).
- A rigorous, documented NLP preprocessing pipeline (cleaning, NLTK tokenization, stop-word elimination, POS tagging, and WordNet lemmatization).
- Baseline sentiment intensity classification powered by NLTK VADER.
- Secure, relational persistence via SQLite (`users` with salted PBKDF2 password derivation and `mood_entries` storing full sentiment metrics).
- A modern, dark-themed three-tab Streamlit dashboard delivering real-time sentiment telemetry, interactive keyword inspection, and historical analytics.

### Key Highlights
- **Multi-Source Ingestion:** Supports direct text input, UTF-8 `.txt` file uploads, and `.csv` files with intelligent auto-detection of feedback columns (`feedback`, `text`, `comment`, `message`, `content`).
- **Sequential NLP Preprocessing:** Follows the exact documented sequence: Case Normalization & Cleaning &rarr; Tokenization &rarr; Stop-word Filtering &rarr; Part-of-Speech Tagging &rarr; WordNet Lemmatization.
- **VADER Sentiment Engine:** Computes compound polarity score ($-1.000$ to $+1.000$) alongside positive, neutral, and negative distributions.
- **Relational Data Persistence:** Automatically logs every analysis into an SQLite `mood_entries` table with user attribution and timestamps.
- **Three-Tab Streamlit Interface:** Features dedicated tabs for Direct Feedback Analysis, Batch File Ingestion, and Stored Records & Analytics.
- **Comprehensive Quality Assurance:** 15 automated `pytest` test cases covering ingestion validation, preprocessing stages, VADER scoring, and database persistence, with 100% clean Ruff linting.

---

## 2. Technology Stack

- **Core Programming Language:** Python 3.10+ (tested on Python 3.14)
- **Frontend / Web UI:** Streamlit with custom dark-themed CSS and glassmorphism cards
- **Natural Language Processing:** NLTK (`punkt`, `stopwords`, `averaged_perceptron_tagger`, `wordnet`)
- **Sentiment Analysis Engine:** NLTK VADER (`SentimentIntensityAnalyzer`)
- **Data Persistence:** SQLite3 (local relational storage with foreign keys)
- **Security & Authentication:** Salted PBKDF2-HMAC-SHA256 password hashing (120,000 iterations)
- **Data Manipulation & Tables:** Pandas
- **Testing & Code Quality:** Pytest, Ruff linter
- **Packaging & Containerization:** `pyproject.toml`, Dockerfile, Render Blueprint (`render.yaml`)

---

## 3. Project Structure

The project follows a clean, modular architecture separating ingestion, text preprocessing, sentiment analysis, persistence, and presentation:

```text
AI_Based_Employee_Wellness_Management_Platform/
├── app.py                         # Streamlit frontend & 3-tab UI
├── pyproject.toml                 # Packaging metadata & dependency definitions
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
└── docs/
    ├── Milestone1_Documentation.md# Comprehensive project documentation
    ├── Milestone1_Documentation.pdf# Official Infosys Springboard PDF report
    └── screenshots/               # Output execution screenshots
        ├── fig1_direct_feedback_nlp.png
        ├── fig2_file_ingestion.png
        ├── fig3_stored_records_analytics.png
        └── fig4_landing_auth.png
```

---

## 4. Objectives Achieved

### 1. Flexible Input & Ingestion
- Real-time direct text input with instant character and word counting.
- Drag-and-drop file upload supporting UTF-8 `.txt` and structured `.csv` files.
- Automatic column detection for CSV headers (`feedback`, `text`, `comment`, etc.) with optional manual override.

### 2. Multi-Stage Preprocessing Pipeline
- **Cleaning:** Strips URLs, punctuation, digits, and normalizes excess whitespace.
- **Tokenization:** Breaks sanitized text into linguistic word tokens using NLTK `punkt`.
- **Stop-word Removal:** Filters out high-frequency English stop words while retaining core meaning.
- **Lemmatization:** Maps words to their morphological roots using WordNet and POS context.

### 3. VADER Sentiment Scoring & Classification
- Evaluates polarity across four dimensions: Positive (`pos`), Neutral (`neu`), Negative (`neg`), and Compound (`compound`).
- Standard calibrated thresholds:
  - $\text{Compound} \ge +0.05 \implies \textbf{Positive}$
  - $\text{Compound} \le -0.05 \implies \textbf{Negative}$
  - Otherwise $\implies \textbf{Neutral}$

### 4. Data Persistence & User Session Management
- Multi-user authentication with salted PBKDF2 password derivation.
- Persistent logging of every analysis into SQLite `mood_entries` table.
- Searchable and filterable history view with KPI aggregation metrics.

---

## 5. Code Implementation

### Module 1: Ingestion Layer (`wellness/ingestion.py`)
Validates input strings and parses `.txt` and `.csv` files safely:

```python
def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise InputValidationError("Text input must be a string.")
    if not text.strip():
        raise InputValidationError("Text input cannot be empty.")
    return text.strip()


def read_input_file(path: str | Path, text_column: str | None = None) -> tuple[str, str]:
    file_path = Path(path)
    if not file_path.exists():
        raise InputValidationError(f"Input file does not exist: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix not in {".txt", ".csv"}:
        raise InputValidationError("Supported input files are .txt and .csv.")
    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8-sig")
    else:
        text = _read_csv(file_path, text_column)
    return validate_text(text), str(file_path)
```

### Module 2: Text Preprocessing (`wellness/preprocessing.py`)
Implements the full 5-stage NLP cleaning, tokenization, filtering, and lemmatization pipeline:

```python
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", _NON_LETTER.sub(" ", _URL.sub(" ", text.lower()))).strip()


def preprocess(text: str) -> tuple[str, tuple[str, ...]]:
    validate_nltk_resources()
    cleaned = clean_text(text)
    raw_tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words("english"))
    filtered = [token for token in raw_tokens if token.isalpha() and token not in stop_words]
    tagged = pos_tag(filtered)
    lemmatizer = WordNetLemmatizer()
    lemmas = tuple(lemmatizer.lemmatize(token, _wordnet_pos(tag)) for token, tag in tagged)
    return cleaned, lemmas
```

### Module 3: Sentiment Pipeline (`wellness/pipeline.py`)
Coordinates preprocessing, executes VADER scoring, and compiles typed analysis reports:

```python
def analyze_text(text: str, source_name: str = "direct text") -> AnalysisReport:
    original = validate_text(text)
    validate_nltk_resources()
    cleaned, tokens = preprocess(original)
    scores = SentimentIntensityAnalyzer().polarity_scores(original)
    compound = scores["compound"]
    label = (
        SentimentLabel.POSITIVE
        if compound >= 0.05
        else SentimentLabel.NEGATIVE
        if compound <= -0.05
        else SentimentLabel.NEUTRAL
    )
    return AnalysisReport(
        input_summary=InputSummary(
            "text", source_name, len(original), len(original.splitlines()), len(tokens)
        ),
        original_text=original,
        cleaned_text=cleaned,
        processed_text=" ".join(tokens),
        tokens=tokens,
        sentiment_scores=SentimentScores(scores["neg"], scores["neu"], scores["pos"], compound),
        sentiment_label=label,
        counts={"characters": len(original), "tokens": len(tokens)},
    )
```

### Module 4: Database & Persistence Layer (`wellness/storage.py`)
Manages SQLite schema creation, PBKDF2 authentication, and persistent mood entry logging:

```python
def save_mood_entry(
    connection: sqlite3.Connection,
    user_id: int,
    source_type: str,
    source_name: str,
    original_text: str,
    cleaned_text: str,
    tokens: tuple[str, ...] | list[str],
    sentiment_label: str,
    neg_score: float,
    neu_score: float,
    pos_score: float,
    compound_score: float,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO mood_entries (
            user_id, source_type, source_name, original_text, cleaned_text,
            tokens, sentiment_label, neg_score, neu_score, pos_score, compound_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            source_type,
            source_name,
            original_text,
            cleaned_text,
            ", ".join(tokens),
            sentiment_label,
            neg_score,
            neu_score,
            pos_score,
            compound_score,
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)
```

---

## 6. System Flow

```text
[User Feedback Input (Direct / TXT / CSV)]
                    │
                    ▼
   [Validation Layer (wellness/ingestion.py)]
                    │
                    ▼
[NLP Preprocessing Pipeline (wellness/preprocessing.py)]
  ├── 1. Regex Cleaning (strip URLs, digits, punctuation)
  ├── 2. Lowercase Normalization
  ├── 3. NLTK Punkt Word Tokenization
  ├── 4. Stop-word Filtering (NLTK English corpus)
  └── 5. WordNet Lemmatization + POS Tagging
                    │
                    ▼
[VADER Sentiment Engine (wellness/pipeline.py)]
  ├── Compute Positive, Neutral, Negative, Compound Scores
  └── Classify Polarity Label (Positive / Neutral / Negative)
                    │
                    ▼
[Persistence Layer (wellness/storage.py)]
  └── Save entry to SQLite `mood_entries` table
                    │
                    ▼
[Streamlit Presentation (app.py)]
  ├── Tab 1: Direct Feedback Analysis & NLP Step-by-Step Breakdown
  ├── Tab 2: Batch File Ingestion (.txt / .csv)
  └── Tab 3: Stored Records & Analytics Dashboard (KPIs, Filtering, Search)
```

---

## 7. How to Run

### Prerequisites
- Python 3.10 or higher
- Git installed
- Google Chrome or any modern browser

### Setup Steps

```bash
# Step 1: Clone the repository
git clone https://github.com/DeekshaG96/AI_Based_Employee_Wellness_Management_Platform.git

# Step 2: Navigate to project directory
cd AI_Based_Employee_Wellness_Management_Platform

# Step 3: Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Step 4: Install package and dependencies
pip install -e ".[dev,ui]"

# Step 5: Download required NLTK datasets
python -m wellness.cli --download-nltk-data

# Step 6: Run automated tests
pytest

# Step 7: Launch the Streamlit application
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 8. Output Screenshots

### Fig 1: Landing Page & User Authentication
The welcoming portal allows employees and team leads to authenticate with secure salted PBKDF2 credentials and connects the persistent SQLite database.

![Landing Page](screenshots/fig4_landing_auth.png)

### Fig 2: Direct Feedback Sentiment Analysis & NLP Pipeline
Demonstrates real-time analysis of employee feedback. The system displays a high-visibility sentiment banner, compound polarity (+0.857), 4-metric score distribution, and step-by-step NLTK preprocessing stages with lemmatized keyword chips.

![Direct Feedback Analysis](screenshots/fig1_direct_feedback_nlp.png)

### Fig 3: File Ingestion (.txt and .csv)
Demonstrates uploading external text files and tabular pulse survey CSV files with auto-detection of text columns and immediate feedback analysis.

![File Ingestion](screenshots/fig2_file_ingestion.png)

### Fig 4: Stored Records & Analytics Dashboard
Displays the persistent SQLite `mood_entries` table with aggregate KPI cards (Total Records, % Positive, % Neutral, % Negative, Average Compound), sentiment filtering, and keyword search.

![Stored Records Dashboard](screenshots/fig3_stored_records_analytics.png)

---

## 9. Verification & Test Results

```text
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\ganch\OneDrive\Desktop\infosys\AI_Based_Employee_Wellness_Management_Platform
configfile: pyproject.toml
testpaths: tests
collected 15 items

tests\test_pipeline.py ...........                                       [ 73%]
tests\test_storage.py ....                                               [100%]

============================= 15 passed in 10.08s =============================

Ruff Code Quality:
All checks passed!
```

---

## 10. Known Limitations

- **Lexicon-Based Sentiment:** VADER relies on curated lexical heuristics; it may occasionally miss deep workplace sarcasm or context-dependent jargon. (This will be addressed by BERT/DistilBERT fine-tuning in Milestone 2).
- **Local SQLite File Storage:** SQLite operates on local filesystem storage (`wellness.db`). While ideal for demonstration and edge deployment, enterprise scale will transition to PostgreSQL.
- **Language Scope:** Preprocessing currently targets English text through NLTK stop-words and WordNet.

---

## 11. Future Scope

- **Milestone 2: Deep Emotion Classification:**
  - Transition from baseline sentiment to multi-class emotion classification (Joy, Sadness, Anger, Fear, Surprise, Exhaustion).
  - Fine-tune transformer architectures (BERT / DistilBERT) for nuanced workplace language understanding.
- **Milestone 3: AI-Driven Recommendation Engine & REST APIs:**
  - Build wellness recommendations tailored to detected emotional trends (e.g., burnout mitigation, mindfulness prompts).
  - Expose FastAPI REST endpoints for integration with Slack, Teams, and HRIS platforms.
- **Milestone 4: Finalization & Production Packaging:**
  - Full CLI tooling, integration testing suite, enhanced multi-tenant dashboards, and cloud deployment.

---

## 12. Conclusion

In Milestone 1, the **MoodMentor Employee Wellness Platform** has successfully established a complete, robust, and validated foundation for AI-driven emotional tone analysis. The implementation satisfies every Milestone 1 requirement:
1. Validated text and file ingestion (`.txt`, `.csv`).
2. Documented 5-stage NLP preprocessing pipeline.
3. VADER sentiment engine with compound polarity scoring.
4. Persistent SQLite storage with user authentication.
5. A modern, responsive 3-tab Streamlit dashboard.
6. 100% test suite pass rate and clean code quality.

This sets a reliable foundation for advancing to Milestone 2 (Deep Emotion Classification with Transformers).
