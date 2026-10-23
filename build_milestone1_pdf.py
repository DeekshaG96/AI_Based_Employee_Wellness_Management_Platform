"""Script to generate professional Milestone 1 Documentation PDF matching Infosys Springboard benchmark standard."""

import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress header and footer on cover page
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header
        self.drawString(
            54, 755, "Infosys Springboard Virtual Internship — MoodMentor (Milestone 1)"
        )
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 747, 558, 747)

        # Footer
        self.drawString(54, 38, "AI-Powered Emotional Tone Analyzer & Employee Wellness Platform")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_str)
        self.line(54, 48, 558, 48)
        self.restoreState()


def build_pdf(filename="docs/Milestone1_Documentation.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        alignment=1,
        textColor=colors.HexColor("#0F172A"),
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        alignment=1,
        textColor=colors.HexColor("#2563EB"),
    )

    meta_style = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=18,
        alignment=1,
        textColor=colors.HexColor("#334155"),
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=14,
        spaceAfter=2.5,
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#0F172A"),
    )

    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#475569"),
        spaceBefore=4,
        spaceAfter=8,
    )

    story = []

    # ==================== PAGE 1: COVER PAGE ====================
    story.append(Spacer(1, 140))
    story.append(Paragraph("Infosys Springboard Virtual Internship", subtitle_style))
    story.append(Spacer(1, 25))
    story.append(
        Paragraph(
            "AI-Powered Emotional Tone Analyzer and<br/>Well-Being Recommendation System",
            title_style,
        )
    )
    story.append(Spacer(1, 15))
    story.append(
        Paragraph("<b>MoodMentor:</b> Employee Wellness Management Platform", subtitle_style)
    )
    story.append(Spacer(1, 40))

    story.append(Paragraph("<b>Student Name:</b> Deeksha Ganesh", meta_style))
    story.append(Paragraph("<b>GitHub:</b> github.com/DeekshaG96", meta_style))
    story.append(Paragraph("<b>Date:</b> 16 October 2026", meta_style))
    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            "<b>Milestone 1 — Text Ingestion, Preprocessing & Baseline Sentiment Analysis</b>",
            ParagraphStyle(
                "CoverMilestone",
                parent=meta_style,
                fontName="Helvetica-Bold",
                fontSize=12,
                leading=16,
                textColor=colors.HexColor("#0369A1"),
            ),
        )
    )
    story.append(PageBreak())

    # ==================== PAGE 2: INTRODUCTION & TECH STACK ====================
    story.append(Paragraph("Introduction", h1_style))
    story.append(
        Paragraph(
            "The <b>AI-Powered Emotional Tone Analyzer and Well-Being Recommendation System (MoodMentor)</b> "
            "is an intelligent platform designed to detect and understand workplace sentiment from unstructured text "
            "such as employee journal entries, 1-on-1 notes, pulse survey responses, and chat messages. "
            "Milestone 1 establishes the foundational data and NLP processing layers: multi-source input ingestion, "
            "an ordered natural language preprocessing pipeline, VADER baseline sentiment classification, "
            "relational SQLite persistence, and an interactive Streamlit application.",
            body_style,
        )
    )

    story.append(Paragraph("Key Highlights:", h2_style))
    story.append(
        Paragraph(
            "• <b>Multi-Source Ingestion:</b> Supports direct text, UTF-8 <code>.txt</code> notes, and multi-row <code>.csv</code> survey data with automatic text column detection.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Ordered NLP Preprocessing:</b> Implements full regex cleaning, NLTK tokenization, stop-word elimination, POS tagging, and WordNet lemmatization.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Baseline Sentiment Engine:</b> Computes calibrated VADER compound polarity (-1.0 to +1.0) alongside positive, neutral, and negative distributions.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Relational Persistence:</b> Stores user accounts with salted PBKDF2 hashing and logs parsed mood entries into an SQLite relational table.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Streamlit 3-Tab Interface:</b> Features dedicated workflows for Direct Text Analysis, File Ingestion, and Stored Records & Analytics.",
            bullet_style,
        )
    )

    story.append(Paragraph("Tech Stack", h1_style))
    tech_data = [
        [
            Paragraph("<b>Component</b>", body_style),
            Paragraph("<b>Technology</b>", body_style),
            Paragraph("<b>Role / Purpose in Milestone 1</b>", body_style),
        ],
        [
            Paragraph("Frontend UI", body_style),
            Paragraph("Streamlit 1.65+", body_style),
            Paragraph(
                "Interactive 3-tab web dashboard with dark glassmorphism styling", body_style
            ),
        ],
        [
            Paragraph("Core Backend", body_style),
            Paragraph("Python 3.10+ (3.14)", body_style),
            Paragraph(
                "Application logic, pipeline orchestration, CLI, and typed data models", body_style
            ),
        ],
        [
            Paragraph("NLP Preprocessing", body_style),
            Paragraph("NLTK & WordNet", body_style),
            Paragraph(
                "Regex sanitization, Punkt tokenization, stop-words, and lemmatization", body_style
            ),
        ],
        [
            Paragraph("Sentiment Engine", body_style),
            Paragraph("NLTK VADER", body_style),
            Paragraph(
                "Lexicon-based sentiment intensity and compound polarity scoring", body_style
            ),
        ],
        [
            Paragraph("Database Layer", body_style),
            Paragraph("SQLite3", body_style),
            Paragraph(
                "Persistent storage for <code>users</code> and <code>mood_entries</code> tables",
                body_style,
            ),
        ],
        [
            Paragraph("Data Handling", body_style),
            Paragraph("Pandas", body_style),
            Paragraph(
                "Tabular preview for CSV uploads and interactive history table rendering",
                body_style,
            ),
        ],
        [
            Paragraph("Quality & Testing", body_style),
            Paragraph("Pytest & Ruff", body_style),
            Paragraph(
                "15 automated unit and integration tests with strict code style checks", body_style
            ),
        ],
    ]
    t = Table(tech_data, colWidths=[100, 110, 294])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Project Structure", h1_style))
    proj_tree = """AI_Based_Employee_Wellness_Management_Platform/
├── app.py                         # Streamlit frontend & 3-tab interface
├── pyproject.toml                 # Packaging metadata & dependencies
├── Dockerfile                     # Container deployment definition
├── render.yaml                    # Render Cloud Blueprint configuration
├── wellness/
│   ├── ingestion.py               # Input validation & TXT/CSV parsing
│   ├── preprocessing.py           # Cleaning, tokenization, stop-words, lemmatization
│   ├── pipeline.py                # Public analysis API coordinating NLP & VADER
│   ├── report.py                  # Typed data models & JSON serialization
│   ├── storage.py                 # SQLite database setup, auth, & mood persistence
│   └── cli.py                     # Command-line entry point
├── tests/
│   ├── test_pipeline.py           # Ingestion, preprocessing, and VADER tests
│   └── test_storage.py            # SQLite setup, auth, and mood persistence tests
└── docs/                          # Documentation & execution screenshots"""
    story.append(Preformatted(proj_tree, code_style))
    story.append(PageBreak())

    # ==================== PAGE 3: OBJECTIVES & CODE MODULES 1-3 ====================
    story.append(Paragraph("Objectives Achieved", h1_style))
    story.append(
        Paragraph(
            "<b>1. Multi-Format Text Ingestion:</b> Direct text, UTF-8 <code>.txt</code> files, and multi-line <code>.csv</code> files with automatic text column detection and robust validation.<br/>"
            "<b>2. Sequential NLP Preprocessing:</b> Strict 5-stage order: lowercasing, regex cleaning, NLTK Punkt tokenization, stop-word filtering, and WordNet lemmatization.<br/>"
            "<b>3. Baseline Sentiment Analysis:</b> NLTK VADER compound polarity scoring (-1.0 to +1.0) and positive/neutral/negative intensity distribution.<br/>"
            "<b>4. Relational Data Persistence:</b> Salted PBKDF2 user authentication and persistent logging of mood entries into SQLite.<br/>"
            "<b>5. Interactive Web Frontend:</b> Three-tab Streamlit dashboard with preset samples, KPI metrics, sentiment filtering, search, and JSON/CSV exports.",
            body_style,
        )
    )

    story.append(Paragraph("Code Implementation", h1_style))
    story.append(Paragraph("Module 1: Text Ingestion (wellness/ingestion.py)", h2_style))
    c1 = """def read_input_file(path: str | Path, text_column: str | None = None) -> tuple[str, str]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in {".txt", ".csv"}:
        raise InputValidationError("Supported input files are .txt and .csv.")
    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8-sig")
    else:
        text = _read_csv(file_path, text_column)
    return validate_text(text), str(file_path)"""
    story.append(Preformatted(c1, code_style))
    story.append(Spacer(1, 3))

    story.append(Paragraph("Module 2: NLP Preprocessing (wellness/preprocessing.py)", h2_style))
    c2 = """def preprocess(text: str) -> tuple[str, tuple[str, ...]]:
    validate_nltk_resources()
    cleaned = clean_text(text)
    raw_tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words("english"))
    filtered = [token for token in raw_tokens if token.isalpha() and token not in stop_words]
    tagged = pos_tag(filtered)
    lemmatizer = WordNetLemmatizer()
    lemmas = tuple(lemmatizer.lemmatize(token, _wordnet_pos(tag)) for token, tag in tagged)
    return cleaned, lemmas"""
    story.append(Preformatted(c2, code_style))
    story.append(Spacer(1, 3))

    story.append(Paragraph("Module 3: Sentiment Pipeline (wellness/pipeline.py)", h2_style))
    c3 = """def analyze_text(text: str, source_name: str = "direct text") -> AnalysisReport:
    original = validate_text(text)
    cleaned, tokens = preprocess(original)
    scores = SentimentIntensityAnalyzer().polarity_scores(original)
    compound = scores["compound"]
    label = SentimentLabel.POSITIVE if compound >= 0.05 else (
        SentimentLabel.NEGATIVE if compound <= -0.05 else SentimentLabel.NEUTRAL
    )
    return AnalysisReport(input_summary=..., original_text=original, cleaned_text=cleaned,
                          processed_text=" ".join(tokens), tokens=tokens,
                          sentiment_scores=SentimentScores(**scores), sentiment_label=label)"""
    story.append(Preformatted(c3, code_style))
    story.append(PageBreak())

    # ==================== PAGE 4: DATABASE, SYSTEM FLOW & HOW TO RUN ====================
    story.append(Paragraph("Module 4: Database Layer (wellness/storage.py)", h2_style))
    c4 = """def save_mood_entry(connection, user_id, source_type, source_name, original_text,
                    cleaned_text, tokens, sentiment_label, neg_score, neu_score,
                    pos_score, compound_score) -> int:
    cursor = connection.execute(
        \"\"\"INSERT INTO mood_entries (user_id, source_type, source_name, original_text,
           cleaned_text, tokens, sentiment_label, neg_score, neu_score, pos_score, compound_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\"\"\",
        (user_id, source_type, source_name, original_text, cleaned_text,
         ", ".join(tokens), sentiment_label, neg_score, neu_score, pos_score, compound_score)
    )
    connection.commit()
    return int(cursor.lastrowid)"""
    story.append(Preformatted(c4, code_style))
    story.append(Spacer(1, 5))

    story.append(Paragraph("System Flow", h1_style))
    flow_steps = (
        "• <b>Step 1:</b> User accesses the Streamlit web portal and authenticates with salted PBKDF2 credentials.<br/>"
        "• <b>Step 2:</b> User inputs direct feedback or uploads a <code>.txt</code> or <code>.csv</code> file.<br/>"
        "• <b>Step 3:</b> <code>ingestion.py</code> validates encoding and non-empty content.<br/>"
        "• <b>Step 4:</b> <code>preprocessing.py</code> sanitizes raw text, tokenizes, filters stop-words, and applies WordNet lemmatization.<br/>"
        "• <b>Step 5:</b> <code>pipeline.py</code> calculates VADER sentiment scores and classifies compound polarity.<br/>"
        "• <b>Step 6:</b> <code>storage.py</code> persists the entry, scores, tokens, and timestamp to the SQLite database.<br/>"
        "• <b>Step 7:</b> <code>app.py</code> renders the sentiment badge, metrics, NLP breakdown, and updates the Stored Records tab."
    )
    story.append(Paragraph(flow_steps, body_style))
    story.append(Spacer(1, 5))

    story.append(Paragraph("How to Run", h1_style))
    run_cmds = """# Step 1: Clone the repository
git clone https://github.com/DeekshaG96/AI_Based_Employee_Wellness_Management_Platform.git
cd AI_Based_Employee_Wellness_Management_Platform

# Step 2: Create and activate virtual environment
python -m venv .venv
.venv\\Scripts\\activate          # On Windows (.venv/bin/activate on Linux/macOS)

# Step 3: Install dependencies & download NLTK data
pip install -e ".[dev,ui]"
python -m wellness.cli --download-nltk-data

# Step 4: Run automated test suite
pytest

# Step 5: Start the Streamlit application
streamlit run app.py
# Application opens automatically at http://localhost:8501"""
    story.append(Preformatted(run_cmds, code_style))
    story.append(PageBreak())

    # ==================== PAGE 5: SCREENSHOTS 1 & 2 ====================
    story.append(Paragraph("Output Screenshots", h1_style))

    if os.path.exists("docs/screenshots/fig4_landing_auth.png"):
        story.append(Paragraph("1. Landing Portal & Authentication:", h2_style))
        story.append(Image("docs/screenshots/fig4_landing_auth.png", width=490, height=240))
        story.append(
            Paragraph(
                "Fig 1: Welcome portal with user registration, salted PBKDF2 login, and SQLite connection",
                caption_style,
            )
        )
        story.append(Spacer(1, 5))

    if os.path.exists("docs/screenshots/fig1_direct_feedback_nlp.png"):
        story.append(
            Paragraph("2. Direct Feedback Analysis & NLP Preprocessing Pipeline:", h2_style)
        )
        story.append(Image("docs/screenshots/fig1_direct_feedback_nlp.png", width=490, height=240))
        story.append(
            Paragraph(
                "Fig 2: Real-time sentiment analysis showing positive polarity (+0.857), 4-metric score cards, and step-by-step NLTK preprocessing breakdown with lemmatized keywords",
                caption_style,
            )
        )

    story.append(PageBreak())

    # ==================== PAGE 6: SCREENSHOTS 3 & 4 ====================
    if os.path.exists("docs/screenshots/fig2_file_ingestion.png"):
        story.append(Paragraph("3. Batch File Ingestion (.txt / .csv):", h2_style))
        story.append(Image("docs/screenshots/fig2_file_ingestion.png", width=490, height=240))
        story.append(
            Paragraph(
                "Fig 3: File ingestion tab accepting UTF-8 .txt documents and .csv survey exports with automatic column auto-detection",
                caption_style,
            )
        )
        story.append(Spacer(1, 5))

    if os.path.exists("docs/screenshots/fig3_stored_records_analytics.png"):
        story.append(
            Paragraph("4. Stored Records & Analytics Dashboard (SQLite Database View):", h2_style)
        )
        story.append(
            Image("docs/screenshots/fig3_stored_records_analytics.png", width=490, height=240)
        )
        story.append(
            Paragraph(
                "Fig 4: Stored Records tab displaying aggregate KPI cards, sentiment filtering, search, and persistent SQLite database records",
                caption_style,
            )
        )

    story.append(PageBreak())

    # ==================== PAGE 7: VERIFICATION, LIMITATIONS, SCOPE & CONCLUSION ====================
    story.append(Paragraph("Testing & Verification", h1_style))
    story.append(
        Paragraph(
            "The automated test suite verifies all functional modules including text ingestion, CSV parsing, "
            "NLTK cleaning and lemmatization, VADER compound scoring, Streamlit file upload adapters, "
            "SQLite schema initialization, user registration, and mood entry persistence.",
            body_style,
        )
    )
    test_out = """============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\\Users\\ganch\\OneDrive\\Desktop\\infosys\\AI_Based_Employee_Wellness_Management_Platform
collected 15 items

tests\\test_pipeline.py ...........                                       [ 73%]
tests\\test_storage.py ....                                               [100%]

============================= 15 passed in 10.08s =============================

Ruff Code Quality: All checks passed!"""
    story.append(Preformatted(test_out, code_style))
    story.append(Spacer(1, 5))

    story.append(Paragraph("Known Limitations", h1_style))
    story.append(
        Paragraph(
            "• <b>Lexicon-Based Sentiment:</b> VADER relies on rule-based heuristics and may miss nuanced workplace sarcasm or subtle emotional context. This will be upgraded in Milestone 2 with transformer models.<br/>"
            "• <b>Local SQLite Storage:</b> SQLite is single-host file storage. While ideal for development and edge deployment, multi-instance enterprise production will transition to managed PostgreSQL.<br/>"
            "• <b>Language Support:</b> The preprocessing pipeline is currently calibrated for English text using NLTK stop-words and WordNet.",
            body_style,
        )
    )
    story.append(Spacer(1, 5))

    story.append(Paragraph("Future Scope", h1_style))
    story.append(
        Paragraph(
            "• <b>Milestone 2: Deep Emotion Classification:</b> Implement multi-class emotion classification (Joy, Sadness, Anger, Fear, Surprise, Exhaustion) using fine-tuned BERT and DistilBERT transformer models.<br/>"
            "• <b>Milestone 3: AI Recommendation Engine & REST API:</b> Develop personalized wellness recommendations based on longitudinal emotional trends, and expose secure FastAPI REST endpoints.<br/>"
            "• <b>Milestone 4: Finalization & Production Packaging:</b> Build CLI tooling, integration test suites, enterprise RBAC authentication, and cloud deployment pipelines.",
            body_style,
        )
    )
    story.append(Spacer(1, 5))

    story.append(Paragraph("Conclusion", h1_style))
    story.append(
        Paragraph(
            "In Milestone 1, the <b>AI-Powered Emotional Tone Analyzer and Well-Being Recommendation System (MoodMentor)</b> "
            "has successfully established a robust, production-ready foundation for workplace sentiment analytics. "
            "All requirements—multi-source ingestion, standardized NLP preprocessing, VADER sentiment scoring, "
            "relational SQLite persistence, and a high-fidelity Streamlit user interface—have been implemented, "
            "rigorously tested with 100% pass rates, and validated against the Infosys Springboard internship objectives.",
            body_style,
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built: {filename}")


if __name__ == "__main__":
    build_pdf()
