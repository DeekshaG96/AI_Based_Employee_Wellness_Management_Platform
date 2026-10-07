"""
Infosys AI-Based Employee Wellness Management Platform
Milestone 1 (M1) Interactive Demonstration Dashboard
Built with Streamlit & NLTK
"""

import streamlit as st
import pandas as pd
import altair as alt
import sys
import os

# Ensure src modules are resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import TextPreprocessor
from src.sentiment import SentimentAnalyzer
from src.ingestion import TextIngestionEngine
from src.recommender import WellnessRecommender
from src.reporting import ReportGenerator

# Initialize components
preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer()
recommender = WellnessRecommender()
reporter = ReportGenerator()

# Page config
st.set_page_config(
    page_title="Infosys AI Employee Wellness Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #0f172a 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: white;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
        color: #e2e8f0;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .step-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 6px;
    }
    .tag-positive {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .tag-neutral {
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .tag-negative {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="main-header">
    <h1>🌿 AI-Based Employee Wellness Management Platform</h1>
    <p>Infosys Springboard Initiative • Milestone 1 (M1) Ingestion, Preprocessing & VADER Sentiment Engine</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/mental-health.png", width=64)
    st.title("Navigation & Controls")
    mode = st.radio(
        "Choose Ingestion Mode:",
        ["✍️ Live Text Input", "📁 File Upload (.txt / .csv)", "📋 M1 Deliverables Audit", "ℹ️ Project Info"]
    )
    
    st.divider()
    st.subheader("VADER Thresholds")
    st.caption("Configured according to standard NLP benchmarks:")
    st.markdown("- **Positive:** Compound ≥ 0.05\n- **Neutral:** -0.05 < Compound < 0.05\n- **Negative:** Compound ≤ -0.05")
    
    st.divider()
    st.markdown("👨‍🏫 **Mentor Contact:** `springboardmentor06@gmail.com`")
    st.markdown("💻 **Author:** Deeksha Ganesh (`DeekshaG96`)")

# TAB 1: LIVE TEXT INPUT
if mode == "✍️ Live Text Input":
    st.subheader("1️⃣ Real-Time Text Input & NLP Preprocessing Pipeline")
    st.write("Demonstrates: **Text input working**, **Text preprocessing**, **Tokenization**, **Stop-word filtering**, **Lemmatization**, **VADER scoring**, and **Invalid/Empty input handling**.")

    col_input, col_preset = st.columns([3, 1])
    with col_preset:
        preset = st.selectbox(
            "Load Quick Test Preset:",
            [
                "Custom Input",
                "🟢 High Morale (Positive)",
                "⚪ Routine Sync (Neutral)",
                "🔴 Workload Stress (Negative)",
                "⚠️ Empty / Whitespace (Invalid Test)"
            ]
        )
    
    default_text = ""
    if preset == "🟢 High Morale (Positive)":
        default_text = "The sprint retrospective went wonderfully! The entire team feels highly motivated, energised, and truly supported by leadership."
    elif preset == "⚪ Routine Sync (Neutral)":
        default_text = "Attended the weekly status sync meeting. Sprint tasks were reviewed and deadlines were recorded in Jira."
    elif preset == "🔴 Workload Stress (Negative)":
        default_text = "I am feeling completely exhausted and overwhelmed with continuous back-to-back meetings and zero time to complete code reviews."
    elif preset == "⚠️ Empty / Whitespace (Invalid Test)":
        default_text = "     \n    \t  "

    with col_input:
        user_text = st.text_area(
            "Enter Employee Feedback / Survey Response:",
            value=default_text,
            height=120,
            placeholder="Type or paste employee feedback here..."
        )

    btn_analyze = st.button("🚀 Analyze Feedback", type="primary")

    if btn_analyze or default_text:
        # Step 1: Ingestion & Validation
        ingest_res = TextIngestionEngine.ingest_raw_text(user_text)

        if not ingest_res["success"]:
            st.error(f"❌ **Input Validation Failed:** {ingest_res['error']}")
            st.info("💡 **M1 Checklist item verified:** System safely handles invalid or empty input without crashing.")
        else:
            raw_str = ingest_res["entries"][0]["text"]

            # Step 2: NLP Preprocessing Pipeline
            prep_res = preprocessor.process(raw_str)

            if not prep_res["is_valid"]:
                st.warning(f"⚠️ {prep_res['error']}")
            else:
                # Step 3: Sentiment & Recommender
                sent_res = analyzer.analyze(raw_str)
                recs = recommender.recommend(sent_res["sentiment"], sent_res["emotion_tone"])

                st.success("✅ Input validated and successfully processed through NLP pipeline!")

                # Display Sentiment Results
                st.markdown("### 📊 VADER Sentiment Analysis Results")
                m1, m2, m3, m4, m5 = st.columns(5)
                
                with m1:
                    badge_class = f"tag-{sent_res['sentiment'].lower()}"
                    st.markdown(f"**Classification:**<br><span class='{badge_class}'>{sent_res['sentiment']}</span>", unsafe_allow_html=True)
                with m2:
                    st.metric("Compound Score", f"{sent_res['compound']:+.4f}")
                with m3:
                    st.metric("Positive Score", f"{sent_res['positive']:.4f}")
                with m4:
                    st.metric("Neutral Score", f"{sent_res['neutral']:.4f}")
                with m5:
                    st.metric("Negative Score", f"{sent_res['negative']:.4f}")

                # Contextual Emotion Tone & Intensity
                st.markdown(f"**Detected Emotion Tone:** `{sent_res['emotion_tone']}` &nbsp;|&nbsp; **Intensity:** `{sent_res['intensity']}`")

                st.divider()

                # Display Step-by-Step Preprocessing Breakdown
                st.markdown("### 🔬 Step-by-Step NLP Pipeline Verification")
                t1, t2, t3, t4 = st.tabs(["1. Cleaned Text", "2. Tokenization", "3. Stop-Word Filtering", "4. Lemmatization (WordNet)"])

                with t1:
                    st.markdown("**Removed punctuation, URLs, emails, and converted to lowercase:**")
                    st.code(prep_res["cleaned_text"], language="text")
                with t2:
                    st.markdown(f"**Word Tokens ({len(prep_res['tokens'])} total):**")
                    st.write(prep_res["tokens"])
                with t3:
                    st.markdown(f"**Filtered Tokens (Stopwords removed, negations preserved): ({len(prep_res['filtered_tokens'])} remaining):**")
                    st.write(prep_res["filtered_tokens"])
                with t4:
                    st.markdown(f"**Lemmatized Root Words (WordNet Lemmatizer):**")
                    st.write(prep_res["lemmatized_tokens"])

                st.divider()

                # Recommendations
                st.markdown("### 💡 Tailored Wellness Recommendations")
                for rec in recs:
                    with st.container():
                        st.markdown(f"**🎯 {rec['title']}** `{rec['tag']}`")
                        st.write(rec['action'])
                        st.caption(f"Resource: {rec['resource']}")

# TAB 2: FILE UPLOAD (.TXT / .CSV)
elif mode == "📁 File Upload (.txt / .csv)":
    st.subheader("2️⃣ Batch Ingestion via .txt and .csv File Upload")
    st.write("Demonstrates: **.txt file upload**, **.csv file upload**, **Batch Preprocessing & VADER scoring**, **Initial Report Generation**, and **Invalid file handling**.")

    uploaded_file = st.file_uploader(
        "Upload Employee Feedback File (.txt or .csv):",
        type=["txt", "csv"],
        help="Upload raw employee comments in .txt (one per line) or a structured survey .csv"
    )

    if uploaded_file is None:
        st.info("👆 Upload your own file or try the pre-loaded sample datasets included in `data/raw/`:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("📄 Load Sample .txt Dataset (`sample_employee_feedback.txt`)"):
                sample_txt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'sample_employee_feedback.txt')
                if os.path.exists(sample_txt_path):
                    with open(sample_txt_path, 'r', encoding='utf-8') as f:
                        st.session_state['loaded_content'] = f.read()
                        st.session_state['loaded_type'] = 'txt'
        with col_s2:
            if st.button("📊 Load Sample .csv Dataset (`employee_wellness_survey_sample.csv`)"):
                sample_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'employee_wellness_survey_sample.csv')
                if os.path.exists(sample_csv_path):
                    st.session_state['loaded_content'] = pd.read_csv(sample_csv_path)
                    st.session_state['loaded_type'] = 'csv'

    content_to_process = None
    file_type = None

    if uploaded_file is not None:
        file_type = "csv" if uploaded_file.name.endswith(".csv") else "txt"
        content_to_process = uploaded_file.getvalue()
    elif 'loaded_content' in st.session_state:
        content_to_process = st.session_state['loaded_content']
        file_type = st.session_state['loaded_type']

    if content_to_process is not None:
        st.divider()
        if file_type == "txt":
            ingest_result = TextIngestionEngine.ingest_txt_file(content_to_process)
        else:
            ingest_result = TextIngestionEngine.ingest_csv_file(content_to_process)

        if not ingest_result["success"]:
            st.error(f"❌ **File Ingestion Error:** {ingest_result['error']}")
        else:
            entries = ingest_result["entries"]
            st.success(f"✅ Successfully ingested {len(entries)} feedback records from {ingest_result['source_type']}!")

            # Process all records through Preprocessing + VADER
            processed_records = []
            with st.spinner("Processing batch records through NLP pipeline..."):
                for item in entries:
                    raw_text = item["text"]
                    p_res = preprocessor.process(raw_text)
                    s_res = analyzer.analyze(raw_text)
                    
                    merged = {
                        "id": item["id"],
                        "raw_text": raw_text,
                        "cleaned_text": p_res.get("cleaned_text", ""),
                        "tokens": p_res.get("tokens", []),
                        "filtered_tokens": p_res.get("filtered_tokens", []),
                        "lemmatized_tokens": p_res.get("lemmatized_tokens", []),
                        "sentiment": s_res.get("sentiment", "Neutral"),
                        "compound": s_res.get("compound", 0.0),
                        "positive": s_res.get("positive", 0.0),
                        "neutral": s_res.get("neutral", 0.0),
                        "negative": s_res.get("negative", 0.0),
                        "emotion_tone": s_res.get("emotion_tone", "Neutral / Routine"),
                        "intensity": s_res.get("intensity", "None")
                    }
                    processed_records.append(merged)

            # Summary and Initial Report
            summary = reporter.generate_summary(processed_records)
            df_results = reporter.to_dataframe(processed_records)

            st.markdown("### 📈 Milestone 1 Initial Report & Summary Dashboard")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Total Records", summary["total_entries"])
            r2.metric("Positive", f"{summary['positive_count']} ({summary['positive_pct']}%)")
            r3.metric("Neutral", f"{summary['neutral_count']} ({summary['neutral_pct']}%)")
            r4.metric("Negative", f"{summary['negative_count']} ({summary['negative_pct']}%)")

            # Chart breakdown
            col_chart1, col_chart2 = st.columns([1, 1])
            with col_chart1:
                st.markdown("#### Sentiment Proportion")
                chart_df = pd.DataFrame({
                    "Sentiment": ["Positive", "Neutral", "Negative"],
                    "Count": [summary["positive_count"], summary["neutral_count"], summary["negative_count"]]
                })
                chart = alt.Chart(chart_df).mark_bar(cornerRadius=6).encode(
                    x=alt.X("Sentiment:N", sort=None),
                    y="Count:Q",
                    color=alt.Color("Sentiment:N", scale=alt.Scale(
                        domain=["Positive", "Neutral", "Negative"],
                        range=["#22c55e", "#94a3b8", "#ef4444"]
                    ))
                ).properties(height=260)
                st.altair_chart(chart, use_container_width=True)

            with col_chart2:
                st.markdown("#### Top Lemmatized Workplace Themes")
                if summary["top_keywords"]:
                    kw_df = pd.DataFrame(summary["top_keywords"], columns=["Keyword", "Frequency"])
                    kw_chart = alt.Chart(kw_df.head(8)).mark_bar().encode(
                        x="Frequency:Q",
                        y=alt.Y("Keyword:N", sort="-x"),
                        color=alt.value("#0284c7")
                    ).properties(height=260)
                    st.altair_chart(kw_chart, use_container_width=True)
                else:
                    st.write("No top keywords found.")

            # Tabular Detailed View
            st.markdown("### 📋 Detailed Records Breakdown")
            st.dataframe(df_results, use_container_width=True)

            # Download buttons
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Detailed Results (.csv)",
                    data=csv_data,
                    file_name="m1_wellness_sentiment_results.csv",
                    mime="text/csv"
                )
            with col_d2:
                md_report = reporter.generate_markdown_report(summary)
                st.download_button(
                    label="📄 Download Milestone 1 Initial Report (.md)",
                    data=md_report.encode('utf-8'),
                    file_name="M1_Initial_Sentiment_Report.md",
                    mime="text/markdown"
                )

# TAB 3: M1 DELIVERABLES AUDIT
elif mode == "📋 M1 Deliverables Audit":
    st.subheader("📋 Milestone 1 (M1) Deliverables Verification Checklist")
    st.write("Direct mapping against the Infosys Springboard requirements specified by the mentor:")

    checklist_items = [
        ("Text input working", "Implemented in Live Text Input tab with direct multi-line input box and immediate analysis.", True),
        (".txt file upload", "Implemented with automatic encoding detection (UTF-8, Latin-1) and line segmentation.", True),
        (".csv file upload", "Implemented with schema validation, automated column auto-detection, and pandas extraction.", True),
        ("Text preprocessing", "Punctuation stripping, regex normalization, URL/email removal, and case folding.", True),
        ("Tokenization", "NLTK word_tokenize with word boundary extraction and punctuation separation.", True),
        ("Stop-word filtering", "NLTK English stopwords removal with negation preservation to protect sentiment integrity.", True),
        ("Lemmatization", "NLTK WordNetLemmatizer mapping verb and noun root forms.", True),
        ("VADER integration", "NLTK SentimentIntensityAnalyzer polarity scores (pos, neu, neg, compound).", True),
        ("Positive/negative/neutral sentiment", "Standard NLP threshold classification: >= 0.05 (Pos), <= -0.05 (Neg), else Neu.", True),
        ("Compound score", "Accurate compound metric calculation displayed with precision and gauge feedback.", True),
        ("Initial report", "Aggregate counts, percentages, average compound metric, charts, and downloadable Markdown/CSV reports.", True),
        ("Handling invalid/empty input", "Graceful exception protection for nulls, whitespace-only files, corrupt CSVs, and empty inputs.", True)
    ]

    for title, desc, status in checklist_items:
        icon = "✅" if status else "❌"
        with st.container():
            c1, c2 = st.columns([1, 10])
            with c1:
                st.markdown(f"### {icon}")
            with c2:
                st.markdown(f"**{title}**")
                st.caption(desc)
            st.divider()

# TAB 4: PROJECT INFO
elif mode == "ℹ️ Project Info":
    st.subheader("ℹ️ Project Objectives & Springboard Architecture")
    st.markdown("""
    ### 🎯 Project Objectives
    1. **Text ingestion system** *(Completed in M1)*
    2. **Text preprocessing pipeline** *(Completed in M1)*
    3. **Sentiment analysis engine** *(Completed in M1 with VADER)*
    4. **Emotion classification system** *(M1 heuristic + M2 ML pipeline)*
    5. **Emotional intensity analysis** *(Completed in M1)*
    6. **Personalized recommendation engine** *(Integrated into M1 dashboard)*
    7. **User management and database** *(Roadmap: M3)*
    8. **REST APIs** *(FastAPI available in `src/api/main.py`)*
    9. **Analytics dashboard** *(Streamlit web app)*
    10. **Testing and deployment** *(Pytest test suite in `tests/`)*

    ---
    ### 📬 Contact & Submission Details
    - **Mentor Email:** `springboardmentor06@gmail.com`
    - **Repository:** [DeekshaG96/Infosys_AI_Based_Employee_Wellness_Management_Platform](https://github.com/DeekshaG96/Infosys_AI_Based_Employee_Wellness_Management_Platform)
    - **Author:** Deeksha Ganesh
    """)
