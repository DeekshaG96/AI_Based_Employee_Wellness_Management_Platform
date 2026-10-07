# AI-Based Employee Wellness Management Platform

An intelligent employee wellness monitoring and assistance platform developed for the Infosys Springboard initiative. The system leverages NLP models (VADER, BERT, DistilBERT) to analyze employee feedback, gauge workplace sentiment, and offer actionable wellness recommendations.

## 🚀 Key Features

- **Sentiment & Mood Tracking:** Real-time sentiment analysis on workplace feedback and survey responses.
- **Multi-Label Classification:** Categorizes employee concerns (e.g., Burnout, Work-Life Balance, Physical Health, Team Dynamics).
- **Personalized Recommendations:** Automated suggest engine for wellness activities, break reminders, and support resources.
- **Dashboard Analytics:** Visual summary of overall team/organizational wellness metrics.

## 🛠️ Tech Stack

- **Machine Learning & NLP:** Python, NLTK, Scikit-learn, Hugging Face Transformers
- **Backend:** FastAPI / Flask
- **Frontend:** Streamlit
- **Database:** SQLite / PostgreSQL

## 📦 Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DeekshaG96/Infosys_AI_Based_Employee_Wellness_Management_Platform.git
   cd Infosys_AI_Based_Employee_Wellness_Management_Platform
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python src/main.py
   ```
