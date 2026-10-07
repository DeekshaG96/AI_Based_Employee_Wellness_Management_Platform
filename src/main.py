"""
Infosys AI-Based Employee Wellness Management Platform
Main CLI entry point for testing and running the application.
"""

import sys
import os
import argparse
import subprocess

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import TextPreprocessor
from src.sentiment import SentimentAnalyzer
from src.ingestion import TextIngestionEngine
from src.recommender import WellnessRecommender
from src.reporting import ReportGenerator

def run_demo():
    """Runs a quick end-to-end demonstration of the M1 pipeline."""
    print("=" * 70)
    print("🌿 INFOSYS AI-BASED EMPLOYEE WELLNESS MANAGEMENT PLATFORM (M1 DEMO)")
    print("=" * 70)

    sample_texts = [
        "The project team delivered an outstanding sprint and everyone feels valued and happy!",
        "Routine daily standup completed with standard progress updates.",
        "I am totally overwhelmed with continuous late-night deployments and experiencing severe burnout."
    ]

    preprocessor = TextPreprocessor()
    analyzer = SentimentAnalyzer()
    recommender = WellnessRecommender()

    for idx, text in enumerate(sample_texts, 1):
        print(f"\n--- [Sample {idx}] ---")
        print(f"Raw Input: \"{text}\"")
        
        # 1. Preprocessing
        prep = preprocessor.process(text)
        print(f"Cleaned Text: {prep['cleaned_text']}")
        print(f"Tokens: {prep['tokens']}")
        print(f"Stopwords Filtered: {prep['filtered_tokens']}")
        print(f"Lemmatized Tokens: {prep['lemmatized_tokens']}")

        # 2. Sentiment Analysis
        sentiment = analyzer.analyze(text)
        print(f"Sentiment: {sentiment['sentiment']} (Compound: {sentiment['compound']:+.4f})")
        print(f"Polarity Breakdown: Pos={sentiment['positive']}, Neu={sentiment['neutral']}, Neg={sentiment['negative']}")
        print(f"Emotion Tone: {sentiment['emotion_tone']} | Intensity: {sentiment['intensity']}")

        # 3. Wellness Recommendations
        recs = recommender.recommend(sentiment['sentiment'], sentiment['emotion_tone'])
        print("Recommended Wellness Interventions:")
        for r in recs:
            print(f"  * [{r['tag']}] {r['title']}: {r['action']}")

    print("\n" + "=" * 70)
    print("✅ M1 Pipeline executed successfully! To launch the interactive web dashboard, run:")
    print("   streamlit run src/app.py")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Infosys AI-Based Employee Wellness Platform")
    parser.add_argument("--demo", action="store_true", help="Run end-to-end M1 demo in terminal")
    parser.add_argument("--analyze", type=str, help="Analyze sentiment of a single custom text")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web dashboard")

    args = parser.parse_args()

    if args.analyze:
        analyzer = SentimentAnalyzer()
        preprocessor = TextPreprocessor()
        prep = preprocessor.process(args.analyze)
        sent = analyzer.analyze(args.analyze)
        print(f"Cleaned: {prep['cleaned_text']}")
        print(f"Lemmatized: {prep['lemmatized_tokens']}")
        print(f"Sentiment: {sent['sentiment']} (Compound: {sent['compound']:+.4f})")
    elif args.app:
        app_path = os.path.join(os.path.dirname(__file__), 'app.py')
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    else:
        run_demo()

if __name__ == "__main__":
    main()
