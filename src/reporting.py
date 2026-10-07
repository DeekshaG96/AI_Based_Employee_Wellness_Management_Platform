"""
Reporting and Analytics Module.
Fulfills M1 Deliverables:
- Initial Report Generation
- Positive/Negative/Neutral sentiment metrics
- Compound score distribution
- Lemmatized keyword frequency
- Exportable summary
"""

from typing import List, Dict, Any
from collections import Counter
import pandas as pd

class ReportGenerator:
    """Generates analytical and tabular reports for processed wellness feedback."""

    @staticmethod
    def generate_summary(processed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates aggregate metrics across a batch of analyzed feedback entries."""
        if not processed_results:
            return {
                "total_entries": 0,
                "positive_count": 0,
                "neutral_count": 0,
                "negative_count": 0,
                "positive_pct": 0.0,
                "neutral_pct": 0.0,
                "negative_pct": 0.0,
                "avg_compound": 0.0,
                "top_keywords": []
            }

        total = len(processed_results)
        pos = sum(1 for r in processed_results if r.get("sentiment") == "Positive")
        neu = sum(1 for r in processed_results if r.get("sentiment") == "Neutral")
        neg = sum(1 for r in processed_results if r.get("sentiment") == "Negative")

        compounds = [r.get("compound", 0.0) for r in processed_results]
        avg_compound = round(sum(compounds) / total, 4) if total > 0 else 0.0

        # Aggregate lemmatized keywords
        all_lemmas = []
        for r in processed_results:
            all_lemmas.extend(r.get("lemmatized_tokens", []))

        counter = Counter(all_lemmas)
        # Filter single character tokens
        top_keywords = [(word, count) for word, count in counter.most_common(15) if len(word) > 2]

        return {
            "total_entries": total,
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg,
            "positive_pct": round((pos / total) * 100, 1),
            "neutral_pct": round((neu / total) * 100, 1),
            "negative_pct": round((neg / total) * 100, 1),
            "avg_compound": avg_compound,
            "top_keywords": top_keywords
        }

    @staticmethod
    def to_dataframe(processed_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts results into a pandas DataFrame ready for table display or CSV export."""
        rows = []
        for r in processed_results:
            rows.append({
                "ID": r.get("id", "-"),
                "Raw Text": r.get("raw_text", ""),
                "Cleaned Text": r.get("cleaned_text", ""),
                "Lemmatized Tokens": ", ".join(r.get("lemmatized_tokens", [])),
                "Sentiment": r.get("sentiment", "Neutral"),
                "Compound Score": r.get("compound", 0.0),
                "Positive Score": r.get("positive", 0.0),
                "Neutral Score": r.get("neutral", 0.0),
                "Negative Score": r.get("negative", 0.0),
                "Emotion Tone": r.get("emotion_tone", "Neutral / Routine"),
                "Intensity": r.get("intensity", "None")
            })
        return pd.DataFrame(rows)

    @staticmethod
    def generate_markdown_report(summary: Dict[str, Any]) -> str:
        """Creates a downloadable text/markdown executive summary."""
        md = f"""# 📊 Employee Wellness Sentiment Analysis - Milestone 1 Initial Report
**Platform:** Infosys AI-Based Employee Wellness Management Platform  
**Analysis Engine:** VADER Lexicon & NLTK Preprocessing Pipeline

---

## 📌 Executive Summary
- **Total Feedback Entries Analyzed:** {summary['total_entries']}
- **Average Compound Score:** {summary['avg_compound']}
- **Overall Workplace Sentiment:** {'🟢 Positive' if summary['avg_compound'] >= 0.05 else ('🔴 Needs Attention' if summary['avg_compound'] <= -0.05 else '⚪ Balanced / Neutral')}

## 📈 Sentiment Distribution
| Sentiment Class | Count | Percentage |
| :--- | :--- | :--- |
| **Positive** | {summary['positive_count']} | {summary['positive_pct']}% |
| **Neutral**  | {summary['neutral_count']}  | {summary['neutral_pct']}% |
| **Negative** | {summary['negative_count']} | {summary['negative_pct']}% |

## 🔑 Key Lemmatized Workplace Themes
"""
        for word, count in summary.get('top_keywords', []):
            md += f"- **{word}**: {count} mentions\n"

        md += """
---
*Generated automatically by Infosys Springboard M1 Analytics Engine.*
"""
        return md
