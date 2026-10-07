"""
Data Ingestion and Validation Engine.
Fulfills M1 Deliverables:
- Text input ingestion
- .txt file upload ingestion
- .csv file upload ingestion
- Handling invalid / empty input
"""

import io
from typing import List, Dict, Any, Union, Tuple
import pandas as pd

class TextIngestionEngine:
    """Manages raw input ingestion from direct text, .txt files, and .csv files."""

    @staticmethod
    def ingest_raw_text(text: str) -> Dict[str, Any]:
        """Validates and cleans direct raw text input."""
        if text is None:
            return {"success": False, "error": "Input is null.", "entries": []}
        trimmed = text.strip()
        if not trimmed:
            return {
                "success": False,
                "error": "Input text is empty or contains only whitespace.",
                "entries": []
            }
        return {
            "success": True,
            "error": None,
            "source_type": "direct_text",
            "entries": [{"id": 1, "text": trimmed}]
        }

    @staticmethod
    def ingest_txt_file(file_content: Union[str, bytes]) -> Dict[str, Any]:
        """Ingests and validates .txt file content, splitting into discrete non-empty lines."""
        if not file_content:
            return {"success": False, "error": "Uploaded .txt file is empty.", "entries": []}

        if isinstance(file_content, bytes):
            try:
                text_str = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_str = file_content.decode('latin-1')
                except Exception as e:
                    return {"success": False, "error": f"Failed to decode file encoding: {str(e)}", "entries": []}
        else:
            text_str = file_content

        lines = [line.strip() for line in text_str.splitlines() if line.strip()]
        if not lines:
            return {
                "success": False,
                "error": "The .txt file contains no readable text lines.",
                "entries": []
            }

        entries = [{"id": idx + 1, "text": line} for idx, line in enumerate(lines)]
        return {
            "success": True,
            "error": None,
            "source_type": "txt_file",
            "entries": entries,
            "line_count": len(entries)
        }

    @staticmethod
    def ingest_csv_file(file_content: Union[str, bytes, io.BytesIO, pd.DataFrame], text_column: str = None) -> Dict[str, Any]:
        """Ingests .csv file content, auto-detects text column or allows user selection."""
        try:
            if isinstance(file_content, pd.DataFrame):
                df = file_content.copy()
            elif isinstance(file_content, bytes):
                df = pd.read_csv(io.BytesIO(file_content))
            elif isinstance(file_content, str):
                df = pd.read_csv(io.StringIO(file_content))
            else:
                df = pd.read_csv(file_content)
        except Exception as e:
            return {
                "success": False,
                "error": f"Could not parse CSV format: {str(e)}",
                "entries": [],
                "columns": []
            }

        if df.empty:
            return {
                "success": False,
                "error": "The uploaded CSV file is empty (contains no data rows).",
                "entries": [],
                "columns": list(df.columns)
            }

        # Auto-detect text column if not specified
        available_cols = list(df.columns)
        selected_col = text_column
        if not selected_col or selected_col not in available_cols:
            candidates = ['feedback_text', 'feedback', 'text', 'comments', 'response', 'survey_response']
            for c in candidates:
                if c in available_cols:
                    selected_col = c
                    break
            if not selected_col:
                # Find the first object/string column
                for col in available_cols:
                    if df[col].dtype == object or df[col].dtype == 'string':
                        selected_col = col
                        break

        if not selected_col:
            return {
                "success": False,
                "error": "No suitable text column found in CSV. Please specify a column.",
                "entries": [],
                "columns": available_cols
            }

        # Extract and clean rows
        df_valid = df.dropna(subset=[selected_col])
        entries = []
        for idx, row in df_valid.iterrows():
            txt_val = str(row[selected_col]).strip()
            if txt_val:
                item = {
                    "id": idx + 1,
                    "text": txt_val,
                    "metadata": {col: row[col] for col in available_cols if col != selected_col}
                }
                entries.append(item)

        if not entries:
            return {
                "success": False,
                "error": f"Column '{selected_col}' exists but contains only empty/blank values.",
                "entries": [],
                "columns": available_cols
            }

        return {
            "success": True,
            "error": None,
            "source_type": "csv_file",
            "selected_column": selected_col,
            "columns": available_cols,
            "entries": entries,
            "total_rows": len(entries)
        }
