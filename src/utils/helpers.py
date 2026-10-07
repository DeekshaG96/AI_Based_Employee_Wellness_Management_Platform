"""Utility functions for data pipelines and text normalization."""
import os
import json

def load_json(filepath: str):
    """Safely loads a JSON configuration file."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def format_percentage(val: float) -> str:
    """Formats a decimal value to a percentage string."""
    return f"{round(val * 100, 1)}%"
