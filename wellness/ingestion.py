"""Validated text and file ingestion."""

import csv
from pathlib import Path

from .errors import InputValidationError

_TEXT_COLUMNS = ("text", "feedback", "comment", "message", "content")


def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise InputValidationError("Text input must be a string.")
    if not text.strip():
        raise InputValidationError("Text input cannot be empty.")
    return text.strip()


def read_input_file(path: str | Path, text_column: str | None = None) -> tuple[str, str]:
    """Read a TXT or CSV file and return its text plus a display source name."""
    file_path = Path(path)
    if not file_path.exists():
        raise InputValidationError(f"Input file does not exist: {file_path}")
    if not file_path.is_file():
        raise InputValidationError(f"Input path is not a file: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix not in {".txt", ".csv"}:
        raise InputValidationError("Supported input files are .txt and .csv.")
    try:
        if suffix == ".txt":
            text = file_path.read_text(encoding="utf-8-sig")
        else:
            text = _read_csv(file_path, text_column)
    except UnicodeDecodeError as exc:
        raise InputValidationError("Input file must be UTF-8 encoded.") from exc
    return validate_text(text), str(file_path)


def _read_csv(path: Path, text_column: str | None) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise InputValidationError("CSV input must contain a header row.")
            columns = [column.strip() for column in reader.fieldnames if column]
            selected = text_column.strip() if text_column else next(
                (column for column in columns if column.lower() in _TEXT_COLUMNS), None
            )
            if not selected or selected not in columns:
                raise InputValidationError(
                    "CSV must contain a text column named text, feedback, comment, "
                    "message, or content; use --text-column to select another column."
                )
            values = []
            for row in reader:
                value = (row.get(selected) or "").strip()
                if value:
                    values.append(value)
    except csv.Error as exc:
        raise InputValidationError(f"Malformed CSV input: {exc}") from exc
    if not values:
        raise InputValidationError(f"CSV text column '{selected}' contains no non-empty text.")
    return "\n".join(values)
