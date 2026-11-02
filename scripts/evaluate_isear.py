"""Evaluate a GoEmotions classifier on a local, untouched ISEAR test CSV/TSV.

The seven-class label mapping is approximate, since the two datasets use different
emotion taxonomies. The report contains aggregate metrics only.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "goemotions-distilbert"
ISEAR_LABELS = ("anger", "disgust", "fear", "joy", "sadness", "shame", "guilt")
ISEAR_ID_TO_LABEL = {
    "1": "joy", "2": "fear", "3": "anger", "4": "sadness",
    "5": "disgust", "6": "shame", "7": "guilt",
}
GOEMOTIONS_TO_ISEAR = {
    "anger": ("anger", "annoyance", "disapproval"),
    "disgust": ("disgust",),
    "fear": ("fear", "nervousness"),
    "joy": (
        "joy", "amusement", "approval", "caring", "excitement", "gratitude",
        "love", "optimism", "pride", "relief", "admiration", "desire",
    ),
    "sadness": ("sadness", "disappointment", "grief"),
    "shame": ("embarrassment",),
    "guilt": ("remorse",),
}


def read_data(path: Path, text_column: str, label_column: str) -> tuple[list[str], list[str]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    texts: list[str] = []
    labels: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames or text_column not in reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(
                f"Expected columns {text_column!r} and {label_column!r}; found {reader.fieldnames}."
            )
        for row_number, row in enumerate(reader, start=2):
            text = (row.get(text_column) or "").strip()
            raw_label = (row.get(label_column) or "").strip().casefold()
            label = ISEAR_ID_TO_LABEL.get(raw_label, raw_label)
            if not text or label not in ISEAR_LABELS:
                raise ValueError(f"Invalid or unsupported ISEAR row {row_number}.")
            texts.append(text)
            labels.append(label)
    if not texts:
        raise ValueError("No records found in the ISEAR file.")
    return texts, labels


def score(predictions: list[str], actual: list[str]) -> dict[str, Any]:
    confusion = {label: {key: 0 for key in ISEAR_LABELS} for label in ISEAR_LABELS}
    for predicted, expected in zip(predictions, actual, strict=True):
        confusion[expected][predicted] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    correct = sum(predicted == expected for predicted, expected in zip(predictions, actual, strict=True))
    for label in ISEAR_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in ISEAR_LABELS if other != label)
        fn = sum(confusion[label][other] for other in ISEAR_LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values()),
        }
        f1_values.append(f1)
    return {
        "examples": len(actual),
        "accuracy": round(correct / len(actual), 4),
        "macro_f1": round(sum(f1_values) / len(ISEAR_LABELS), 4),
        "per_class": per_class,
        "confusion_matrix_rows_actual_columns_predicted": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_file", type=Path, help="A local held-out ISEAR test CSV/TSV. No training rows.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="emotion")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional aggregate JSON report path.")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive.")
    texts, actual = read_data(args.test_file, args.text_column, args.label_column)
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Install the ML extras: python -m pip install -e '.[ml]'") from exc
    classifier = pipeline("text-classification", model=args.model, top_k=None, device=-1)
    problem_type = getattr(classifier.model.config, "problem_type", None)
    if problem_type is None:
        classifier.model.config.problem_type = "multi_label_classification"
    elif problem_type != "multi_label_classification":
        raise SystemExit("The selected checkpoint is not configured for GoEmotions multi-label output.")
    outputs = classifier(texts, batch_size=args.batch_size, truncation=True)
    mapped: list[str] = []
    for output in outputs:
        scores = {item["label"].strip().lower(): float(item["score"]) for item in output}
        class_scores = {
            category: max((scores.get(label, 0.0) for label in sources), default=0.0)
            for category, sources in GOEMOTIONS_TO_ISEAR.items()
        }
        mapped.append(max(class_scores, key=class_scores.get))

    result = {
        "model": args.model,
        "dataset": "ISEAR external transfer evaluation",
        "label_mapping": GOEMOTIONS_TO_ISEAR,
        "mapping_note": "Approximate taxonomy mapping: shame to embarrassment and guilt to remorse. This is an external transfer check, not a same-label benchmark.",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": score(mapped, actual),
        "privacy": "Only aggregate metrics are written; input text and predictions are not saved.",
        "limitations": "ISEAR is a different domain and label taxonomy. Verify the data source, license, and split before reporting results.",
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()