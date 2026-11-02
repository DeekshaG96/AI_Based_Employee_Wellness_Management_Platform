"""Score VADER on a local binary sentiment CSV/TSV.

The input must already have explicit positive and negative labels. Neutral and any
other labels are excluded. The script saves only aggregate results.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_rows(
    path: Path,
    text_column: str,
    label_column: str,
    positive_label: str,
    negative_label: str,
) -> tuple[list[tuple[str, int]], int]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    examples: list[tuple[str, int]] = []
    ignored = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames or text_column not in reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(
                f"Expected columns {text_column!r} and {label_column!r}; found {reader.fieldnames}."
            )
        positive = positive_label.strip().casefold()
        negative = negative_label.strip().casefold()
        for row_number, row in enumerate(reader, start=2):
            text = (row.get(text_column) or "").strip()
            label = (row.get(label_column) or "").strip().casefold()
            if not text:
                raise ValueError(f"Empty text in row {row_number}.")
            if label == positive:
                examples.append((text, 1))
            elif label == negative:
                examples.append((text, 0))
            else:
                ignored += 1
    if not examples:
        raise ValueError("No positive/negative examples found. Set the label names to match the file.")
    return examples, ignored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="A local CSV/TSV export of the labelled test subset.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--positive-label", default="positive")
    parser.add_argument("--negative-label", default="negative")
    parser.add_argument("--threshold", type=float, default=0.05, help="VADER compound cutoff for positive.")
    parser.add_argument("--output", type=Path, help="Optional JSON path for aggregate metrics.")
    args = parser.parse_args()
    if not -1.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between -1 and 1.")
    examples, ignored = read_rows(
        args.dataset, args.text_column, args.label_column,
        args.positive_label, args.negative_label,
    )
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise SystemExit("Install NLTK with: python -m pip install nltk") from exc
    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError as exc:
        raise SystemExit("VADER lexicon is missing. Run python -m nltk.downloader vader_lexicon.") from exc

    tp = tn = fp = fn = 0
    for text, actual in examples:
        predicted = int(analyzer.polarity_scores(text)["compound"] >= args.threshold)
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        elif predicted == 1:
            fp += 1
        else:
            fn += 1
    positive_precision = tp / (tp + fp) if tp + fp else 0.0
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    positive_f1 = (
        2 * positive_precision * positive_recall / (positive_precision + positive_recall)
        if positive_precision + positive_recall else 0.0
    )
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall else 0.0
    )
    result: dict[str, Any] = {
        "model": "VADER",
        "dataset_file": args.dataset.name,
        "evaluated_examples": len(examples),
        "ignored_non_binary_examples": ignored,
        "compound_positive_threshold": args.threshold,
        "accuracy": round((tp + tn) / len(examples), 4),
        "precision_positive": round(positive_precision, 4),
        "recall_positive": round(positive_recall, 4),
        "f1_positive": round(positive_f1, 4),
        "precision_negative": round(negative_precision, 4),
        "recall_negative": round(negative_recall, 4),
        "f1_negative": round(negative_f1, 4),
        "macro_f1": round((positive_f1 + negative_f1) / 2, 4),
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "privacy": "Only aggregate metrics are written; input text is not saved.",
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()