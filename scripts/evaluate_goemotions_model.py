"""Evaluate a multi-label Hugging Face model on GoEmotions test.tsv."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "goemotions"
DEFAULT_MODEL = "bhadresh-savani/bert-base-go-emotion"


def read_split(data_dir: Path, split: str) -> tuple[list[str], list[set[int]]]:
    texts: list[str] = []
    targets: list[set[int]] = []
    split_path = data_dir / split
    delimiter = "," if split_path.suffix.lower() == ".csv" else "\t"
    with split_path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter=delimiter), start=1):
            if row and row_number == 1 and row[0].strip().lower() == "text":
                continue
            if len(row) != 3 or not row[0].strip():
                raise ValueError(f"Malformed row in {split_path}.")
            texts.append(row[0])
            targets.append({int(value.strip()) for value in row[1].split(",") if value.strip()})
    return texts, targets


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    predictions: list[dict[str, float]],
    targets: list[set[int]],
    labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target counts do not match.")
    tp = fp = fn = exact = 0
    per_label: dict[str, dict[str, float | int]] = {}
    macro_precision = macro_recall = macro_f1 = 0.0
    for index, label in enumerate(labels):
        label_tp = label_fp = label_fn = 0
        for prediction, target in zip(predictions, targets, strict=True):
            predicted = prediction.get(label.lower(), 0.0) >= threshold
            actual = index in target
            label_tp += predicted and actual
            label_fp += predicted and not actual
            label_fn += actual and not predicted
        precision = divide(label_tp, label_tp + label_fp)
        recall = divide(label_tp, label_tp + label_fn)
        f1 = divide(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": label_tp + label_fn,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        tp += label_tp
        fp += label_fp
        fn += label_fn
    for prediction, target in zip(predictions, targets, strict=True):
        predicted_ids = {
            index for index, label in enumerate(labels)
            if prediction.get(label.lower(), 0.0) >= threshold
        }
        exact += predicted_ids == target

    micro_precision = divide(tp, tp + fp)
    micro_recall = divide(tp, tp + fn)
    return {
        "examples": len(targets),
        "threshold": threshold,
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(divide(2 * micro_precision * micro_recall, micro_precision + micro_recall), 4),
        "macro_precision": round(macro_precision / len(labels), 4),
        "macro_recall": round(macro_recall / len(labels), 4),
        "macro_f1": round(macro_f1 / len(labels), 4),
        "exact_match_accuracy": round(exact / len(targets), 4) if targets else 0.0,
        "per_label": per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Optional quick run; omit to evaluate the full split.")
    parser.add_argument("--output", type=Path, help="Optional JSON path for aggregate metrics only.")
    args = parser.parse_args()
    if not (args.data_dir / args.split).is_file() and args.split == "test.csv" and (args.data_dir / "test.tsv").is_file():
        args.split = "test.tsv"
    labels = [line.strip() for line in (args.data_dir / "emotions.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    texts, targets = read_split(args.data_dir, args.split)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be a positive integer.")
        texts, targets = texts[:args.limit], targets[:args.limit]

    try:
        import torch
        import transformers
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Install project dependencies first: python -m pip install -r requirements.txt") from exc

    classifier = pipeline("text-classification", model=args.model, top_k=None, device=-1)
    problem_type = getattr(classifier.model.config, "problem_type", None)
    if problem_type is None:
        # This checkpoint does not declare the loss type; the dataset and model card are multi-label.
        classifier.model.config.problem_type = "multi_label_classification"
        problem_type = "multi_label_classification"
    elif problem_type != "multi_label_classification":
        raise SystemExit(f"{args.model} is configured as {problem_type!r}, not multi-label classification.")

    predicted_rows = classifier(texts, batch_size=args.batch_size, truncation=True)
    predictions = [
        {item["label"].strip().lower(): float(item["score"]) for item in row}
        for row in predicted_rows
    ]
    if not predictions or not set(labels).issubset(predictions[0]):
        missing = sorted(set(labels) - (set(predictions[0]) if predictions else set()))
        raise SystemExit(f"Model label names do not match the dataset. Missing labels: {missing}")

    result = {
        "model": args.model,
        "model_problem_type": problem_type or "inferred by Transformers",
        "dataset": "GoEmotions simplified agree-2",
        "split": args.split,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": evaluate(predictions, targets, labels, args.threshold),
        "runtime": {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "privacy": "Only aggregate metrics are written; input texts and predictions are not saved.",
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
