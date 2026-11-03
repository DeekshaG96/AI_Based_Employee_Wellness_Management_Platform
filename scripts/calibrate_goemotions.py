"""Calibrate GoEmotions thresholds on validation, then score a held-out test split.

Thresholds are selected on validation only. Reports contain aggregate metrics and
confusion counts only, never source text or predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "goemotions"
DEFAULT_MODEL = ROOT / "models" / "goemotions-distilbert"


def read_split(path: Path, limit: int | None) -> tuple[list[str], list[set[int]]]:
    texts: list[str] = []
    targets: list[set[int]] = []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter=delimiter), start=1):
            if row_number == 1 and row and row[0].strip().lower() == "text":
                continue
            if len(row) < 2 or not row[0].strip():
                raise ValueError(f"Malformed record in {path} at row {row_number}.")
            try:
                ids = {int(part.strip()) for part in row[1].split(",") if part.strip()}
            except ValueError as exc:
                raise ValueError(f"Invalid label ID in {path} at row {row_number}.") from exc
            if not ids:
                raise ValueError(f"Missing labels in {path} at row {row_number}.")
            texts.append(row[0])
            targets.append(ids)
            if limit is not None and len(texts) >= limit:
                break
    if not texts:
        raise ValueError(f"No records found in {path}.")
    return texts, targets


def _metric_arrays(
    predictions: list[dict[str, float]],
    targets: list[set[int]],
    labels: list[str],
) -> tuple[Any, Any]:
    import numpy as np

    scores = np.asarray(
        [[row.get(label.lower(), 0.0) for label in labels] for row in predictions],
        dtype=np.float32,
    )
    actual = np.zeros(scores.shape, dtype=np.bool_)
    for row_index, label_ids in enumerate(targets):
        actual[row_index, list(label_ids)] = True
    return scores, actual


def metrics(
    predictions: list[dict[str, float]],
    targets: list[set[int]],
    labels: list[str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    import numpy as np

    scores, actual = _metric_arrays(predictions, targets, labels)
    threshold_values = np.asarray([thresholds[label] for label in labels], dtype=np.float32)
    predicted = scores >= threshold_values[None, :]
    true_positive = np.logical_and(predicted, actual).sum(axis=0)
    false_positive = np.logical_and(predicted, np.logical_not(actual)).sum(axis=0)
    false_negative = np.logical_and(np.logical_not(predicted), actual).sum(axis=0)
    true_negative = np.logical_and(np.logical_not(predicted), np.logical_not(actual)).sum(axis=0)

    precisions = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros(len(labels), dtype=np.float64),
        where=(true_positive + false_positive) != 0,
    )
    recalls = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros(len(labels), dtype=np.float64),
        where=(true_positive + false_negative) != 0,
    )
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros(len(labels), dtype=np.float64),
        where=(precisions + recalls) != 0,
    )

    per_label: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        per_label[label] = {
            "threshold": round(float(threshold_values[index]), 4),
            "precision": round(float(precisions[index]), 4),
            "recall": round(float(recalls[index]), 4),
            "f1": round(float(f1_scores[index]), 4),
            "support": int(actual[:, index].sum()),
            "tp": int(true_positive[index]),
            "fp": int(false_positive[index]),
            "fn": int(false_negative[index]),
            "tn": int(true_negative[index]),
        }

    micro_tp = int(true_positive.sum())
    micro_fp = int(false_positive.sum())
    micro_fn = int(false_negative.sum())
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "examples": len(targets),
        "macro_precision": round(float(precisions.mean()), 4),
        "macro_recall": round(float(recalls.mean()), 4),
        "macro_f1": round(float(f1_scores.mean()), 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "exact_match_accuracy": round(float(np.all(predicted == actual, axis=1).mean()), 4),
        "hamming_loss": round(float(np.mean(predicted != actual)), 4),
        "per_label_confusion": per_label,
    }


def tune_thresholds(
    predictions: list[dict[str, float]],
    targets: list[set[int]],
    labels: list[str],
    min_support: int,
) -> tuple[float, dict[str, float]]:
    import numpy as np

    scores, actual = _metric_arrays(predictions, targets, labels)
    candidates = [round(value / 100, 2) for value in range(5, 96, 5)]
    supports = actual.sum(axis=0)
    global_threshold = candidates[0]
    best_macro_f1 = -1.0

    for candidate in candidates:
        predicted = scores >= candidate
        true_positive = np.logical_and(predicted, actual).sum(axis=0)
        false_positive = np.logical_and(predicted, np.logical_not(actual)).sum(axis=0)
        false_negative = np.logical_and(np.logical_not(predicted), actual).sum(axis=0)
        precision = np.divide(
            true_positive,
            true_positive + false_positive,
            out=np.zeros(len(labels), dtype=np.float64),
            where=(true_positive + false_positive) != 0,
        )
        recall = np.divide(
            true_positive,
            true_positive + false_negative,
            out=np.zeros(len(labels), dtype=np.float64),
            where=(true_positive + false_negative) != 0,
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(len(labels), dtype=np.float64),
            where=(precision + recall) != 0,
        )
        macro_f1 = float(f1.mean())
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            global_threshold = candidate

    thresholds: dict[str, float] = {}
    for index, label in enumerate(labels):
        if supports[index] < min_support:
            thresholds[label] = global_threshold
            continue

        class_scores = scores[:, index]
        class_actual = actual[:, index]
        best_threshold = global_threshold
        best_f1 = -1.0
        for candidate in candidates:
            predicted = class_scores >= candidate
            tp = int(np.logical_and(predicted, class_actual).sum())
            fp = int(np.logical_and(predicted, np.logical_not(class_actual)).sum())
            fn = int(np.logical_and(np.logical_not(predicted), class_actual).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = candidate
        thresholds[label] = best_threshold
    return global_threshold, thresholds


def score_split(
    classifier: Any,
    data_dir: Path,
    split_names: tuple[str, ...],
    labels: list[str],
    batch_size: int,
    limit: int | None,
) -> tuple[list[dict[str, float]], list[set[int]], str]:
    path = next((data_dir / name for name in split_names if (data_dir / name).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing split: expected one of {split_names} in {data_dir}.")
    texts, targets = read_split(path, limit)
    output = classifier(texts, batch_size=batch_size, truncation=True)
    predictions = [
        {item["label"].strip().lower(): float(item["score"]) for item in row}
        for row in output
    ]
    if len(predictions) != len(targets):
        raise ValueError("Model output count does not match dataset rows.")
    expected_labels = {label.lower() for label in labels}
    if predictions and not expected_labels.issubset(predictions[0]):
        missing = sorted(expected_labels - set(predictions[0]))
        raise ValueError(f"Model labels do not match GoEmotions. Missing: {missing}")
    for target in targets:
        if min(target) < 0 or max(target) >= len(labels):
            raise ValueError(f"Label ID outside the expected {len(labels)} GoEmotions classes.")
    return predictions, targets, path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5, help="Uncalibrated comparison threshold.")
    parser.add_argument("--limit", type=int, help="Optional smoke run; omit for full splits.")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "goemotions-calibration.json")
    args = parser.parse_args()
    if args.batch_size < 1 or args.min_support < 1 or not 0.0 < args.threshold < 1.0:
        raise SystemExit("batch size and min support must be positive; threshold must be in (0, 1).")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive.")
    labels = [line.strip() for line in (args.data_dir / "emotions.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 28:
        raise SystemExit(f"Expected the 28 GoEmotions labels; found {len(labels)}.")

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Install the ML extras: python -m pip install -e '.[ml]'") from exc
    classifier = pipeline("text-classification", model=args.model, top_k=None, device=-1)
    if getattr(classifier.model.config, "problem_type", None) is None:
        classifier.model.config.problem_type = "multi_label_classification"
    elif classifier.model.config.problem_type != "multi_label_classification":
        raise SystemExit("The selected checkpoint is not configured for multi-label classification.")

    validation_predictions, validation_targets, validation_split = score_split(
        classifier, args.data_dir, ("validation.csv", "dev.tsv"), labels, args.batch_size, args.limit
    )
    global_threshold, thresholds = tune_thresholds(
        validation_predictions, validation_targets, labels, args.min_support
    )
    fixed_thresholds = {label: args.threshold for label in labels}
    validation_fixed = metrics(validation_predictions, validation_targets, labels, fixed_thresholds)
    validation_tuned = metrics(validation_predictions, validation_targets, labels, thresholds)

    test_predictions, test_targets, test_split = score_split(
        classifier, args.data_dir, ("test.csv", "test.tsv"), labels, args.batch_size, args.limit
    )
    test_fixed = metrics(test_predictions, test_targets, labels, fixed_thresholds)
    test_tuned = metrics(test_predictions, test_targets, labels, thresholds)
    result = {
        "model": args.model,
        "dataset": "GoEmotions simplified agree-2",
        "label_count": len(labels),
        "validation_split": validation_split,
        "test_split": test_split,
        "threshold_policy": "Per-label F1 thresholds chosen on validation only; labels below min_support use the validation-selected global macro-F1 threshold.",
        "min_support": args.min_support,
        "global_validation_threshold": global_threshold,
        "thresholds": thresholds,
        "validation_fixed_threshold_metrics": validation_fixed,
        "validation_calibrated_metrics": validation_tuned,
        "held_out_test_fixed_threshold_metrics": test_fixed,
        "held_out_test_calibrated_metrics": test_tuned,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "privacy": "Only aggregate counts and metrics are written; source text and predictions are not saved.",
        "limitations": "Calibration may overfit the development split. GoEmotions is Reddit-derived and is not a validated employee-journal benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()