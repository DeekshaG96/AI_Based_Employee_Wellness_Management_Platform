"""Compare aggregate reports from two GoEmotions fine-tuning runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test_metrics") or {}
    training = data.get("training") or {}
    return {
        "report": path.name,
        "base_model": data.get("base_model", "unknown"),
        "examples_train": next(
            (
                values
                for key, values in (training.get("examples_per_split") or {}).items()
                if key.endswith(("train.csv", "train.tsv"))
            ),
            "unknown",
        ),
        "validation_macro_f1": data.get("validation_macro_f1_at_fixed_threshold", "unknown"),
        "test_macro_f1": test.get("macro_f1", "unknown"),
        "test_micro_f1": test.get("micro_f1", "unknown"),
        "test_precision": test.get("micro_precision", "unknown"),
        "test_recall": test.get("micro_recall", "unknown"),
        "test_exact_match_accuracy": test.get("exact_match_accuracy", "unknown"),
        "training_seconds": data.get("training_seconds", "unknown"),
        "seed": training.get("seed", "unknown"),
        "limit": training.get("limit"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="First model training JSON report.")
    parser.add_argument("second", type=Path, help="Second model training JSON report.")
    parser.add_argument("--output", type=Path, help="Optional JSON comparison report.")
    args = parser.parse_args()
    rows = [summarize(args.first), summarize(args.second)]
    result = {
        "comparison_policy": "Compare the same splits, training seed, limit, and fixed-threshold metrics. A non-null limit is a smoke run, not full-data evidence.",
        "models": rows,
        "privacy": "Only aggregate model metadata and metrics are included.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = (
        "base_model", "validation_macro_f1", "test_macro_f1", "test_micro_f1",
        "test_precision", "test_recall", "test_exact_match_accuracy", "training_seconds",
    )
    widths = {column: max(len(column), *(len(str(row[column])) for row in rows)) for column in columns}
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))
    if args.output:
        print(f"Saved aggregate comparison to {args.output}")


if __name__ == "__main__":
    main()