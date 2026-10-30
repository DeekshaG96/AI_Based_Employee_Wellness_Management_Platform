"""Profile GoEmotions TSV splits without printing or saving journal/comment text."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "goemotions"
SPLITS = ("train.csv", "validation.csv", "test.csv")


def profile_split(data_dir: Path, split_name: str, labels: list[str]) -> dict[str, Any]:
    path = data_dir / split_name
    label_counts: Counter[str] = Counter()
    malformed_rows = empty_text_rows = invalid_label_ids = 0
    ids: set[str] = set()
    duplicate_ids = 0
    total_labels = 0
    row_count = 0

    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    has_record_ids = path.suffix.lower() != ".csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter=delimiter), start=1):
            if row and row_number == 1 and row[0].strip().lower() == "text":
                continue
            row_count += 1
            if len(row) != 3:
                malformed_rows += 1
                continue
            text, raw_labels, record_id = row
            empty_text_rows += not bool(text.strip())
            if has_record_ids and record_id in ids:
                duplicate_ids += 1
            if has_record_ids:
                ids.add(record_id)
            row_ids = [part.strip() for part in raw_labels.split(",") if part.strip()]
            total_labels += len(row_ids)
            for raw_id in row_ids:
                try:
                    index = int(raw_id)
                    if index < 0 or index >= len(labels):
                        invalid_label_ids += 1
                    else:
                        label_counts[labels[index]] += 1
                except ValueError:
                    invalid_label_ids += 1

    return {
        "rows": row_count,
        "malformed_rows": malformed_rows,
        "empty_text_rows": empty_text_rows,
        "duplicate_comment_ids": duplicate_ids if has_record_ids else None,
        "invalid_label_ids": invalid_label_ids,
        "average_labels_per_row": round(total_labels / row_count, 3) if row_count else 0,
        "label_counts": dict(sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    labels_path = args.data_dir / "emotions.txt"
    if not labels_path.is_file():
        raise SystemExit(f"Could not find {labels_path}. Pass --data-dir to the archive data folder.")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 28:
        raise SystemExit(f"Expected 28 emotion labels, found {len(labels)}.")
    output = {
        "dataset": "GoEmotions simplified agree-2 split",
        "data_dir": str(args.data_dir),
        "label_count": len(labels),
        "labels": labels,
        "splits": {
            name.removesuffix(".tsv"): profile_split(args.data_dir, name, labels)
            for name in SPLITS
        },
        "privacy": "Raw text is analyzed in memory and is never printed or saved by this script.",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
