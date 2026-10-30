"""Export local GoEmotions TSV splits as privacy-ignored CSV files."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "archive (1)" / "data"
DEFAULT_OUTPUT = ROOT / "data" / "goemotions"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    labels = [line.strip() for line in (args.source_dir / "emotions.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 28:
        raise SystemExit(f"Expected 28 GoEmotions labels; found {len(labels)}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (("train.tsv", "train.csv"), ("dev.tsv", "validation.csv"), ("test.tsv", "test.csv")):
        count = 0
        with (args.source_dir / source_name).open(encoding="utf-8", newline="") as source, (args.output_dir / target_name).open("w", encoding="utf-8", newline="") as target:
            reader = csv.reader(source, delimiter=chr(9))
            writer = csv.writer(target)
            writer.writerow(("text", "labels", "label_names"))
            for row_number, row in enumerate(reader, start=1):
                if len(row) != 3 or not row[0].strip():
                    raise ValueError(f"Malformed row {row_number} in {source_name}.")
                ids = [int(value.strip()) for value in row[1].split(",") if value.strip()]
                if not ids or min(ids) < 0 or max(ids) >= len(labels):
                    raise ValueError(f"Invalid label IDs in {source_name}, row {row_number}.")
                writer.writerow((row[0], ",".join(map(str, ids)), ",".join(labels[index] for index in ids)))
                count += 1
        print(f"{target_name}: {count:,} examples")
    (args.output_dir / "emotions.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# GoEmotions splits\n\n"
        "Prepared from the local Google Research GoEmotions simplified agree-2 archive for MoodMentor.\n\n"
        "- train.csv: 43,410 examples\n- validation.csv: 5,426 examples\n- test.csv: 5,427 examples\n"
        "- emotions.txt: the 28-label ordering used by the model\n\n"
        "Each CSV has text, labels, and label_names columns. labels and label_names are comma-separated lists in correctly quoted CSV fields because each text can have multiple labels. GoEmotions is multi-label; do not reduce each record to one emotion.\n\n"
        "These files contain Reddit-derived text. Keep them local and do not commit or upload them. Training and evaluation scripts write only aggregate metrics, not input texts.\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
