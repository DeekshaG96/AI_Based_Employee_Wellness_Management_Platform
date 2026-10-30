"""Fine-tune a multi-label DistilBERT model on the local GoEmotions splits.

Only aggregate evaluation metrics are written. Raw Reddit comments stay in the
local dataset folder and are never copied into the model artifact or report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "goemotions"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "goemotions-distilbert"
DEFAULT_METRICS_PATH = ROOT / "docs" / "goemotions-distilbert-training.json"


def read_split(data_dir: Path, split_name: str, labels: list[str]) -> tuple[list[str], list[list[float]]]:
    texts: list[str] = []
    targets: list[list[float]] = []
    split_path = data_dir / split_name
    delimiter = "," if split_path.suffix.lower() == ".csv" else "\t"
    with split_path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter=delimiter), start=1):
            if row and row_number == 1 and row[0].strip().lower() == "text":
                continue
            if len(row) != 3 or not row[0].strip():
                raise ValueError(f"Malformed GoEmotions record in {split_path} at row {row_number}.")
            try:
                label_ids = {int(value.strip()) for value in row[1].split(",") if value.strip()}
            except ValueError as exc:
                raise ValueError(f"Invalid label ID in {split_path} at row {row_number}.") from exc
            if not label_ids or min(label_ids) < 0 or max(label_ids) >= len(labels):
                raise ValueError(f"Out-of-range label ID in {split_path} at row {row_number}.")
            target = [0.0] * len(labels)
            for label_id in label_ids:
                target[label_id] = 1.0
            texts.append(row[0])
            targets.append(target)
    if not texts:
        raise ValueError(f"No records found in {split_path}.")
    return texts, targets


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EncodedGoEmotions:
    def __init__(self, encodings: dict[str, list[list[int]]], targets: list[list[float]]) -> None:
        self.encodings = encodings
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.targets[index]
        return item


def make_collator(tokenizer: Any):
    import torch

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        labels = torch.tensor([feature.pop("labels") for feature in features], dtype=torch.float32)
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return collate


def calculate_metrics(probabilities: Any, targets: Any, labels: list[str], threshold: float) -> dict[str, Any]:
    import torch

    predicted = probabilities >= threshold
    actual = targets >= 0.5
    true_positive = (predicted & actual).sum(dim=0).to(torch.float64)
    false_positive = (predicted & ~actual).sum(dim=0).to(torch.float64)
    false_negative = (~predicted & actual).sum(dim=0).to(torch.float64)

    per_label: dict[str, dict[str, float | int]] = {}
    precisions = []
    recalls = []
    f1_scores = []
    for index, label in enumerate(labels):
        tp = true_positive[index]
        fp = false_positive[index]
        fn = false_negative[index]
        precision = tp / (tp + fp).clamp(min=1)
        recall = tp / (tp + fn).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_label[label] = {
            "precision": round(float(precision.item()), 4),
            "recall": round(float(recall.item()), 4),
            "f1": round(float(f1.item()), 4),
            "support": int(actual[:, index].sum().item()),
        }

    micro_precision = true_positive.sum() / (true_positive.sum() + false_positive.sum()).clamp(min=1)
    micro_recall = true_positive.sum() / (true_positive.sum() + false_negative.sum()).clamp(min=1)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall).clamp(min=1e-12)
    return {
        "examples": int(targets.shape[0]),
        "threshold": threshold,
        "micro_precision": round(float(micro_precision.item()), 4),
        "micro_recall": round(float(micro_recall.item()), 4),
        "micro_f1": round(float(micro_f1.item()), 4),
        "macro_precision": round(float(torch.stack(precisions).mean().item()), 4),
        "macro_recall": round(float(torch.stack(recalls).mean().item()), 4),
        "macro_f1": round(float(torch.stack(f1_scores).mean().item()), 4),
        "exact_match_accuracy": round(
            float((predicted == actual).all(dim=1).float().mean().item()), 4
        ),
        "per_label": per_label,
    }


def evaluate(model: Any, loader: Any, device: Any, labels: list[str], threshold: float) -> dict[str, Any]:
    import torch

    model.eval()
    all_probabilities = []
    all_targets = []
    with torch.inference_mode():
        for batch in loader:
            targets = batch.pop("labels")
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**model_inputs).logits
            all_probabilities.append(torch.sigmoid(logits).cpu())
            all_targets.append(targets)
    return calculate_metrics(
        torch.cat(all_probabilities, dim=0),
        torch.cat(all_targets, dim=0),
        labels,
        threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--base-model", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--positive-weight-cap", type=float, default=8.0, help="Cap square-root class weights for sparse GoEmotions labels.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Optional small run; omit to use every split record.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        raise SystemExit("--epochs, --batch-size, and --max-length must be positive.")
    if not 0.0 < args.threshold < 1.0 or not 0.0 <= args.warmup_ratio < 1.0:
        raise SystemExit("--threshold must be in (0, 1) and --warmup-ratio in [0, 1).")
    if args.positive_weight_cap < 1.0:
        raise SystemExit("--positive-weight-cap must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be a positive integer.")

    try:
        import torch
        import transformers
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit(
            "Install the local ML dependencies first: python -m pip install -r requirements.txt"
        ) from exc

    labels = [
        label.strip()
        for label in (args.data_dir / "emotions.txt").read_text(encoding="utf-8").splitlines()
        if label.strip()
    ]
    if len(labels) != 28 or len(set(labels)) != len(labels):
        raise SystemExit("Expected the GoEmotions 28-label list in emotions.txt.")

    set_seed(args.seed)
    texts_by_split: dict[str, list[str]] = {}
    targets_by_split: dict[str, list[list[float]]] = {}
    split_files = {
        "train.tsv": "train.csv",
        "dev.tsv": "validation.csv",
        "test.tsv": "test.csv",
    }
    if not all((args.data_dir / filename).is_file() for filename in split_files.values()):
        split_files = {name: name for name in ("train.tsv", "dev.tsv", "test.tsv")}
    for split_name, source_name in split_files.items():
        texts, targets = read_split(args.data_dir, source_name, labels)
        if args.limit is not None:
            texts, targets = texts[: args.limit], targets[: args.limit]
        texts_by_split[split_name] = texts
        targets_by_split[split_name] = targets
        print(f"Loaded {source_name}: {len(texts):,} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    encoded: dict[str, EncodedGoEmotions] = {}
    for split_name, texts in texts_by_split.items():
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        encoded[split_name] = EncodedGoEmotions(tokenized, targets_by_split[split_name])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    print(f"Training device: {device}")

    collator = make_collator(tokenizer)
    train_loader = DataLoader(
        encoded["train.tsv"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        encoded["dev.tsv"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        encoded["test.tsv"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )
    model.config.emotion_threshold = args.threshold
    model.config.emotion_positive_weight_cap = args.positive_weight_cap
    model.to(device)
    train_targets = torch.tensor(targets_by_split["train.tsv"], dtype=torch.float32)
    positive_counts = train_targets.sum(dim=0).to(device)
    negative_counts = len(train_targets) - positive_counts
    positive_weights = torch.sqrt((negative_counts + 1.0) / (positive_counts + 1.0))
    positive_weights = positive_weights.clamp(min=1.0, max=args.positive_weight_cap)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_dev_f1 = -1.0
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels_batch = batch.pop("labels").to(device)
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(**model_inputs).logits
                    loss = loss_function(logits, labels_batch)
            else:
                logits = model(**model_inputs).logits
                loss = loss_function(logits, labels_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach().cpu().item())
            if step % 25 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"step {step}/{len(train_loader)} "
                    f"mean_loss={total_loss / step:.4f}"
                )

        dev_metrics = evaluate(model, dev_loader, device, labels, args.threshold)
        print(
            f"Epoch {epoch} validation: "
            f"micro_f1={dev_metrics['micro_f1']:.4f}, "
            f"macro_f1={dev_metrics['macro_f1']:.4f}"
        )
        if dev_metrics["micro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["micro_f1"]
            best_epoch = epoch
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    test_metrics = evaluate(best_model, test_loader, device, labels, args.threshold)
    training_metrics = {
        "dataset": "GoEmotions simplified agree-2",
        "dataset_source": "Google Research GoEmotions",
        "label_count": len(labels),
        "labels": labels,
        "multi_label": True,
        "base_model": args.base_model,
        "fine_tuned_model_path": str(args.output_dir),
        "best_epoch_by_validation_micro_f1": best_epoch,
        "validation_micro_f1": round(best_dev_f1, 4),
        "test_metrics": test_metrics,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "decision_threshold": args.threshold,
            "positive_weight_cap": args.positive_weight_cap,
            "seed": args.seed,
            "examples_per_split": {split_files[name]: len(texts) for name, texts in texts_by_split.items()},
            "limit": args.limit,
            "device": str(device),
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seconds": round(time.monotonic() - started, 1),
        "privacy": "Only aggregate metrics are recorded; source comments are not saved.",
        "limitations": "Reddit text may not represent private journal or workplace writing.",
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(training_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(training_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
