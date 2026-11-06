# Milestone 2 GPU runbook

This runbook is the reproducible path for improving GoEmotions macro-F1 with
the attached `archive (1)` dataset. It keeps the held-out test split untouched
until model selection and threshold calibration are complete.

## What “near 100%” means here

Near-perfect held-out performance is not a realistic or honest expectation for
GoEmotions: it is a noisy, multi-label Reddit dataset with imperfect human
agreement and rare classes. The objective is the highest reproducible macro-F1
under one fixed protocol, with per-label metrics and limitations reported.
Never tune thresholds on the held-out test split.

## GPU environment

Use a CUDA-enabled machine or Colab runtime. Keep the Downloads reference
folder read-only and do not commit its dataset, model weights, credentials,
database, or generated files.

```powershell
python -m pip install -e ".[ml,dev]"
```

Set these paths to the read-only attached assets:

```powershell
$data = "C:\Users\ganch\Downloads\MoodMentor\archive (1)\data"
$attachedModel = "C:\Users\ganch\Downloads\MoodMentor\models\goemotions-distilbert"
```

## DistilBERT fine-tuning

Run the full 43,410-example training split. Do not use `--limit` for milestone
evidence. The validation split selects the best epoch; the test split is scored
once after selection.

```powershell
python -m scripts.train_goemotions_model `
  --data-dir $data `
  --base-model $attachedModel `
  --epochs 3 `
  --batch-size 16 `
  --learning-rate 2e-5 `
  --max-length 64 `
  --threshold 0.5 `
  --positive-weight-cap 8 `
  --seed 42 `
  --output-dir .venv-training\attached-distilbert-full `
  --metrics-output .venv-training\attached-distilbert-full.json
```

If GPU memory is insufficient, reduce only `--batch-size` to 8 or 4. Keep the
seed, splits, learning rate, and evaluation policy unchanged when comparing
runs. Gradient accumulation is not currently enabled, so record any batch-size
change in the report.

## BERT comparison

Run a same-policy BERT baseline with a separate output directory:

```powershell
python -m scripts.train_goemotions_model `
  --data-dir $data `
  --base-model bert-base-uncased `
  --epochs 3 `
  --batch-size 16 `
  --learning-rate 2e-5 `
  --max-length 64 `
  --threshold 0.5 `
  --positive-weight-cap 8 `
  --seed 42 `
  --output-dir .venv-training\bert-full `
  --metrics-output .venv-training\bert-full.json
```

Compare only runs with the same split sizes, seed, threshold policy, and
non-null `limit` status:

```powershell
python -m scripts.compare_training_runs `
  .venv-training\attached-distilbert-full.json `
  .venv-training\bert-full.json `
  --output .venv-training\model-comparison.json
```

## Validation-only calibration

After each training run, choose per-label thresholds from validation and score
the untouched test split:

```powershell
python -m scripts.calibrate_goemotions `
  --data-dir $data `
  --model .venv-training\attached-distilbert-full `
  --batch-size 16 `
  --output .venv-training\attached-distilbert-calibration.json

python -m scripts.calibrate_goemotions `
  --data-dir $data `
  --model .venv-training\bert-full `
  --batch-size 16 `
  --output .venv-training\bert-calibration.json
```

The report's `validation_calibrated_metrics` is useful for model development,
but the milestone comparison must use
`held_out_test_calibrated_metrics`. Calibration can overfit validation; report
both fixed-threshold and calibrated values.

## Acceptance checklist

- `examples_per_split` is 43,410 / 5,426 / 5,427.
- `limit` is `null`.
- The model and `emotions.txt` contain the same 28 labels in the same order.
- Validation selects the checkpoint; test is not used for training or tuning.
- Both fixed and calibrated macro-F1, micro-F1, precision, recall, exact match,
  and per-label results are saved.
- The `0.82` target is called passed only if a fresh validation artifact
  reaches at least `0.82` under this policy.
- No SemEval `0.90` claim is made without the mentor-approved local export.

The current attached checkpoint evidence is documented in
`docs/MILESTONE_2_EXECUTION.md`; its best measured held-out calibrated
macro-F1 is `0.4872`, so it is not near 100% and is not presented as such.
