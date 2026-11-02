# MoodMentor Milestone 2: execution and evidence

## What the milestone asks for

The mentor deck defines Milestone 2 for Weeks 3–4. Its measurable criteria are:

- Fine-tune and compare BERT and DistilBERT on GoEmotions.
- Predict at least six emotion categories with confidence scores.
- Report macro F1, accuracy, precision, and recall, including per-class analysis.
- Target at least 0.82 GoEmotions validation macro F1.
- Target at least 0.90 polarity accuracy on a SemEval binary subset.
- Evaluate against a held-out ISEAR split.
- Demonstrate a mixed-emotion input, baseline versus transformer output, one error case, and a recovery path.

The SemEval paper describes multiple affect tasks. The deck does not identify which SemEval subtask or exact positive/negative export it means. Confirm that dataset definition with the mentor before comparing scores. Source: [SemEval-2018 Task 1 paper](https://aclanthology.org/S18-1001/).

## Verified project state

| Requirement | Evidence in this project | Status |
| --- | --- | --- |
| GoEmotions source and splits | `archive (1)/data` has train/dev/test; `data/goemotions` contains 43,410 train, 5,426 validation, and 5,427 test rows with all 28 labels | Ready |
| Multi-label fine-grained inference | The local API pipeline returns independent GoEmotions scores and identifies the active estimator | Implemented |
| VADER baseline | VADER scores and polarity are returned alongside emotion analysis | Implemented |
| Training workflow | `scripts/train_goemotions_model.py` supports full splits; checkpoint selection now follows validation macro F1 | Ready to run |
| Validation-only calibration and confusion counts | `scripts/calibrate_goemotions.py` selects thresholds on validation, then reports untouched test metrics and per-class TP/FP/FN/TN | Implemented |
| Full-data fine-tuning evidence | The recorded local run used 2,000 rows from each split, one epoch, and 250 optimizer updates on CPU | Not complete |
| GoEmotions target | The public checkpoint's measured held-out test macro F1 is 0.2980 at threshold 0.5. The limited local run's validation macro F1 is 0.0322. Neither meets the 0.82 target, and neither is a result from full-data fine-tuning | Not met |
| SemEval polarity score | No SemEval file is present. `scripts/evaluate_semeval_binary.py` now evaluates an explicitly labelled positive/negative export with VADER and reports accuracy, precision, recall, F1, and a confusion matrix | Awaiting the agreed dataset export and run |
| ISEAR holdout | No ISEAR file is present. `scripts/evaluate_isear.py` evaluates a separate local test file and saves aggregate results only | Awaiting licensed data and run |
| Streamlit hosted demo | The free hosted showcase uses a lexical fallback to stay within its memory limit. Its scores are marked as heuristics, not model confidence. The transformer comparison must use the full local app/model | Demo limitation |

The source GoEmotions profile contains 54,263 examples: 43,410 train, 5,426 development, and 5,427 held-out test. The full local model weights are excluded from Git. Do not commit model checkpoints, raw datasets, or user journal text.

## Reproducible run guide

Run full training on a GPU machine such as Colab. The current local Python environment is CPU-only, and the recorded 250-step candidate is too small to represent full-dataset training.

### 1. Prepare and fine-tune DistilBERT

From the repository root, install the optional model dependencies and prepare the local splits:

```powershell
python -m pip install -e ".[ml]"
python -m scripts.prepare_goemotions_csv
python -m scripts.train_goemotions_model --base-model distilbert/distilbert-base-uncased --epochs 3 --batch-size 8 --max-length 64 --threshold 0.5 --positive-weight-cap 8 --seed 42 --output-dir models/goemotions-distilbert --metrics-output docs/goemotions-distilbert-training.json
```

The trainer uses the training split for updates, selects checkpoints using validation macro F1 at a fixed threshold, and evaluates the selected checkpoint on the held-out test split. Do not use `--limit` for final results.

### 2. Compare a BERT-base run

Train a second run on the same splits and seed. Keep the artifacts separate:

```powershell
python -m scripts.train_goemotions_model --base-model bert-base-uncased --epochs 3 --batch-size 8 --max-length 64 --threshold 0.5 --positive-weight-cap 8 --seed 42 --output-dir models/goemotions-bert --metrics-output docs/goemotions-bert-training.json
```

Compare macro F1 first, then micro F1, precision, recall, per-label support, runtime, and memory. The two runs must use the same data splits and evaluation policy. Produce a compact aggregate table with: `python -m scripts.compare_training_runs docs/goemotions-distilbert-training.json docs/goemotions-bert-training.json --output docs/goemotions-model-comparison.json`.

### 3. Calibrate thresholds without test leakage

After training, calibrate each label from validation only and score the held-out test split:

```powershell
python -m scripts.calibrate_goemotions --model models/goemotions-distilbert --output docs/goemotions-distilbert-calibration.json
```

Repeat for `models/goemotions-bert`. The report includes fixed-threshold and calibrated results, per-label confusion counts, and the validation-selected thresholds. Calibration can overfit; include both validation and untouched test values in the mentor review.

### 4. Evaluate the SemEval binary subset

Export the mentor-approved SemEval test subset locally with columns `text` and `label`, and labels such as `positive` and `negative`. Keep the file out of Git. Then run:

```powershell
python -m scripts.evaluate_semeval_binary data/semeval/test-binary.csv --text-column text --label-column label --positive-label positive --negative-label negative --output docs/semeval-vader-test.json
```

If the file uses numeric labels, set `--positive-label` and `--negative-label` to those exact values. The evaluator ignores other labels and reports how many it excluded. Use an untouched test export. A tuned threshold must be selected on a separate validation split, never on this test file.

### 5. Evaluate ISEAR as an external holdout

Use a local, untouched ISEAR test file with one text column and one label column. The evaluator maps GoEmotions signals to ISEAR's seven categories; shame-to-embarrassment and guilt-to-remorse are approximate mappings, so report this as external transfer rather than same-taxonomy accuracy. Verify the dataset license and split before use. Source label reference: [ISEAR dataset card](https://huggingface.co/datasets/savalera/isear-from-original).

```powershell
python -m scripts.evaluate_isear data/isear/test.csv --model models/goemotions-distilbert --text-column text --label-column emotion --output docs/isear-transfer-test.json
```

The evaluator reports accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix without storing source text.

## Live demo sequence

1. Sign in to the local full application and state that the analysis is an optional reflection aid, not a diagnosis or employee-performance score.
2. Enter a mixed example: “I feel proud that I finished the work, but worried about tomorrow.”
3. Show VADER's polarity baseline and the transformer's independent GoEmotions scores. Explain that these scores do not have to sum to 100% and that the strongest label can be wrong.
4. Show the estimator name. If the local transformer cannot load, the system labels the lexical fallback as a heuristic and avoids calling it model confidence.
5. Show the recommendation, then the history chart after a saved check-in.
6. Demonstrate an error case such as “I am not happy.” Discuss negation and context as limitations; recover by inviting the user to describe the nuance in their own words and treat the model output as a prompt, not a fact.
7. Show the privacy/consent boundary and account history deletion control. Never use real private journal entries in screenshots or public reports.

For the mentor's question about accuracy: a majority-class or frequent-class bias can make overall accuracy look high while rare emotions perform poorly. Macro F1 gives each label equal weight. The existing GoEmotions test report has labels with zero F1, so per-class results and support matter more than one headline score.

## Evidence to capture after GPU and data runs

- Full training commands, seeds, runtime, model names, and split sizes.
- Validation and held-out test JSON reports for both encoders.
- Per-class metrics/confusion counts and threshold policy.
- SemEval and ISEAR aggregate reports, with dataset source, split, and label mapping.
- A live mixed-emotion walkthrough and one documented error/recovery example.

Do not claim the numeric targets as passed until the corresponding untouched validation/test artifacts show them. The currently available evidence does not meet the 0.82 GoEmotions macro-F1 target or establish the SemEval 0.90 target.