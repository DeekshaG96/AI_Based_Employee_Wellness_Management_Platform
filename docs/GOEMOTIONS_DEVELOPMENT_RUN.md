# GoEmotions development training record

**Run date:** 2026-10-30
**Dataset:** Google Research GoEmotions, simplified `agree-2` annotations
**Base model:** `distilbert/distilbert-base-uncased`
**Task:** 28-label multi-label emotion classification

## Dataset layout

The project keeps separate `train.csv`, `validation.csv`, and `test.csv` files under `data/goemotions/`. The files contain 43,410, 5,426, and 5,427 records, respectively. The local training run below used only the first 2,000 records from each split so it could show the complete training and evaluation workflow on a CPU.

## Development run

- Training records: 2,000
- Validation records: 2,000
- Held-out test records: 2,000
- Epochs: 1
- Optimizer updates: 250
- Batch size: 8
- Maximum tokens: 64
- Decision threshold: 0.25
- Positive class weight cap: 8
- Device: CPU
- Elapsed time: 859 seconds
- Final mean training loss: 0.5193

## Measured metrics

| Split | Micro F1 | Macro F1 |
|---|---:|---:|
| Validation subset | 0.2617 | 0.0322 |
| Held-out test subset | 0.2752 | 0.0342 |

The test subset micro precision was 0.2170 and recall was 0.3761. The very low macro F1 and per-label results show that this short, limited-data run does not learn the full set of emotions reliably. It strongly favors a small number of labels. These are development diagnostics only, not a claim of full-dataset model quality.

## Checkpoint decision

The checkpoint is saved locally at `.venv-training/goemotions-balanced-250/` and is ignored by Git. It was deliberately not copied to `models/goemotions-distilbert/`, so the application will not load this undertrained candidate. The aggregate metrics are saved alongside it at `.venv-training/goemotions-balanced-250-metrics.json`.

For an app-ready fine-tuned model, run the full training job on all available records using a CUDA GPU (for example, through the included `goemotions_milestone2_colab_bundle.zip`), then inspect full validation and test metrics before placing the model in `models/goemotions-distilbert/`. GoEmotions contains Reddit-derived text and should be treated as a general benchmark, not a validated employee-wellbeing model.
## Follow-up CPU pilot: 10,000-example DistilBERT run (2026-11-04)

A deterministic random sample of 10,000 examples from the 43,410-record GoEmotions training split was used for one epoch. The complete validation split (5,426 examples) selected the checkpoint, and the complete held-out test split (5,427 examples) was evaluated once. The run used `distilbert/distilbert-base-uncased`, 28 multi-label outputs, batch size 16, maximum sequence length 64, learning rate `2e-5`, threshold 0.5, seed 42, CPU-only PyTorch, and took 1,950.6 seconds. Final mean training loss was 0.4381.

| Evaluation | Examples | Macro F1 | Micro F1 |
|---|---:|---:|---:|
| Validation, fixed threshold 0.5 | 5,426 | 0.1741 | 0.4405 |
| Held-out test, fixed threshold 0.5 | 5,427 | 0.1692 | 0.4419 |

For context, the existing `BruceIC/emoticare` model scored macro F1 0.4019 on validation and 0.4142 on test at the same fixed 0.5 threshold; its validation-calibrated test macro F1 is 0.4872. The pilot is substantially weaker and is not calibrated. Its checkpoint remains local at `.venv-training/goemotions-10k-1epoch/`; it was not placed in `models/goemotions-distilbert/` or selected by the app. The saved aggregate and per-label metrics are in `docs/goemotions-distilbert-10k-cpu-pilot.json`. No source comments or journal entries are included.

This is a limited CPU pilot, not full GoEmotions fine-tuning or a workplace-validated model. The 0.82 macro-F1 target remains unmet. A full-data, multi-epoch GPU run and matched evaluation are still required before model replacement.
