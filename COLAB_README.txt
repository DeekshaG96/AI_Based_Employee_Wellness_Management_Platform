MoodMentor Milestone 2: GoEmotions model training and evaluation
==============================================================

This bundle contains the GoEmotions simplified train/dev/test splits, which include Reddit-derived text. Upload the dataset to Colab only if the dataset terms and your institute guidelines permit it. Never upload MoodMentor users' private journal entries.

1. In Colab, select a GPU runtime if available.
2. Upload and extract this ZIP into /content/MoodMentor.
3. Run these cells in order:

   %cd /content/MoodMentor
   !pip -q install "transformers>=4.38,<5"
   !python -u scripts/prepare_goemotions_csv.py

Train both encoders with identical data, seed, and settings. Each uses the full 43,410-row training split and selects its checkpoint by validation macro F1 at the fixed 0.5 threshold.

   !python -u scripts/train_goemotions_model.py --base-model distilbert/distilbert-base-uncased --epochs 3 --batch-size 8 --max-length 64 --threshold 0.5 --positive-weight-cap 8 --seed 42 --output-dir .venv-training/goemotions-distilbert --metrics-output .venv-training/goemotions-distilbert-metrics.json
   !python -u scripts/train_goemotions_model.py --base-model bert-base-uncased --epochs 3 --batch-size 8 --max-length 64 --threshold 0.5 --positive-weight-cap 8 --seed 42 --output-dir .venv-training/goemotions-bert --metrics-output .venv-training/goemotions-bert-metrics.json
   !python -m scripts.compare_training_runs .venv-training/goemotions-distilbert-metrics.json .venv-training/goemotions-bert-metrics.json --output .venv-training/goemotions-model-comparison.json

After training, calibrate each model on validation and score its untouched test split. The test labels do not determine thresholds.

   !python -u scripts.calibrate_goemotions.py --model .venv-training/goemotions-distilbert --output .venv-training/goemotions-distilbert-calibration.json
   !python -u scripts.calibrate_goemotions.py --model .venv-training/goemotions-bert --output .venv-training/goemotions-bert-calibration.json

Download the chosen model and aggregate evidence. Keep the BERT/DistilBERT comparison and calibration JSON files for the mentor review.

   import shutil
   from google.colab import files
   shutil.make_archive('/content/goemotions-distilbert-candidate', 'zip', root_dir='.venv-training/goemotions-distilbert')
   files.download('/content/goemotions-distilbert-candidate.zip')
   files.download('.venv-training/goemotions-distilbert-metrics.json')
   files.download('.venv-training/goemotions-distilbert-calibration.json')
   files.download('.venv-training/goemotions-model-comparison.json')

Extract the selected model ZIP into models/goemotions-distilbert on the local MoodMentor machine, then restart the local API. Do not present a limited --limit run as full-dataset training. The milestone's 0.82 macro-F1 value is a target, not a guaranteed result. GoEmotions is Reddit-derived and is not a validated employee-wellbeing dataset.

Use docs/MILESTONE_2_EXECUTION.md for the SemEval binary evaluation, ISEAR external holdout, privacy notes, and live demo checklist.