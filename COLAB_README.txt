MoodMentor GoEmotions full-data fine-tuning in Google Colab
==========================================================

This ZIP contains the GoEmotions simplified train/dev/test files, which include Reddit-derived text. The optional cloud workflow sends those dataset files to Google Colab when you upload the ZIP. Use it only if that is permitted by the dataset terms and your project or institute guidelines. Never upload MoodMentor users' private journal entries.

1. In Colab, select a GPU runtime if available.
2. Upload and extract this ZIP into /content/MoodMentor.
3. Run these cells in order:

   %cd /content/MoodMentor
   !pip -q install "transformers>=4.38,<5"
   !python -u scripts/prepare_goemotions_csv.py
   !python -u scripts/train_goemotions_model.py --epochs 3 --batch-size 8 --max-length 64 --threshold 0.25 --positive-weight-cap 8 --output-dir .venv-training/goemotions-full --metrics-output .venv-training/goemotions-full-metrics.json

The preparation step creates the CSV files expected by the trainer. Full training uses all 43,410 training rows, evaluates the validation split after each epoch, retains the best validation checkpoint, and scores it against the held-out test split. The metrics JSON contains aggregate scores only. Runtime can vary with the selected GPU.

4. Download the candidate model and aggregate report with:

   import shutil
   from google.colab import files
   shutil.make_archive('/content/goemotions-full-candidate', 'zip', root_dir='.venv-training/goemotions-full')
   files.download('/content/goemotions-full-candidate.zip')
   files.download('.venv-training/goemotions-full-metrics.json')

5. Review the validation and test metrics. If the candidate is suitable, extract the model ZIP into models/goemotions-distilbert on the local MoodMentor machine, then restart the local API so it can load the fine-tuned model.

Do not present a limited --limit run as full-dataset training. GoEmotions is Reddit-derived and is not a validated employee-wellbeing dataset; treat its metrics as a general research benchmark.