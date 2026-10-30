# GoEmotions dataset review

## Scope and source

MoodMentor's local NLP task uses the archived GoEmotions simplified agree-2 dataset under `archive (1)/data`. The archive contains tab-separated training, development, and test splits plus a 28-label name list. Each row has three fields: text, one or more comma-separated label IDs, and a comment ID. Label IDs use the order in `emotions.txt`.

The inspected splits contain **54,263 entries**: 43,410 training, 5,426 development, and 5,427 held-out test entries. The training split has 51,103 assigned labels across the 28 categories, averaging 1.177 labels per entry. Neutral is most frequent (14,219 training labels); grief is least frequent (77). This imbalance makes a single accuracy number insufficient, so the model evaluation reports macro and micro scores as well as per-label metrics.

The data quality scan found no malformed rows, empty text, repeated comment IDs within a split, or invalid label IDs. The raw archive is intentionally excluded from Git; these checks and the profile script do not print or save any source text.

## Multi-label handling

GoEmotions is multi-label: one text may have several emotion labels. MoodMentor therefore uses a multi-label DistilBERT model and evaluates every label independently. It retains all fine-grained signals for analysis and maps the strongest signal to a broad emotion family for recommendations. The independent model scores are not normalized to sum to 100%.

## Reproducible review and evaluation

Run `python -m scripts.analyze_goemotions` to inspect split sizes, label frequencies, malformed rows, empty text, duplicate IDs, and average labels per example. The script never prints or saves source text. Run `python -m scripts.evaluate_goemotions_model --output docs/goemotions-distilbert-evaluation.json` to evaluate the pretrained model on the held-out test split. It reports per-label and macro/micro precision, recall, and F1, plus exact-match accuracy at the 0.5 threshold. The measured held-out result was micro precision **0.8618**, recall **0.3321**, F1 **0.4795**, macro F1 **0.2980**, and exact-match accuracy **0.3243**. This is a conservative model with weak recall on several rare classes; use its results as reflective signals, and calibrate thresholds on the development split before relying on it for decisions. The script saves aggregate metrics only, with runtime versions recorded.

The local archive can also be read from another location by passing its `data` directory with `--data-dir`.

## Interpretation and limitations

GoEmotions was collected from English-language Reddit comments, not employee journals. Its label balance and writing style may not represent a workplace population. Emotion classification is an uncertain language signal; sarcasm, context, dialect, and cultural differences can affect results. MoodMentor is for reflection and learning, not diagnosis or workplace decisions. Do not include private journal text in public dataset reports.

Source: [GoEmotions dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions). Model: [GoEmotions DistilBERT checkpoint](https://huggingface.co/bhadresh-savani/bert-base-go-emotion).
## Training a local MoodMentor checkpoint

Run python -m scripts.train_goemotions_model to fine-tune the DistilBERT base model on the archived training split. The script uses the development split for checkpoint selection and the test split only for final reporting. It writes model weights under models/goemotions-distilbert and aggregate metrics to docs/goemotions-distilbert-training.json. Raw comment text is not saved in the output. The local Compose API automatically uses the fine-tuned artifact when it exists.
