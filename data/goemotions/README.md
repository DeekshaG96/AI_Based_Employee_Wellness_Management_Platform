# GoEmotions splits

Prepared from the local Google Research GoEmotions simplified agree-2 archive for MoodMentor.

- train.csv: 43,410 examples
- validation.csv: 5,426 examples
- test.csv: 5,427 examples
- emotions.txt: the 28-label ordering used by the model

Each CSV has text, labels, and label_names columns. labels and label_names are comma-separated lists in correctly quoted CSV fields because each text can have multiple labels. GoEmotions is multi-label; do not reduce each record to one emotion.

These files contain Reddit-derived text. Keep them local and do not commit or upload them. Training and evaluation scripts write only aggregate metrics, not input texts.
