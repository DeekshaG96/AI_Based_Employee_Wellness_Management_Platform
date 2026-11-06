# Streamlit Community Cloud showcase

Entrypoint: streamlit_cloud/app.py
Dependencies: streamlit_cloud/requirements.txt
Suggested Python runtime: 3.12

The showcase runs VADER and lightweight emotion cues inside Streamlit by default. If HF_TOKEN is configured, a guest can opt an individual journal entry into Hugging Face Inference with SamLowe/roberta-base-go_emotions (28 GoEmotions labels). Guest history and optional self-ratings are temporary and held in the Streamlit session.

## Guest mode and account mode

The app starts in guest mode when no account API is configured. Guest check-ins can be cleared in the sidebar and are not durable. Do not enter sensitive or identifying information in guest mode.

For real sign-in, registration, sign-out, and persistent history, connect this app to a deployed MoodMentor FastAPI service backed by a persistent database. In Streamlit Community Cloud, open the app's **Settings → Secrets** and add the following line:

    API_URL = "https://your-moodmentor-api.example.com"

Use your own HTTPS API address. The API must expose MoodMentor's /auth/register, /auth/login, /auth/logout, /analyze, and /history/{user_id} endpoints and use a persistent database. The app does not contain or store account passwords. Authentication and saved journal history are handled by the API. If API_URL is missing, account access explains what is needed and guest mode remains available. No particular hosting provider is required.

Account check-ins store the journal text, analysis, and recommendations in that account's database. Optional self-rated mood, sleep, energy, and context annotations are currently guest-only because the API schema does not persist those fields. Sign out revokes the server session and clears the active browser session when the API is reachable.

## Model evaluation

The Model evaluation tab presents the saved GoEmotions benchmark, threshold comparison, and definitions for precision, recall, F1, exact-match accuracy, and Hamming loss. It reads docs/goemotions-bruceic-calibration.json. That benchmark evaluates BruceIC/emoticare, not the optional live SamLowe/roberta-base-go_emotions model, and does not establish accuracy on workplace journal text.

## Presentation-ready demo path

Use the **Guided demo example** selector in the Mood journal tab to load synthetic
examples without entering personal information:

1. **Mixed emotion:** “I feel proud that I finished the work, but worried about tomorrow.”
2. **Negation edge case:** “I am not happy with how today went.”
3. **Recovery moment:** “I felt overwhelmed earlier, then a short walk helped me feel calmer.”

After each analysis, show the VADER tone, the named estimator, the emotion
signals, and the supportive next steps. Explain that heuristic scores are
relative cue shares, transformer scores are independent multi-label signals,
and neither is a diagnosis. Then use **Trends & history** to show the separate
text-tone and self-reported-mood views. These synthetic examples are intended
for demonstrations; do not use real private journal text in a public showcase.

## Deploy

1. Create or select the Streamlit app connected to this repository and the streamlit-cloud branch.
2. Set the entrypoint to streamlit_cloud/app.py and select Python 3.12 in Advanced settings.
3. Add API_URL to Settings → Secrets only if a persistent MoodMentor API is available. Add HF_TOKEN only if you want to offer optional guest-mode transformer inference. Never commit either secret.
4. Deploy, then review the app logs if dependency installation reports an error.
