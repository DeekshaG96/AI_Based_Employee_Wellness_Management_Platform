# Streamlit Community Cloud showcase

Entrypoint: `streamlit_cloud/app.py`
Dependencies: `streamlit_cloud/requirements.txt`
Suggested Python runtime: 3.12

This lightweight showcase runs MoodMentor's existing VADER and preprocessing pipeline inside Streamlit. It deliberately disables the large transformer download and uses the clearly labeled keyword fallback for emotion signals. It does not call FastAPI, PostgreSQL, or Render.

Check-ins and imported file rows are held only in each browser session's Streamlit state. There is no account system or persistent database in this deployment. Visitors' text is processed by the hosted app, so the page warns them not to submit sensitive or identifying details. The full local Docker version remains available for authenticated accounts and persistent PostgreSQL history.

## Deploy

1. Use the prepared GitHub branch named `streamlit-cloud` (it avoids the existing `main`-branch workflows).
2. At [share.streamlit.io](https://share.streamlit.io), create an app from that repository and the `streamlit-cloud` branch.
3. Set the entrypoint to `streamlit_cloud/app.py` and select Python 3.12 in Advanced settings.
4. Leave secrets empty; this showcase has no API URL, database password, or external model key.
5. Deploy, then check the app logs if dependency or NLTK resource setup reports an error.

The requirements file is next to the entrypoint, which lets Community Cloud install only the smaller showcase dependencies instead of the full local API and transformer stack.