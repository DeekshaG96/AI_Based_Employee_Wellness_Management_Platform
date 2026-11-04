# Saved MoodMentor accounts (Streamlit + Vercel + Neon)

This setup gives the Streamlit showcase persistent accounts and journal history without Render:

- Streamlit Community Cloud hosts the website.
- Vercel runs the existing FastAPI account and analysis service.
- Neon stores accounts, saved check-ins, and recommendations in PostgreSQL.

The free hosting tiers have usage limits and can sleep or scale down. Vercel Hobby is intended for personal, non-commercial projects; Neon Free has monthly compute and storage limits. This is suitable for a student project demo, not a clinical or production service.

## One-time setup

1. **Create a Neon database.** Sign in at [Neon](https://neon.com/), create a PostgreSQL project, and copy its pooled connection string. Use the pooled URL (the hostname contains `-pooler`) and keep it private.
2. **Create the API deployment.** Sign in at [Vercel](https://vercel.com/) with the GitHub account that owns the project. Import `DeekshaG96/AI_Based_Employee_Wellness_Management_Platform`, select the `streamlit-cloud` branch, and leave the project root directory empty (repository root).
3. **Add the database address before deploying.** In the Vercel project’s Environment Variables, add `DATABASE_URL` and paste the Neon pooled connection string. Select Production (and Preview too if you will use preview deployments). Save it, then deploy. Do not put this value in GitHub or this chat.
4. **Check the API.** Open `https://YOUR-VERCEL-PROJECT.vercel.app/health`. It should return `{"status":"ok","database":"connected"}`. `/docs` opens the API reference.
5. **Connect the website.** In Streamlit Community Cloud, open the MoodMentor app’s **Settings → Secrets** and add:

   ```toml
   API_URL = "https://YOUR-VERCEL-PROJECT.vercel.app"
   ```

   Save and reboot the Streamlit app.
6. **Check saved accounts.** Create an account, write and analyze a check-in, sign out, and sign in again. The saved history should still be present.

## Emotion model behavior

The Vercel API is deliberately configured to use VADER plus the lightweight emotion cue estimator, so it can start without downloading PyTorch or model weights. Those cue scores are not Transformer confidence values. The full local Docker setup can load the GoEmotions Transformer checkpoint. The public Streamlit guest mode also has a separate, optional Hugging Face inference flow when its owner configures the required secret and the visitor opts in for an entry.

## Privacy and secrets

- Add `DATABASE_URL` only in Vercel’s private Environment Variables screen and `API_URL` only in Streamlit’s private Secrets screen.
- Never commit either value or send it in chat.
- Passwords are stored as salted hashes; journal text and analysis history are saved in the connected database. Avoid entering real sensitive or identifying information in this student demo.
- To stop saving new data, remove `API_URL` from Streamlit Secrets. Existing database records remain until deleted from Neon.
