# MoodMentor with Neon Postgres

MoodMentor can use Neon as the PostgreSQL provider for its existing FastAPI
account and journal-history service. Neon is optional for local guest mode:
the Streamlit app remains usable in memory when no API service is configured.

## Supported architecture

- **Streamlit:** dashboard and guest experience.
- **Vercel or another Python host:** `vercel_api.py`, the existing FastAPI app.
- **Neon:** PostgreSQL storage for users, sessions, mood logs, and recommendations.

The existing FastAPI authentication flow remains authoritative. This project
does not currently use Neon Auth, Neon Realtime, Neon Functions, or an AI
Gateway. Do not provision a second authentication system or duplicate health
function unless the application architecture is deliberately changed.

## Configure Neon

1. Create a Neon PostgreSQL project and copy its pooled connection string.
2. Keep the connection string private. Do not commit it to Git, Streamlit
   secrets, source files, screenshots, or chat.
3. Configure the pooled URL as `DATABASE_URL` in the API host's private
   environment variables.
4. Deploy the API from the repository root using the existing `vercel.json`
   and `vercel_api.py`.
5. Open `/health` on the deployed API. It should return:

   ```json
   {"status":"ok","database":"connected"}
   ```

The API initializes the existing SQLAlchemy schema on startup:
`users`, `auth_sessions`, `mood_logs`, and `recommendations`.

## Connect Streamlit

In Streamlit Community Cloud, add the deployed API origin to the app's
private secrets:

```toml
API_URL = "https://your-moodmentor-api.example"
```

Reboot the Streamlit app, register a test account, save a check-in, sign out,
and sign in again to verify persistent history. If `API_URL` is absent or the
API is unreachable, the app should remain in guest mode and explain the
account-connection limitation.

## Local verification

Run the API with a Neon URL in the process environment:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://user:password@host-pooler.region.aws.neon.tech/dbname?sslmode=require"
python -m uvicorn wellness_api.main:app --host 127.0.0.1 --port 8000
```

Then verify:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```

Use a `.env` or `.env.local` file only for local development, keep it ignored
by Git, and rotate any credential that is accidentally exposed.

## Privacy and operational limits

Neon-backed accounts persist journal text and analysis history. Use synthetic
demo entries, restrict database access, configure retention and backups, and
avoid treating MoodMentor output as a clinical, employment, or performance
decision. Free hosting plans may sleep or impose compute and storage limits;
this setup is suitable for a student demonstration, not a production clinical
service.
