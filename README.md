# Voxa Multilingual AI Voice Assistant

Voxa is a small web app with an HTML/CSS/JavaScript frontend and Python FastAPI backend. Gemini requests and the API key stay on the backend. Conversation history is stored in the current browser and sent with each request to preserve context.

## Requirements

- Python 3.10 or newer
- A Gemini API key
- A modern browser (Chrome or Edge recommended for built-in microphone recognition)

## Configure

Create `.env` in the project root, next to `requirements.txt`:

```env
GEMINI_API_KEY=your_gemini_api_key
# Optional; defaults to gemini-3.6-flash
GEMINI_MODEL=gemini-3.6-flash
# Optional; retries this model only when the primary model returns a quota 429.
# Set only after confirming it can generate with this API project.
# GEMINI_FALLBACK_MODEL=your-verified-model
```

Keep secrets out of `frontend/`. `.env` is ignored by Git.

## Run locally

From the project root, create and activate an environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the backend:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, serve the frontend:

```powershell
python -m http.server 5500 --directory frontend
```

Open <http://127.0.0.1:5500>. API docs: <http://127.0.0.1:8000/docs>. Health: <http://127.0.0.1:8000/api/health>.

Microphone input requests browser microphone permission. Use the local HTTP server above (`http://localhost:5500` or `http://127.0.0.1:5500`); browsers allow microphone access in secure contexts such as localhost and HTTPS. Supported browsers use built-in speech recognition. Other browsers, or browsers whose built-in recognition service is unavailable, fall back to WAV recording and the backend `POST /api/transcribe` endpoint. `POST /api/speech` returns gTTS audio as MP3.

## API

- `GET /api/health` — reports API availability and whether a backend key is configured; never returns the key.
- `POST /api/chat` — accepts `{"message":"...","history":[{"role":"user|assistant","text":"..."}]}` and returns `{"reply":"..."}`.
- `POST /api/speech` — accepts `{"text":"...","language":"en"}` and returns MP3 audio.
- `POST /api/transcribe` — accepts multipart form field `audio` containing WAV/AIFF/FLAC audio and optional `language` (such as `en-US`).

CORS permits the local frontend/API origins listed in `backend/main.py`. Update the allowlist deliberately if you host the frontend elsewhere.

## Troubleshooting

- Gemini errors show Google's sanitized API message, including the quota metric or retry window when Google returns one. Voxa retries a `429` with `GEMINI_FALLBACK_MODEL` only when that optional backend setting is configured; a project-wide quota still requires waiting for Google's retry window or changing the API project's quota/billing plan. A model appearing in the model list does not guarantee that it is currently serving.
- For microphone access, open the frontend at `http://localhost:5500` or `http://127.0.0.1:5500`, click the mic button, and allow microphone access. Microphone capture requires a secure context: localhost and HTTPS qualify, while ordinary HTTP pages on a LAN IP do not. Opening the files directly with `file://` can also cause API origin/CORS issues, so use the local web server above.
- If browser-native speech recognition is unavailable, Voxa records WAV audio and sends it to the backend transcription endpoint. That recognition service requires an internet connection.
