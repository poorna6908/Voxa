import io
import logging
import os
from pathlib import Path
import re
from typing import Literal

from google import genai
from google.genai import types
import speech_recognition as sr
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from gtts import gTTS
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
logger = logging.getLogger("voxa.gemini")
app = FastAPI(title="Voxa Multilingual Voice Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://voxa-frontend-one.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_INSTRUCTION = "You are a helpful multilingual voice assistant. Always detect the language the user is writing or speaking in, and respond in that exact same language. If the user switches languages mid-conversation, switch your response language too."
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Set this only to a model that has been verified for the configured API project.
# Model listings can include models that are temporarily unavailable or quota-limited.
FALLBACK_MODEL_NAME = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
TTS_LANGUAGES = {"auto":"en","en":"en","hi":"hi","es":"es","fr":"fr","de":"de","it":"it","pt":"pt","ja":"ja","ko":"ko","zh-CN":"zh-CN","ar":"ar","ru":"ru","bn":"bn","ta":"ta","te":"te","mr":"mr","ur":"ur"}

class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=12000)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    history: list[Message] = Field(default_factory=list, max_length=80)

class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    language: str = "auto"

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured in the backend .env file.")
    return genai.Client(api_key=api_key)


def safe_provider_error(exc: Exception) -> tuple[int, str]:
    """Keep Google's useful error while ensuring the configured key cannot leak."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None) or str(exc)
    if api_key:
        message = message.replace(api_key, "[redacted API key]")
    message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted API key]", message)
    message = " ".join(message.split())[:900]
    status_code = 429 if str(code) == "429" else 502
    if status_code == 429:
        message = (
            f"{message} Check the quota metric and retry time below. If the exhausted quota is "
            "specific to the model, configure GEMINI_FALLBACK_MODEL in the backend .env with "
            "a model verified for this API project; project-wide quota requires waiting or a quota change."
        )
    return status_code, f"Gemini API error ({code or type(exc).__name__}): {message}"


def is_quota_error(exc: Exception) -> bool:
    if str(getattr(exc, "code", "")) != "429":
        return False
    message = str(getattr(exc, "message", None) or exc).lower()
    return "quota" in message or "resource_exhausted" in message or "rate limit" in message

@app.get("/api/health")
def health():
    return {"status": "ok", "gemini_configured": bool(os.getenv("GEMINI_API_KEY")), "model": MODEL_NAME, "fallback_model": FALLBACK_MODEL_NAME or None}

@app.post("/api/chat")
def chat(request: ChatRequest):
    client = None
    try:
        history = [
            types.Content(
                role="model" if item.role == "assistant" else "user",
                parts=[types.Part.from_text(text=item.text)],
            )
            for item in request.history
        ]
        client = get_client()
        config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
        model_used = MODEL_NAME
        try:
            chat_session = client.chats.create(model=MODEL_NAME, history=history, config=config)
            response = chat_session.send_message(request.message)
        except Exception as exc:
            if not FALLBACK_MODEL_NAME or FALLBACK_MODEL_NAME == MODEL_NAME or not is_quota_error(exc):
                raise
            logger.warning("Primary Gemini model quota exhausted; retrying with configured fallback model %s", FALLBACK_MODEL_NAME)
            try:
                fallback_session = client.chats.create(model=FALLBACK_MODEL_NAME, history=history, config=config)
                response = fallback_session.send_message(request.message)
            except Exception as fallback_exc:
                _, primary_detail = safe_provider_error(exc)
                _, fallback_detail = safe_provider_error(fallback_exc)
                raise HTTPException(
                    status_code=429,
                    detail=f"{primary_detail} Configured fallback model {FALLBACK_MODEL_NAME} also failed: {fallback_detail}",
                ) from fallback_exc
            model_used = FALLBACK_MODEL_NAME
        answer = (response.text or "").strip()
        if not answer:
            raise HTTPException(status_code=502, detail="Gemini returned an empty response.")
        return {"reply": answer, "model": model_used}
    except HTTPException:
        raise
    except Exception as exc:
        status_code, detail = safe_provider_error(exc)
        logger.error("%s", detail)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    finally:
        if client is not None:
            client.close()

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = "en-US"):
    content = await audio.read()
    try:
        if not content or len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio must be non-empty and smaller than 15 MB.")
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(content)) as source:
            data = recognizer.record(source)
        return {"text": recognizer.recognize_google(data, language=language)}
    except HTTPException:
        raise
    except sr.UnknownValueError as exc:
        raise HTTPException(status_code=422, detail="Sorry, I couldn't understand the audio.") from exc
    except sr.RequestError as exc:
        raise HTTPException(status_code=502, detail="Speech recognition service is unavailable.") from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=415, detail="Use a WAV, AIFF, or FLAC audio file.") from exc
    finally:
        await audio.close()

@app.post("/api/speech")
def speech(request: SpeechRequest):
    output = io.BytesIO()
    try:
        gTTS(text=request.text, lang=TTS_LANGUAGES.get(request.language, "en")).write_to_fp(output)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Text-to-speech service is unavailable.") from exc
    return Response(output.getvalue(), media_type="audio/mpeg", headers={"Cache-Control":"no-store"})
