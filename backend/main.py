import io
import logging
import os
from pathlib import Path
from typing import Literal

from groq import Groq
import speech_recognition as sr
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from gtts import gTTS
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
logger = logging.getLogger("voxa.groq")
app = FastAPI(title="Voxa Multilingual Voice Assistant", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:8000", "http://127.0.0.1:8000", "https://voxa-assistant.vercel.app"], allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])
SYSTEM_INSTRUCTION = "You are a helpful multilingual voice assistant. Always detect the language the user is writing or speaking in, and respond in that exact same language. If the user switches languages mid-conversation, switch your response language too."
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured in the backend environment.")
    return Groq(api_key=api_key)


def safe_provider_error(exc: Exception) -> tuple[int, str]:
    """Return a safe client error without forwarding provider text or request data."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return 429, "Groq rate limit or quota reached. Please try again later."
    if status_code in (401, 403):
        return 502, "Groq authentication failed. Check the backend API key configuration."
    return 502, "Groq request failed. Please try again later."

@app.get("/api/health")
def health():
    return {"status": "ok", "groq_configured": bool(os.getenv("GROQ_API_KEY")), "model": MODEL_NAME}

@app.post("/api/chat")
def chat(request: ChatRequest):
    client = None
    try:
        client = get_client()
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        messages.extend({"role": item.role, "content": item.text} for item in request.history)
        messages.append({"role": "user", "content": request.message})
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise HTTPException(status_code=502, detail="Groq returned an empty response.")
        return {"reply": answer, "model": MODEL_NAME}
    except HTTPException:
        raise
    except Exception as exc:
        status_code, detail = safe_provider_error(exc)
        logger.error("Groq request failed (%s, status=%s)", type(exc).__name__, getattr(exc, "status_code", "unknown"))
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
