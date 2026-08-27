#!/usr/bin/env python3
"""Private OpenAI-compatible faster-whisper CUDA server."""

from __future__ import annotations

import os
import secrets
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel


MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", "large-v3-turbo")
PUBLIC_MODEL_ID = os.environ.get("WHISPER_PUBLIC_MODEL_ID", "whisper-large-v3-turbo")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
API_KEY = os.environ.get("WHISPER_API_KEY", "")

model: WhisperModel | None = None
loaded_at: float | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, loaded_at
    if not API_KEY:
        raise RuntimeError("WHISPER_API_KEY 환경변수를 먼저 설정하세요")
    started = time.perf_counter()
    model = WhisperModel(
        MODEL_PATH,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        local_files_only=Path(MODEL_PATH).exists(),
    )
    loaded_at = time.time()
    print(f"Whisper model loaded in {time.perf_counter() - started:.2f}s", flush=True)
    yield
    model = None


app = FastAPI(title="Voice Memo Doc CUDA", version="1.0.0", lifespan=lifespan)


def authorize(authorization: str | None) -> None:
    expected = f"Bearer {API_KEY}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if model is not None else "loading",
        "model": PUBLIC_MODEL_ID,
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "loaded_at": loaded_at,
    }


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)) -> dict[str, object]:
    authorize(authorization)
    return {
        "object": "list",
        "data": [{"id": PUBLIC_MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form(PUBLIC_MODEL_ID, alias="model"),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    condition_on_previous_text: bool = Form(default=False),
    authorization: str | None = Header(default=None),
):
    authorize(authorization)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if model_name not in {"whisper-1", PUBLIC_MODEL_ID, MODEL_PATH}:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    if response_format not in {"json", "text", "verbose_json"}:
        raise HTTPException(
            status_code=400,
            detail="response_format must be json, text, or verbose_json",
        )

    suffix = Path(file.filename or "audio.bin").suffix or ".bin"
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        segments_iter, info = model.transcribe(
            temp_path,
            language=language or None,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=temperature,
            initial_prompt=prompt or None,
            vad_filter=True,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=response_format == "verbose_json",
        )
        segments = list(segments_iter)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    text = "".join(segment.text for segment in segments).strip()
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "json":
        return {"text": text}

    duration = float(info.duration)
    return JSONResponse(
        {
            "task": "transcribe",
            "language": info.language,
            "duration": duration,
            "text": text,
            "segments": [
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segments
            ],
            "performance": {
                "elapsed_seconds": round(elapsed, 3),
                "realtime_factor": round(elapsed / duration, 4) if duration else None,
                "audio_seconds_per_second": round(duration / elapsed, 2) if elapsed else None,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("WHISPER_HOST", "127.0.0.1"),
        port=int(os.environ.get("WHISPER_PORT", "8001")),
    )
