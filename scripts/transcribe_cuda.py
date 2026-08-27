#!/usr/bin/env python3
"""CUDA transcription for voice-memo-doc.

Supports two execution modes:

- local: run faster-whisper directly on an NVIDIA GPU.
- api: call a private OpenAI-compatible faster-whisper server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


CUDA_MODELS = {
    "turbo": "turbo",
    "large-v3-turbo": "turbo",
    "whisper-large-v3-turbo": "turbo",
    "large-v3": "large-v3",
    "small": "small",
    "base": "base",
}
DEFAULT_LOCAL_MODEL = "large-v3-turbo"
DEFAULT_API_MODEL = "whisper-large-v3-turbo"
DEFAULT_SERVER_URL = os.environ.get("VOICE_MEMO_CUDA_URL", "http://127.0.0.1:8001")
DEFAULT_API_KEY = os.environ.get("VOICE_MEMO_CUDA_API_KEY", "")


def resolve_cuda_model(model_name: str) -> str:
    """Resolve local faster-whisper aliases while preserving full model IDs."""
    return CUDA_MODELS.get(model_name, model_name)


def _normalized_result(
    payload: dict[str, Any],
    *,
    elapsed: float,
    model: str,
    backend: str,
    compute_type: str,
) -> dict[str, Any]:
    segments = [
        {
            "start": round(float(segment.get("start", 0)), 2),
            "end": round(float(segment.get("end", 0)), 2),
            "text": str(segment.get("text", "")).strip(),
        }
        for segment in payload.get("segments", [])
    ]
    duration = float(payload.get("duration") or 0)
    if not duration and segments:
        duration = max(segment["end"] for segment in segments)

    performance = payload.get("performance") or {}
    server_elapsed = performance.get("elapsed_seconds")
    return {
        "text": str(payload.get("text", "")).strip(),
        "segments": segments,
        "language": str(payload.get("language", "")),
        "duration_sec": round(duration, 2),
        "elapsed_sec": round(elapsed, 3),
        "server_elapsed_sec": (
            round(float(server_elapsed), 3) if server_elapsed is not None else None
        ),
        "rtf": round(duration / elapsed, 2) if elapsed > 0 and duration > 0 else 0,
        "backend": backend,
        "model": model,
        "device": "cuda",
        "compute_type": compute_type,
    }


def transcribe_via_api(
    input_path: str,
    *,
    server_url: str = DEFAULT_SERVER_URL,
    api_key: str = DEFAULT_API_KEY,
    model: str = DEFAULT_API_MODEL,
    language: str = "ko",
    initial_prompt: str | None = None,
    condition_on_previous_text: bool = False,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Transcribe through an OpenAI-compatible audio transcription API."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests가 필요합니다: python3 -m pip install requests") from exc

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path}")

    base = server_url.rstrip("/")
    endpoint = (
        f"{base}/audio/transcriptions"
        if base.endswith("/v1")
        else f"{base}/v1/audio/transcriptions"
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data: dict[str, str] = {
        "model": model,
        "language": language,
        "response_format": "verbose_json",
        "temperature": "0",
        "condition_on_previous_text": str(condition_on_previous_text).lower(),
    }
    if initial_prompt:
        data["prompt"] = initial_prompt

    started = time.perf_counter()
    with path.open("rb") as audio_file:
        response = requests.post(
            endpoint,
            headers=headers,
            data=data,
            files={"file": (path.name, audio_file)},
            timeout=(10, timeout),
        )
    elapsed = time.perf_counter() - started

    if not response.ok:
        detail = response.text[:1000]
        raise RuntimeError(f"CUDA API 오류 HTTP {response.status_code}: {detail}")

    payload = response.json()
    compute_type = str((payload.get("performance") or {}).get("compute_type", "remote"))
    return _normalized_result(
        payload,
        elapsed=elapsed,
        model=model,
        backend="cuda-api",
        compute_type=compute_type,
    )


def transcribe_locally(
    input_path: str,
    *,
    model: str = DEFAULT_LOCAL_MODEL,
    language: str = "ko",
    initial_prompt: str | None = None,
    condition_on_previous_text: bool = False,
    compute_type: str = "int8_float16",
) -> dict[str, Any]:
    """Run faster-whisper directly on a local NVIDIA CUDA GPU."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper가 필요합니다: pip install -r requirements-cuda.txt") from exc

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path}")

    resolved_model = resolve_cuda_model(model)
    load_started = time.perf_counter()
    whisper = WhisperModel(resolved_model, device="cuda", compute_type=compute_type)
    model_load_sec = time.perf_counter() - load_started

    started = time.perf_counter()
    segments_iter, info = whisper.transcribe(
        str(path),
        language=None if language == "auto" else language,
        task="transcribe",
        beam_size=1,
        best_of=1,
        temperature=0,
        initial_prompt=initial_prompt,
        vad_filter=True,
        condition_on_previous_text=condition_on_previous_text,
        word_timestamps=True,
    )
    segments = list(segments_iter)
    elapsed = time.perf_counter() - started
    duration = float(info.duration)
    result = {
        "text": "".join(segment.text for segment in segments).strip(),
        "segments": [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ],
        "language": info.language,
        "duration": duration,
    }
    normalized = _normalized_result(
        result,
        elapsed=elapsed,
        model=resolved_model,
        backend="cuda-local",
        compute_type=compute_type,
    )
    normalized["model_load_sec"] = round(model_load_sec, 3)
    return normalized


def _format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _render_output(
    result: dict[str, Any],
    output: str | None,
    json_output: bool,
    srt: bool,
) -> str:
    suffix = Path(output).suffix.lower() if output else ""
    if srt or suffix == ".srt":
        blocks = []
        for index, segment in enumerate(result.get("segments", []), 1):
            blocks.append(
                f"{index}\n{_format_srt_time(segment['start'])} --> {_format_srt_time(segment['end'])}\n"
                f"{segment['text']}\n"
            )
        return "\n".join(blocks)
    if suffix == ".md":
        title = Path(output).stem.replace("_", " ")
        lines = [title, ""]
        for segment in result.get("segments", []):
            lines.append(
                f"[{_format_time(segment['start'])} - {_format_time(segment['end'])}] {segment['text']}"
            )
        return "\n".join(lines) + "\n"
    if json_output or suffix == ".json":
        return json.dumps(result, ensure_ascii=False, indent=2)
    return result["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="voice-memo-doc CUDA transcription")
    parser.add_argument("--input", "-i", required=True, help="입력 오디오 파일")
    parser.add_argument("--output", "-o", default=None, help="출력 파일(txt, json, md, srt)")
    parser.add_argument("--backend", choices=("api", "local"), default="api")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="기본값: API=whisper-large-v3-turbo, local=large-v3-turbo",
    )
    parser.add_argument("--language", "-l", default="ko")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--pass", dest="pass_num", choices=("1", "2"), default="1")
    parser.add_argument("--prompt", "-p", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--srt", action="store_true")
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.pass_num == "2" and not prompt:
        parser.error("--pass 2에는 --prompt 또는 --prompt-file이 필요합니다")

    condition_on_previous_text = args.pass_num == "2"
    selected_model = args.model or (
        DEFAULT_API_MODEL if args.backend == "api" else DEFAULT_LOCAL_MODEL
    )
    try:
        if args.backend == "api":
            result = transcribe_via_api(
                args.input,
                server_url=args.server_url,
                api_key=args.api_key,
                model=selected_model,
                language=args.language,
                initial_prompt=prompt,
                condition_on_previous_text=condition_on_previous_text,
                timeout=args.timeout,
            )
        else:
            result = transcribe_locally(
                args.input,
                model=selected_model,
                language=args.language,
                initial_prompt=prompt,
                condition_on_previous_text=condition_on_previous_text,
                compute_type=args.compute_type,
            )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    result["pass"] = int(args.pass_num)
    if prompt:
        result["initial_prompt"] = prompt
    content = _render_output(result, args.output, args.json, args.srt)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(content, encoding="utf-8")
        print(f"Saved: {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
