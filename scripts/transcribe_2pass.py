#!/usr/bin/env python3
"""
2-Pass Transcription (Whisper Courtside Edition method)

Pass 1: Quick transcription to get draft text
Pass 2: Re-transcribe with LLM-generated initial_prompt for improved accuracy

Usage:
  # Pass 1: Get draft transcript
  python3 scripts/transcribe_2pass.py --input audio.wav --pass 1

  # Claude Code/OpenClaw analyzes draft → generates prompt (see SKILL.md)

  # Pass 2: Re-transcribe with prompt
  python3 scripts/transcribe_2pass.py --input audio.wav --pass 2 --prompt "아르테미스, SLS, ..."

  # Full auto (pass1 → stdout, expects prompt on stdin for pass2)
  python3 scripts/transcribe_2pass.py --input audio.wav --pass auto
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_MODEL = "mlx-community/whisper-turbo"
SAMPLE_RATE = 16000


def load_audio(path, sr=SAMPLE_RATE):
    """Load audio file as float32 numpy array at target sample rate."""
    import subprocess, tempfile
    p = Path(path)
    if p.suffix.lower() in ('.wav',):
        audio, file_sr = sf.read(str(p), dtype='float32')
        if file_sr != sr:
            # Resample via ffmpeg
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp.close()
            subprocess.run(['ffmpeg', '-y', '-i', str(p), '-ac', '1', '-ar', str(sr),
                           '-acodec', 'pcm_s16le', tmp.name], capture_output=True)
            audio, _ = sf.read(tmp.name, dtype='float32')
            Path(tmp.name).unlink()
        elif audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio
    else:
        # Non-WAV: use ffmpeg
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        subprocess.run(['ffmpeg', '-y', '-i', str(p), '-ac', '1', '-ar', str(sr),
                       '-acodec', 'pcm_s16le', tmp.name], capture_output=True)
        audio, _ = sf.read(tmp.name, dtype='float32')
        Path(tmp.name).unlink()
        return audio


def transcribe(audio, model=DEFAULT_MODEL, language="auto", initial_prompt=None,
               condition_on_previous_text=True, timestamps=False):
    """Run Whisper MLX transcription."""
    from mlx_whisper import transcribe as mlx_transcribe

    kwargs = {}
    if language and language != "auto":
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    if timestamps:
        kwargs["word_timestamps"] = True

    t0 = time.time()
    result = mlx_transcribe(
        audio,
        path_or_hf_repo=model,
        verbose=False,
        condition_on_previous_text=condition_on_previous_text,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        temperature=0.0,
        **kwargs,
    )
    elapsed = time.time() - t0

    text = result.get("text", "").strip()
    detected_lang = result.get("language", "")
    duration = len(audio) / SAMPLE_RATE

    out = {
        "text": text,
        "language": detected_lang,
        "duration_sec": round(duration, 1),
        "elapsed_sec": round(elapsed, 1),
        "rtf": round(duration / elapsed, 1) if elapsed > 0 else 0,
        "model": model,
    }

    if timestamps and "segments" in result:
        out["segments"] = [
            {
                "start": round(s["start"], 2),
                "end": round(s["end"], 2),
                "text": s["text"].strip(),
            }
            for s in result["segments"]
        ]

    return out


def main():
    parser = argparse.ArgumentParser(
        description="2-Pass Transcription (Courtside Edition method)")
    parser.add_argument("--input", "-i", required=True, help="Input audio file")
    parser.add_argument("--pass", dest="pass_num", default="1", choices=["1", "2", "auto"],
                       help="Pass number: 1=draft, 2=refined, auto=both")
    parser.add_argument("--prompt", "-p", default=None,
                       help="Initial prompt for pass 2 (domain terms, entities)")
    parser.add_argument("--prompt-file", default=None,
                       help="Read initial prompt from file")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Whisper model")
    parser.add_argument("--language", "-l", default="auto", help="Language code")
    parser.add_argument("--output", "-o", default=None, help="Output file (json, txt, or srt)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--timestamps", "-t", action="store_true",
                       help="Include segment timestamps (adds 'segments' to output)")
    parser.add_argument("--srt", action="store_true",
                       help="Output in SRT subtitle format (implies --timestamps)")
    args = parser.parse_args()
    if args.srt:
        args.timestamps = True

    audio = load_audio(args.input)
    print(f"Audio: {len(audio)/SAMPLE_RATE:.1f}s", file=sys.stderr)

    if args.pass_num == "1":
        # Pass 1: Draft transcription (no prompt, fast)
        print("=== Pass 1: Draft transcription ===", file=sys.stderr)
        result = transcribe(audio, args.model, args.language,
                          condition_on_previous_text=False,
                          timestamps=args.timestamps)
        result["pass"] = 1
        _output(result, args)

    elif args.pass_num == "2":
        # Pass 2: Refined with prompt
        prompt = args.prompt
        if args.prompt_file:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        if not prompt:
            print("ERROR: --prompt or --prompt-file required for pass 2", file=sys.stderr)
            sys.exit(1)

        print(f"=== Pass 2: Refined transcription ===", file=sys.stderr)
        print(f"Prompt: {prompt[:100]}...", file=sys.stderr)
        result = transcribe(audio, args.model, args.language,
                          initial_prompt=prompt,
                          condition_on_previous_text=True,
                          timestamps=args.timestamps)
        result["pass"] = 2
        result["initial_prompt"] = prompt
        _output(result, args)

    elif args.pass_num == "auto":
        # Auto: Pass 1, print draft for LLM, read prompt from stdin, Pass 2
        print("=== Pass 1: Draft transcription ===", file=sys.stderr)
        r1 = transcribe(audio, args.model, args.language,
                       condition_on_previous_text=False,
                       timestamps=args.timestamps)

        # Output pass 1 result for LLM consumption
        print("\n--- PASS 1 DRAFT ---", file=sys.stderr)
        print(r1["text"][:500], file=sys.stderr)
        print(f"--- (language: {r1['language']}, {r1['elapsed_sec']}s) ---\n", file=sys.stderr)

        # Output full draft to stdout for LLM to read
        draft_json = json.dumps({
            "pass": 1,
            "text": r1["text"],
            "language": r1["language"],
        }, ensure_ascii=False)
        print(draft_json)
        sys.stdout.flush()

        # Read prompt from stdin (LLM provides this)
        print("Waiting for initial_prompt on stdin...", file=sys.stderr)
        prompt = sys.stdin.readline().strip()
        if not prompt:
            print("No prompt received, outputting pass 1 result", file=sys.stderr)
            r1["pass"] = 1
            _output(r1, args)
            return

        print(f"=== Pass 2: Refined with prompt ===", file=sys.stderr)
        print(f"Prompt: {prompt[:100]}...", file=sys.stderr)
        r2 = transcribe(audio, args.model, args.language,
                       initial_prompt=prompt,
                       condition_on_previous_text=True,
                       timestamps=args.timestamps)
        r2["pass"] = 2
        r2["initial_prompt"] = prompt
        _output(r2, args)


def _format_srt_time(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _to_srt(result):
    """Convert result with segments to SRT format string."""
    segments = result.get("segments", [])
    if not segments:
        return f"1\n00:00:00,000 --> 00:00:01,000\n{result['text']}\n"
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
    return "\n".join(lines)


def _output(result, args):
    """Write result to file or stdout."""
    is_srt = args.srt or (args.output and args.output.endswith(".srt"))
    content = None

    if is_srt:
        content = _to_srt(result)
    elif args.json or (args.output and args.output.endswith(".json")):
        content = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        content = result["text"]

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(content, encoding="utf-8")
        print(f"Saved: {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
