#!/usr/bin/env python3
"""Compare MLX and CUDA API transcription speed on the same audio file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transcribe_2pass import SAMPLE_RATE, load_audio, transcribe as transcribe_mlx
from transcribe_cuda import transcribe_via_api


def markdown_table(results: list[dict]) -> str:
    lines = [
        "| Backend | Model | Audio | Elapsed | Speed | Characters |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['backend']} | {result['model']} | {result['duration_sec']:.1f}s | "
            f"{result['elapsed_sec']:.3f}s | {result['rtf']:.2f}x | {len(result['text']):,} |"
        )
    return "\n".join(lines)


def compact_result(result: dict) -> dict:
    """Keep benchmark evidence without embedding the complete transcript."""
    text = result.get("text", "")
    summary = {key: value for key, value in result.items() if key not in {"text", "segments"}}
    summary["text_characters"] = len(text)
    summary["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    summary["segment_count"] = len(result.get("segments", []))
    return summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX/CUDA 동일 음성 속도 비교")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--backends", default="mlx,cuda", help="mlx,cuda 중 쉼표로 선택")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--mlx-model", default="large-v3-turbo")
    parser.add_argument("--cuda-model", default="whisper-large-v3-turbo")
    parser.add_argument("--cuda-url", default=None)
    parser.add_argument("--cuda-api-key", default=None)
    parser.add_argument("--output", "-o", default=None, help="결과 JSON 파일")
    parser.add_argument(
        "--include-transcripts",
        action="store_true",
        help="결과 JSON에 전체 text와 segments 포함(기본값은 요약만 저장)",
    )
    args = parser.parse_args()

    selected = {item.strip() for item in args.backends.split(",") if item.strip()}
    invalid = selected - {"mlx", "cuda"}
    if invalid:
        parser.error(f"지원하지 않는 backend: {', '.join(sorted(invalid))}")

    results: list[dict] = []
    if "mlx" in selected:
        audio = load_audio(args.input)
        mlx_result = transcribe_mlx(
            audio,
            model=args.mlx_model,
            language=args.language,
            condition_on_previous_text=False,
            timestamps=False,
        )
        mlx_result["backend"] = "mlx"
        results.append(mlx_result)

    if "cuda" in selected:
        kwargs = {
            "model": args.cuda_model,
            "language": args.language,
            "condition_on_previous_text": False,
        }
        if args.cuda_url:
            kwargs["server_url"] = args.cuda_url
        if args.cuda_api_key:
            kwargs["api_key"] = args.cuda_api_key
        results.append(transcribe_via_api(args.input, **kwargs))

    print(markdown_table(results))
    saved_results = (
        results
        if args.include_transcripts
        else [compact_result(result) for result in results]
    )
    input_path = Path(args.input).resolve()
    payload = {
        "input": input_path.name,
        "input_sha256": file_sha256(input_path),
        "results": saved_results,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
