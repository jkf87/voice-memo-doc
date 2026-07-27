#!/usr/bin/env python3
"""
음성→텍스트 변환 (STT) 스크립트
mlx_whisper CLI (Apple Silicon 최적화)를 사용하여
오디오 파일을 텍스트로 변환합니다.

mlx-whisper는 uv tool로 설치되어 있으므로 subprocess로 호출합니다.
CLI 명령: mlx_whisper (underscore)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .model_options import DEFAULT_MODEL, MODELS, resolve_model
except ImportError:
    from model_options import DEFAULT_MODEL, MODELS, resolve_model

MLX_WHISPER_CMD = "mlx_whisper"


def check_dependencies():
    """mlx_whisper CLI가 설치되어 있는지 확인"""
    try:
        result = subprocess.run(
            [MLX_WHISPER_CMD, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print("mlx_whisper CLI를 실행할 수 없습니다.")
            sys.exit(1)
    except FileNotFoundError:
        print("mlx_whisper CLI가 설치되어 있지 않습니다.")
        print("설치: uv tool install mlx-whisper")
        sys.exit(1)


def transcribe_audio(
    input_path: str,
    model: str = DEFAULT_MODEL,
    language: str = "ko",
):
    """
    mlx_whisper CLI로 오디오 파일을 텍스트로 변환합니다.

    Args:
        input_path: 입력 오디오 파일 경로
        model: 모델 이름 또는 HuggingFace 경로
        language: 언어 코드

    Returns:
        dict with text, segments, language
    """
    if not Path(input_path).exists():
        print(f"파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="mlx_whisper_")
    model_path = resolve_model(model)

    print(f"오디오 로드 중: {input_path}")
    print(f"모델: {model_path}")
    print(f"언어: {language}")
    print("-" * 40)

    cmd = [
        MLX_WHISPER_CMD,
        str(Path(input_path).resolve()),
        "--model", model_path,
        "--language", language,
        "--output-format", "json",
        "--output-dir", tmp_dir,
        "--word-timestamps", "True",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("변환 시간이 초과되었습니다 (10분).")
        sys.exit(1)

    if proc.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"mlx_whisper 오류:\n{proc.stderr}")
        sys.exit(1)

    # mlx_whisper는 {output_dir}/{input_stem}.json 으로 결과를 저장
    input_stem = Path(input_path).stem
    json_path = Path(tmp_dir) / f"{input_stem}.json"

    if not json_path.exists():
        json_files = list(Path(tmp_dir).glob("*.json"))
        if json_files:
            json_path = json_files[0]
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"결과 JSON 파일을 찾을 수 없습니다.")
            print(f"stdout: {proc.stdout}")
            print(f"stderr: {proc.stderr}")
            sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        raw_result = json.load(f)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    text = raw_result.get("text", "").strip()
    segments = raw_result.get("segments", [])
    detected_lang = raw_result.get("language", language)

    print(f"변환 완료! 총 {len(text)}자, {len(segments)}개 세그먼트")
    print(f"감지된 언어: {detected_lang}")

    return {
        "text": text,
        "segments": [
            {
                "start": s.get("start", 0),
                "end": s.get("end", 0),
                "text": s.get("text", ""),
            }
            for s in segments
        ],
        "language": detected_lang,
    }


def main():
    parser = argparse.ArgumentParser(description="음성→텍스트 변환 (STT) - mlx_whisper CLI")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="입력 오디오 파일 경로",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="출력 파일 경로 (기본: stdout). .json이면 JSON, 아니면 텍스트",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"모델 이름 (기본: {DEFAULT_MODEL}). 선택: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--language", "-l",
        default="ko",
        help="언어 코드 (기본: ko)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 형식으로 출력",
    )
    args = parser.parse_args()

    check_dependencies()

    result = transcribe_audio(args.input, args.model, args.language)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if args.output.endswith(".json") or args.json:
            Path(args.output).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            Path(args.output).write_text(result["text"], encoding="utf-8")
        print(f"저장됨: {args.output}")
    else:
        if args.json:
            print()
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print()
            print(result["text"])


if __name__ == "__main__":
    main()
