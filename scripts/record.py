#!/usr/bin/env python3
"""
음성 녹음 스크립트
macOS 마이크에서 WAV 파일로 음성을 녹음합니다.
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# 녹음 중단 플래그
_stop_recording = False


def signal_handler(sig, frame):
    """Ctrl+C로 녹음 중단"""
    global _stop_recording
    _stop_recording = True
    print("\n녹음을 중단합니다...")


def check_dependencies():
    """필수 패키지 확인"""
    missing = []
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        missing.append("sounddevice")
    try:
        import soundfile  # noqa: F401
    except ImportError:
        missing.append("soundfile")

    if missing:
        print(f"필수 패키지가 없습니다: {', '.join(missing)}")
        print(f"설치: pip install {' '.join(missing)}")
        sys.exit(1)


def record_audio(output_path: str, duration: int = 300, sample_rate: int = 16000):
    """
    마이크에서 음성을 녹음합니다.

    Args:
        output_path: 저장할 WAV 파일 경로
        duration: 최대 녹음 시간(초)
        sample_rate: 샘플레이트 (Hz)
    """
    import sounddevice as sd
    import soundfile as sf
    import numpy as np

    global _stop_recording
    _stop_recording = False

    # 출력 디렉토리 확인
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"녹음 설정:")
    print(f"  출력: {output_path}")
    print(f"  최대 시간: {duration}초")
    print(f"  샘플레이트: {sample_rate}Hz")
    print(f"  채널: mono")
    print()

    # 사용 가능한 입력 장치 확인
    try:
        device_info = sd.query_devices(kind="input")
        print(f"입력 장치: {device_info['name']}")
    except Exception as e:
        print(f"입력 장치를 찾을 수 없습니다: {e}")
        sys.exit(1)

    print("\n녹음 시작! (Ctrl+C로 중단)")
    print("-" * 40)

    # 녹음 데이터 저장용 버퍼
    recorded_frames = []

    def callback(indata, frames, time_info, status):
        """오디오 스트림 콜백"""
        if status:
            print(f"  [경고] {status}", file=sys.stderr)
        recorded_frames.append(indata.copy())

    # Ctrl+C 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=int(sample_rate * 0.5),  # 0.5초 블록
        ):
            start_time = time.time()
            while not _stop_recording:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    print(f"\n최대 녹음 시간({duration}초)에 도달했습니다.")
                    break
                # 경과 시간 표시
                mins, secs = divmod(int(elapsed), 60)
                print(f"\r  녹음 중... {mins:02d}:{secs:02d}", end="", flush=True)
                time.sleep(0.1)

    except Exception as e:
        print(f"\n녹음 오류: {e}")
        sys.exit(1)

    if not recorded_frames:
        print("녹음된 데이터가 없습니다.")
        sys.exit(1)

    # WAV 파일로 저장
    audio_data = np.concatenate(recorded_frames, axis=0)
    elapsed = len(audio_data) / sample_rate

    sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")

    mins, secs = divmod(int(elapsed), 60)
    print(f"\n\n녹음 완료!")
    print(f"  시간: {mins:02d}:{secs:02d}")
    print(f"  파일: {output_path}")
    print(f"  크기: {Path(output_path).stat().st_size / 1024:.1f} KB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="음성 녹음")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="출력 WAV 파일 경로 (기본: /tmp/voice_memo_{timestamp}.wav)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=300,
        help="최대 녹음 시간(초) (기본: 300)",
    )
    parser.add_argument(
        "--sample-rate",
        "-r",
        type=int,
        default=16000,
        help="샘플레이트 Hz (기본: 16000)",
    )
    args = parser.parse_args()

    # 기본 출력 경로 생성
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"/tmp/voice_memo_{timestamp}.wav"

    check_dependencies()
    record_audio(args.output, args.duration, args.sample_rate)


if __name__ == "__main__":
    main()
