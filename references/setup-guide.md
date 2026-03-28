# Voice Memo Doc 설정 가이드

## 1. Python 가상환경 및 패키지 설치

```bash
# 스킬 디렉토리에서 venv 생성
cd /path/to/voice-memo-doc
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install mlx-whisper sounddevice soundfile
```

macOS에서 `sounddevice`가 PortAudio를 필요로 할 수 있습니다:

```bash
brew install portaudio
```

### 시스템 오디오 캡처 (선택)

실시간 모드에서 컴퓨터 소리(유튜브, 회의 앱 등)를 전사하려면 BlackHole 가상 오디오 디바이스를 설치합니다:

```bash
brew install blackhole-2ch
```

설치 후 macOS 오디오 MIDI 설정에서 **다중 출력 장치**를 만들어야 합니다:

1. `/Applications/Utilities/Audio MIDI Setup.app` 실행
2. 좌측 하단 **+** → **다중 출력 장치 생성**
3. **BlackHole 2ch** + **내장 스피커**(또는 사용 중인 출력)를 모두 체크
4. 시스템 설정 → 사운드 → 출력에서 **다중 출력 장치** 선택

이렇게 하면 소리를 들으면서 동시에 BlackHole로 캡처할 수 있습니다. GUI의 실시간 모드에서 오디오 소스를 **🔊 BlackHole 2ch**로 선택하세요.

## 2. Whisper MLX 모델

모델은 첫 실행 시 HuggingFace에서 자동 다운로드됩니다. 사용 가능한 모델:

| 모델명 | HuggingFace 경로 | 크기 | 속도 |
|--------|------------------|------|------|
| `turbo` (기본) | mlx-community/whisper-turbo | ~1.5GB | 매우 빠름 |
| `large-v3` | mlx-community/whisper-large-v3-mlx | ~3GB | 높은 정확도 |
| `large-v3-turbo` | mlx-community/whisper-large-v3-turbo | ~1.6GB | 빠름+정확 |
| `small` | mlx-community/whisper-small-mlx | ~500MB | 빠름 |
| `base` | mlx-community/whisper-base-mlx | ~150MB | 가장 빠름 |

> **참고**: Apple Silicon (M1/M2/M3/M4) Mac이 필요합니다. MLX는 Apple GPU를 활용합니다.

## 3. 설치 확인

```bash
# venv 활성화
source .venv/bin/activate

# mlx-whisper 확인
python3 -c "import mlx_whisper; print('mlx-whisper OK')"

# 녹음 테스트 (5초)
python3 scripts/record.py --output /tmp/test.wav --duration 5

# STT 테스트
python3 scripts/transcribe.py --input /tmp/test.wav
```

## 4. 문제 해결

### "입력 장치를 찾을 수 없습니다"
- macOS 시스템 설정 → 사운드 → 입력에서 마이크가 선택되어 있는지 확인
- 터미널 앱에 마이크 권한이 부여되어 있는지 확인 (시스템 설정 → 개인정보 보호 → 마이크)

### "mlx-whisper 패키지가 필요합니다"
```bash
source .venv/bin/activate
pip install mlx-whisper
```

### 모델 다운로드가 느린 경우
더 작은 모델을 사용해 보세요:
```bash
python3 scripts/transcribe.py --input audio.wav --model small
```

### macOS에서 PortAudio 오류
```bash
brew install portaudio
pip install --force-reinstall sounddevice
```
