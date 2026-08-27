# voice-memo-doc

Apple Silicon(MLX)과 NVIDIA GPU(CUDA)를 지원하는 고정확도 음성 전사 + 문서 생성 스킬.

[Whisper: Courtside Edition](https://arxiv.org/abs/2602.18966) 논문의 2-Pass Multi-Agent 파이프라인을 구현하여, AI 에이전트가 직접 도메인 분석을 수행하고 Whisper의 `initial_prompt`를 최적화합니다.

## Features

- **MLX/CUDA 이중 백엔드** — Mac에서는 MLX, NVIDIA GPU에서는 faster-whisper를 로컬 또는 API 방식으로 실행
- **동일 음원 벤치마크** — 1시간 합성 음원 기준 M5 Air 19.5x, RTX A4000 38.78x 실측
- **2-Pass P4 파이프라인** — 6-agent 분석(Topic → NER → Jargon → Sentence Builder)으로 CER 8.6%, WER 7.8% 개선
- **LLM-agnostic** — Claude Code, OpenClaw, Codex 등 어떤 AI 에이전트에서든 동작
- **문서 생성** — 전사 결과를 메모/회의록/아이디어 마크다운으로 자동 변환

## Quick Start

### Apple Silicon MLX

```bash
# 의존성 설치
bash scripts/setup.sh

# 단일 전사
python3 scripts/transcribe.py --input audio.wav --language ko

# 고정확도 2-pass 전사
python3 scripts/transcribe_2pass.py --input audio.wav --pass 1 --language ko --json
# → AI 에이전트가 SKILL.md의 Step 2에 따라 6-agent 분석 수행
python3 scripts/transcribe_2pass.py --input audio.wav --pass 2 --language ko \
  --prompt "생성된 자연어 프롬프트"

# 문서 생성
python3 scripts/memo_to_doc.py --input transcript.txt --output memo.md
```

### NVIDIA CUDA 로컬 실행

```bash
python3 -m pip install -r requirements-cuda.txt
python3 scripts/transcribe_cuda.py --backend local \
  --input audio.wav --language ko --output transcript.txt
```

CUDA 2-pass도 같은 프롬프트 규칙을 사용합니다.

```bash
python3 scripts/transcribe_cuda.py --backend local \
  --input audio.wav --pass 2 --language ko \
  --prompt "생성된 자연어 프롬프트" --output transcript.json --json
```

### NVIDIA CUDA 원격 API 실행

OpenAI 호환 faster-whisper 서버가 이미 떠 있다면 Mac에서도 원격 GPU를 사용할 수 있습니다.

GPU 서버를 직접 띄울 때는 다음처럼 실행합니다. 기본 바인딩은 `127.0.0.1`이므로 외부 접근은 Tailscale Serve나 다른 사설 프록시를 사용하세요.

```bash
python3 -m pip install -r requirements-cuda-server.txt
export WHISPER_API_KEY='YOUR_PRIVATE_KEY'
python3 scripts/serve_cuda.py
```

```bash
export VOICE_MEMO_CUDA_URL='http://GPU-SERVER:8001'
export VOICE_MEMO_CUDA_API_KEY='YOUR_PRIVATE_KEY'

python3 scripts/transcribe_cuda.py --backend api \
  --input audio.wav --language ko --output transcript.json --json
```

API 키를 저장소나 명령 기록에 직접 넣지 말고 환경변수로 전달하세요. 자세한 서버 조건과 문제 해결은 [`references/cuda-guide.md`](references/cuda-guide.md)를 참고하세요.

## 2-Pass Pipeline (Courtside Edition)

```
Audio → [Whisper Pass 1] → Draft Transcript
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              Topic Agent  NER Agent  Jargon Agent
                    │          │          │
                    │     NER Decider  Jargon Decider
                    │      (≥0.85)     (≤224 tok)
                    └──────────┼──────────┘
                               ▼
                       Sentence Builder
                    (자연어, 핵심용어 뒤쪽배치)
                               │
                               ▼
                    [Whisper Pass 2 + prompt]
                               │
                               ▼
                    Safeguard (길이 ≥80%?)
                        ▼            ▼
                   Pass 2 채택    Pass 1 폴백
```

AI 에이전트가 Step 2의 6-agent 역할을 수행합니다. 스크립트는 `--prompt` 문자열만 받으므로 어떤 LLM이든 호환됩니다.

## Benchmark Results

| Metric | Baseline (1-pass) | 2-Pass P4 | Improvement |
|--------|-------------------|-----------|-------------|
| CER | 16.2% | 14.8% | -8.6% |
| WER | 29.5% | 27.2% | -7.8% |

테스트: 8분 한국어 IT 튜토리얼 영상, whisper-large-v3-turbo 모델

### Backend Speed (2026-08-27 실측)

| Backend | Hardware | Audio | Elapsed | Real-time speed |
|---|---|---:|---:|---:|
| MLX | MacBook Air M5, 32GB | 3,600초 | 185.0초 | 19.50x |
| CUDA API | RTX A4000 16GB, `int8_float16` | 3,600초 | 92.821초 | 38.78x |

A4000 서버 내부 전사 시간은 91.678초(39.27x)였고, Tailscale 업로드와 응답 수신을 포함한 클라이언트 시간은 92.821초였습니다. 이 조건에서는 A4000이 M5 Air보다 약 1.99배 빨랐습니다. 음원은 반복 합성 한국어이므로 이 표는 처리량 비교용이며 자연 발화 정확도 비교가 아닙니다. 전체 조건은 [`benchmarks/m5-air-vs-a4000-1hour.md`](benchmarks/m5-air-vs-a4000-1hour.md)에 기록했습니다.

직접 재현하려면 다음 명령을 사용합니다.

```bash
python3 scripts/benchmark_backends.py --input audio.ogg \
  --backends mlx,cuda --language ko \
  --cuda-url "$VOICE_MEMO_CUDA_URL" \
  --cuda-api-key "$VOICE_MEMO_CUDA_API_KEY" \
  --output benchmarks/result.json
```

### Prompt Guidelines

- 자연어 문장으로 작성 (쉼표 나열 금지)
- 224 토큰 이하 (Whisper 제한)
- 핵심 용어를 문장 뒤쪽에 배치
- 마크다운 기호 금지 (`**`, `` ` `` 등)
- 확실한 용어만 포함 (잘못된 용어는 오히려 성능 저하)

## Scripts

| Script | Description |
|--------|-------------|
| `transcribe_2pass.py` | 2-pass 전사 (pass 1 draft, pass 2 with prompt) |
| `transcribe.py` | 단일 Whisper 전사 |
| `transcribe_cuda.py` | CUDA 로컬/API 전사, JSON/Markdown/SRT 출력 |
| `serve_cuda.py` | 인증이 필요한 OpenAI 호환 CUDA 전사 API |
| `record.py` | CLI 녹음 (WAV, 16kHz mono) |
| `memo_to_doc.py` | 전사 → 마크다운 문서 변환 |
| `benchmark_stt.py` | CER/WER 벤치마크 (SRT ground truth 비교) |
| `benchmark_backends.py` | 동일 음원의 MLX/CUDA 처리 속도 비교 |
| `setup.sh` | 의존성 설치 (`--yes` 비대화 모드) |

## Models

권장 모델은 두 백엔드 모두 Whisper `large-v3-turbo`입니다.

| Backend | 기본 모델 값 | 실행 엔진 |
|---|---|---|
| MLX | `mlx-community/whisper-large-v3-turbo` | `mlx-whisper` |
| CUDA local | `large-v3-turbo` (`turbo`로 해석) | `faster-whisper` |
| CUDA API | `whisper-large-v3-turbo` | OpenAI 호환 서버 |

```bash
# 고정확도 large-v3-turbo 모델을 명시적으로 선택
python3 scripts/transcribe_2pass.py --input audio.wav --pass 1 \
  --model large-v3-turbo --language ko

# 저사양 Mac에서 small 모델 사용
python3 scripts/transcribe_2pass.py --input audio.wav --pass 1 --model small
```

MLX 및 CUDA 로컬 전사는 `turbo`, `large-v3`, `large-v3-turbo`, `small`, `base`
단축 이름과 전체 Hugging Face 모델 경로를 지원합니다. API 모드의 모델명은 서버가 제공하는 ID를 사용합니다.

## Requirements

- Python 3.9+
- MLX: Apple Silicon Mac(M1+), macOS 13+, `mlx-whisper`
- CUDA local: NVIDIA GPU, 호환 CUDA/cuDNN, `faster-whisper`, `ctranslate2`
- CUDA API: GPU 서버에 접근 가능한 네트워크와 API 키
- Disk: large-v3-turbo 모델 캐시 약 1.6GB 이상

## References

- [Whisper: Courtside Edition (arxiv:2602.18966)](https://arxiv.org/abs/2602.18966) — 2-Pass Multi-Agent Pipeline
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Apple Silicon Whisper

## License

Private repository. All rights reserved.
