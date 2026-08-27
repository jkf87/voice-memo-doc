---
name: voice-memo-doc
description: Transcribe audio files with Whisper on Apple Silicon MLX or NVIDIA CUDA (local or OpenAI-compatible API), then generate markdown documents. Supports high-accuracy 2-pass transcription based on the Courtside Edition paper (arxiv:2602.18966), where the AI agent performs domain and entity analysis. Use for file transcription, STT, speech-to-text, audio-to-document, voice memo documentation, backend speed comparison, or high-accuracy transcription on Mac and CUDA machines.
---

# Voice Memo Doc

Transcribe audio files with Whisper on Apple Silicon MLX or NVIDIA CUDA and generate markdown documents. Supports 2-pass high-accuracy mode where the active AI agent acts as the LLM analysis pipeline.

## Setup

MLX on Apple Silicon:

```bash
bash scripts/setup.sh
```

Installs: portaudio (brew), mlx-whisper, sounddevice, soundfile.

CUDA on a machine with an NVIDIA GPU:

```bash
python3 -m pip install -r requirements-cuda.txt
```

To host the included private API:

```bash
python3 -m pip install -r requirements-cuda-server.txt
export WHISPER_API_KEY='YOUR_PRIVATE_KEY'
python3 scripts/serve_cuda.py
```

For a remote GPU server, install only `requests` on the client and set `VOICE_MEMO_CUDA_URL` and `VOICE_MEMO_CUDA_API_KEY`. Never store the key in the repository.

## Backend Selection

| Situation | Command |
|---|---|
| Apple Silicon Mac | `scripts/transcribe_2pass.py` |
| NVIDIA GPU on the same machine | `scripts/transcribe_cuda.py --backend local` |
| NVIDIA GPU exposed as a private API | `scripts/transcribe_cuda.py --backend api` |

Use the existing MLX flow unless the user asks for CUDA or a CUDA server is already in scope. CUDA local mode must run on the NVIDIA host; a Mac client uses API mode.

## Workflow

### Standard Transcription (Single-Pass)

For quick transcription where speed matters more than perfect accuracy:

```bash
python3 scripts/transcribe.py --input audio.wav --language ko --output transcript.txt
```

CUDA local:

```bash
python3 scripts/transcribe_cuda.py --backend local \
  --input audio.wav --language ko --output transcript.txt
```

CUDA API:

```bash
python3 scripts/transcribe_cuda.py --backend api \
  --server-url "$VOICE_MEMO_CUDA_URL" \
  --input audio.wav --language ko --output transcript.json --json
```

### High-Accuracy Transcription (2-Pass, Courtside Edition)

For content with proper nouns, technical terms, or domain-specific jargon. Based on arxiv:2602.18966.

**Step 1**: Run pass 1 to get draft transcript:

```bash
python3 scripts/transcribe_2pass.py --input audio.wav --pass 1 --language ko --json
```

For CUDA, replace the command with `scripts/transcribe_cuda.py` and add `--backend local` or `--backend api`. The draft analysis and prompt construction steps remain identical.

**Step 2**: Analyze the draft transcript. Perform this multi-agent analysis:

1. **Topic Classification**: Identify the domain/topic in one line.
2. **Named Entity Recognition**: Extract proper nouns (people, organizations, products). Include both the misrecognized form and correct spelling.
3. **NER Validation**: Keep only entities that are likely actually spoken. Discard uncertain ones to prevent over-correction. Maximum 5.
4. **Jargon Extraction**: Extract domain-specific technical terms, abbreviations, acronyms.
5. **Jargon Validation**: Keep only terms that are likely to be misrecognized by Whisper (English abbreviations, foreign words). Maximum 5.
6. **Sentence Builder**: Combine all validated elements into ONE natural Korean sentence. Place the most important terms near the END of the sentence (Whisper weights later tokens more heavily). Keep under 20 words. No markdown, no quotes, no bold — pure Korean text only.

Example analysis for a space documentary:

- Topic: NASA 달 탐사 발표
- NER: 제러드 아이젝만, NASA, 게이트웨이
- Jargon: SLS, CLPS, RTG, 아르테미스
- Sentence: `NASA의 달 탐사 계획으로 게이트웨이 중단과 제러드 아이젝만의 아르테미스 SLS 발사를 다룹니다`

**Step 3**: Run pass 2 with the generated sentence as prompt:

```bash
python3 scripts/transcribe_2pass.py --input audio.wav --pass 2 --language ko \
  --prompt "NASA의 달 탐사 계획으로 게이트웨이 중단과 제러드 아이젝만의 아르테미스 SLS 발사를 다룹니다"
```

CUDA equivalent:

```bash
python3 scripts/transcribe_cuda.py --backend api --input audio.wav --pass 2 --language ko \
  --prompt "NASA의 달 탐사 계획으로 게이트웨이 중단과 제러드 아이젝만의 아르테미스 SLS 발사를 다룹니다"
```

**Step 4**: Safeguard check. If pass 2 output is less than 80% the length of pass 1, discard pass 2 and use pass 1 result (prompt caused decoder degradation).

### Document Generation

Convert transcript to formatted markdown:

```bash
python3 scripts/memo_to_doc.py --input transcript.txt --output memo.md --template memo
```

Templates: `memo` (default), `meeting`, `idea`

### Full Pipeline Example

```bash
# Record
python3 scripts/record.py --output /tmp/memo.wav --duration 120

# Transcribe (single-pass)
python3 scripts/transcribe.py --input /tmp/memo.wav --language ko | \
python3 scripts/memo_to_doc.py --output ~/Documents/memo.md
```

## 2-Pass Prompt Guidelines

Critical rules for generating the initial_prompt (from paper findings):

- **Natural sentence, NOT comma-separated list** — Whisper's decoder expects natural text continuation
- **Under 224 tokens** — Whisper only uses the last 224 tokens of the prompt
- **High-value terms at the END** — later tokens have more influence on decoding
- **No markdown formatting** — asterisks, backticks, bold markers corrupt the decoder
- **Conservative** — only include terms you're confident about. Wrong terms cause degradation.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/transcribe_2pass.py` | 2-pass transcription (pass 1 draft, pass 2 with prompt) |
| `scripts/transcribe.py` | Single-pass Whisper transcription |
| `scripts/transcribe_cuda.py` | CUDA local/API transcription with text, JSON, Markdown, and SRT output |
| `scripts/serve_cuda.py` | Authenticated OpenAI-compatible CUDA transcription API |
| `scripts/record.py` | CLI audio recording (WAV, 16kHz mono) |
| `scripts/memo_to_doc.py` | Transcript to markdown document |
| `scripts/benchmark_stt.py` | CER/WER accuracy benchmarking against SRT ground truth |
| `scripts/benchmark_backends.py` | MLX/CUDA throughput comparison on the same audio |
| `scripts/setup.sh` | Dependency installer |

## References

- `references/setup-guide.md` — BlackHole setup for system audio capture, troubleshooting
- `references/cuda-guide.md` — CUDA local/API setup, verification, and troubleshooting
- `benchmarks/m5-air-vs-a4000-1hour.md` — reproducible 1-hour speed comparison

## Models

기본 계열: Whisper `large-v3-turbo` (809M, ~1.6GB)

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `large-v3-turbo` | 809M | ~1s/3s chunk | Best | 기본 (권장) |
| `turbo` | 809M | ~1s/3s chunk | Best | 호환 별칭 |
| `large-v3` | 1.55B | 느림 | Best | 최대 모델 |
| `small` | 244M | ~0.3s/3s chunk | Good | 저사양 Mac |
| `base` | 74M | 가장 빠름 | Basic | 최소 사양 |

MLX와 CUDA local 스크립트는 단축 이름과 전체 Hugging Face 모델 경로를 지원합니다. CUDA API 모드에서는 서버가 제공하는 모델 ID를 사용합니다.

```bash
python3 scripts/transcribe_2pass.py --input audio.wav --pass 1 \
  --model large-v3-turbo --language ko
```

## Requirements

- Python 3.9+
- MLX: Apple Silicon Mac, `mlx-whisper`, `sounddevice`, `soundfile`, `portaudio`
- CUDA local: NVIDIA GPU, compatible CUDA/cuDNN, `faster-whisper`, `ctranslate2`
- CUDA API: `requests` and private access to the transcription server
- Disk: ~1.6GB (large-v3-turbo model cache)
