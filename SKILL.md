---
name: voice-memo-doc
description: Transcribe audio files to text using Whisper MLX (Apple Silicon) and generate markdown documents. Supports high-accuracy 2-pass transcription based on the Courtside Edition paper (arxiv:2602.18966) where the AI agent acts as the LLM pipeline for domain/entity analysis. Use when the user asks for file transcription, STT, speech-to-text, audio-to-document, voice memo documentation, or high-accuracy transcription on macOS with Apple Silicon.
---

# Voice Memo Doc

Transcribe audio files with Whisper MLX on Apple Silicon and generate markdown documents. Supports 2-pass high-accuracy mode where Claude acts as the LLM analysis pipeline.

## Setup

```bash
bash scripts/setup.sh
```

Installs: portaudio (brew), mlx-whisper, sounddevice, soundfile.

## Workflow

### Standard Transcription (Single-Pass)

For quick transcription where speed matters more than perfect accuracy:

```bash
python3 scripts/transcribe.py --input audio.wav --language ko --output transcript.txt
```

### High-Accuracy Transcription (2-Pass, Courtside Edition)

For content with proper nouns, technical terms, or domain-specific jargon. Based on arxiv:2602.18966.

**Step 1**: Run pass 1 to get draft transcript:

```bash
python3 scripts/transcribe_2pass.py --input audio.wav --pass 1 --language ko --json
```

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
| `scripts/record.py` | CLI audio recording (WAV, 16kHz mono) |
| `scripts/memo_to_doc.py` | Transcript to markdown document |
| `scripts/benchmark_stt.py` | CER/WER accuracy benchmarking against SRT ground truth |
| `scripts/setup.sh` | Dependency installer |

## References

- `references/setup-guide.md` — BlackHole setup for system audio capture, troubleshooting

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.9+
- `mlx-whisper`, `sounddevice`, `soundfile`, `portaudio`
