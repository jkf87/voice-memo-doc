# voice-memo-doc

Apple Silicon (MLX) 기반 고정확도 음성 전사 + 문서 생성 스킬.

[Whisper: Courtside Edition](https://arxiv.org/abs/2602.18966) 논문의 2-Pass Multi-Agent 파이프라인을 구현하여, AI 에이전트가 직접 도메인 분석을 수행하고 Whisper의 `initial_prompt`를 최적화합니다.

## Features

- **Whisper MLX** — Apple Silicon Neural Engine 활용, 실시간 대비 34x 속도
- **2-Pass P4 파이프라인** — 6-agent 분석(Topic → NER → Jargon → Sentence Builder)으로 CER 8.6%, WER 7.8% 개선
- **LLM-agnostic** — Claude Code, OpenClaw, Codex 등 어떤 AI 에이전트에서든 동작
- **문서 생성** — 전사 결과를 메모/회의록/아이디어 마크다운으로 자동 변환

## Quick Start

```bash
# 1. 의존성 설치
bash scripts/setup.sh

# 2. 단일 전사
python3 scripts/transcribe.py --input audio.wav --language ko

# 3. 고정확도 2-pass 전사
python3 scripts/transcribe_2pass.py --input audio.wav --pass 1 --language ko --json
# → AI 에이전트가 SKILL.md의 Step 2에 따라 6-agent 분석 수행
python3 scripts/transcribe_2pass.py --input audio.wav --pass 2 --language ko \
  --prompt "생성된 자연어 프롬프트"

# 4. 문서 생성
python3 scripts/memo_to_doc.py --input transcript.txt --output memo.md
```

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

테스트: 8분 한국어 IT 튜토리얼 영상, whisper-turbo 모델

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
| `record.py` | CLI 녹음 (WAV, 16kHz mono) |
| `memo_to_doc.py` | 전사 → 마크다운 문서 변환 |
| `benchmark_stt.py` | CER/WER 벤치마크 (SRT ground truth 비교) |
| `setup.sh` | 의존성 설치 (`--yes` 비대화 모드) |

## Requirements

- Apple Silicon Mac (M1+)
- Python 3.9+
- macOS 13+

## References

- [Whisper: Courtside Edition (arxiv:2602.18966)](https://arxiv.org/abs/2602.18966) — 2-Pass Multi-Agent Pipeline
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Apple Silicon Whisper

## License

Private repository. All rights reserved.
