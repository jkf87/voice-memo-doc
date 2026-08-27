# CUDA 실행 가이드

`transcribe_cuda.py`는 두 가지 방식으로 실행됩니다.

- `local`: 현재 컴퓨터의 NVIDIA GPU에서 `faster-whisper`를 직접 실행
- `api`: 다른 NVIDIA 컴퓨터의 OpenAI 호환 전사 API를 호출

Mac은 CUDA를 직접 실행할 수 없으므로 원격 NVIDIA GPU를 쓸 때 `api` 모드를 선택합니다.

## CUDA local

테스트한 서버 스택은 다음과 같습니다.

- NVIDIA RTX A4000 16GB
- faster-whisper 1.2.1
- CTranslate2 4.8.1
- CUDA 12 계열과 cuDNN 9 호환 런타임
- `large-v3-turbo`, `int8_float16`

프로젝트용 가상환경에서 설치합니다.

```bash
python3 -m pip install -r requirements-cuda.txt
python3 scripts/transcribe_cuda.py --backend local \
  --input audio.wav --language ko --compute-type int8_float16 \
  --output transcript.json --json
```

`nvidia-smi`가 정상이어도 CTranslate2가 CUDA 또는 cuDNN DLL을 찾지 못할 수 있습니다. 이 경우 CUDA 12/cuDNN 9 런타임의 라이브러리 경로가 실행 프로세스의 `PATH`에 포함됐는지 확인합니다.

## CUDA API

클라이언트에는 NVIDIA 드라이버가 필요하지 않습니다. 서버 URL과 비밀키만 환경변수로 전달합니다.

GPU 서버에서 포함된 API를 실행합니다.

```bash
python3 -m pip install -r requirements-cuda-server.txt
export WHISPER_API_KEY='YOUR_PRIVATE_KEY'
python3 scripts/serve_cuda.py
```

기본 주소는 `http://127.0.0.1:8001`입니다. `WHISPER_HOST`, `WHISPER_PORT`, `WHISPER_MODEL_PATH`, `WHISPER_COMPUTE_TYPE` 환경변수로 변경할 수 있습니다.

```bash
export VOICE_MEMO_CUDA_URL='http://GPU-SERVER:8001'
export VOICE_MEMO_CUDA_API_KEY='YOUR_PRIVATE_KEY'

python3 scripts/transcribe_cuda.py --backend api \
  --input audio.wav --language ko --output transcript.json --json
```

URL에 `/v1`을 붙여도 되고 생략해도 됩니다. 스크립트는 `/v1/audio/transcriptions`로 정규화합니다.

서버는 다음 multipart 필드를 받는 OpenAI 호환 형태여야 합니다.

- `file`
- `model`
- `language`
- `response_format=verbose_json`
- `temperature`
- `prompt`
- `condition_on_previous_text`

2-pass 실행에서는 `prompt`와 `condition_on_previous_text=true`가 서버까지 전달됩니다.

## 출력과 속도

출력 확장자 또는 옵션으로 형식을 선택합니다.

```bash
# 일반 텍스트
python3 scripts/transcribe_cuda.py --backend api --input audio.wav --output transcript.txt

# 구간 타임스탬프 JSON
python3 scripts/transcribe_cuda.py --backend api --input audio.wav --output transcript.json --json

# 마크다운 또는 SRT
python3 scripts/transcribe_cuda.py --backend api --input audio.wav --output transcript.md
python3 scripts/transcribe_cuda.py --backend api --input audio.wav --output transcript.srt --srt
```

JSON에는 `elapsed_sec`, `server_elapsed_sec`, `rtf`, `device`, `compute_type`이 포함됩니다. 이 프로젝트의 `rtf`는 통상적인 elapsed/audio 비율이 아니라 읽기 쉬운 **실시간 대비 처리 배수(audio/elapsed)** 입니다.

## 보안과 운영

- 전사 서버를 인터넷에 직접 공개하지 않습니다.
- Tailscale 같은 사설망과 Bearer 인증을 함께 사용합니다.
- API 키를 README, 셸 기록, 결과 JSON에 저장하지 않습니다.
- 큰 파일은 업로드 시간이 포함되므로 `elapsed_sec`와 `server_elapsed_sec`를 함께 봅니다.
- 상주 API는 모델이 이미 적재된 상태이고, 단발 local CLI는 모델 로딩 시간이 추가될 수 있습니다.

## 속도 비교

```bash
python3 scripts/benchmark_backends.py --input audio.ogg \
  --backends mlx,cuda --language ko \
  --cuda-url "$VOICE_MEMO_CUDA_URL" \
  --cuda-api-key "$VOICE_MEMO_CUDA_API_KEY" \
  --output benchmarks/result.json
```

동일한 음원, 언어, 모델 계열로 비교합니다. 반복 합성음은 처리량 확인에는 유용하지만 자연 발화 정확도나 잡음 강건성을 대표하지 않습니다.
