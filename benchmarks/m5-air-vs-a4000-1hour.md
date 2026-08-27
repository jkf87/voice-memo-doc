# M5 Air MLX vs RTX A4000 CUDA

측정일: 2026-08-27 KST

## 결과

| Backend | Hardware | Model/runtime | Audio | Elapsed | Speed |
|---|---|---|---:|---:|---:|
| MLX | MacBook Air M5, 32GB | `mlx-community/whisper-large-v3-turbo`, mlx-whisper 0.4.3 | 3,600.0초 | 185.000초 | 19.50x |
| CUDA API | NVIDIA RTX A4000 16GB | `large-v3-turbo`, faster-whisper 1.2.1, CTranslate2 4.8.1, `int8_float16` | 3,600.0초 | 92.821초 | 38.78x |

CUDA 서버 내부 처리 시간은 91.678초(39.27x)였습니다. 92.821초는 Tailscale을 통한 파일 업로드와 응답 수신을 포함한 Mac 클라이언트 기준입니다. 이 측정에서 A4000 CUDA는 M5 Air MLX보다 1.99배 빨랐습니다.

## 입력과 조건

- 입력: 1시간짜리 반복 한국어 합성 음원(OGG, 10MB)
- 길이: 3,600.0065초
- SHA-256: `95a4a12c298638c5fc887a12ca837618f094a5644f17d5796427b4f7577d244a`
- 언어 고정: `ko`
- Pass 1 조건: prompt 없음, `condition_on_previous_text=false`
- CUDA 서버는 모델이 이미 적재된 상주 API 상태
- 측정 당시 A4000에는 Qwen 서버도 함께 적재돼 있었고 총 GPU 메모리 사용량은 약 15.3GB
- MLX 측정 시간은 `mlx_whisper.transcribe()` 호출을 포함하고 오디오를 16kHz 배열로 변환하는 사전 디코딩 시간은 제외
- CUDA 클라이언트 측정 시간은 HTTP 업로드 및 응답 수신 포함

## 해석 제한

이 음원은 동일 문장이 반복되는 깨끗한 합성 음성입니다. 따라서 이 결과는 장비별 처리량 비교에는 사용할 수 있지만 실제 회의, 강의, 다중 화자, 소음 환경의 정확도 비교에는 사용할 수 없습니다. 정확도 비교에는 별도의 자연 발화 음원과 정답 SRT가 필요합니다.

원시 결과는 `m5-air-vs-a4000-1hour.json`에 저장했습니다.
