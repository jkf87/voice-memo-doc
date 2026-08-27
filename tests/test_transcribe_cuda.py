import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_backends import compact_result
import transcribe_cuda


class FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def json(self):
        return {
            "text": "CUDA 전사 성공",
            "language": "ko",
            "duration": 10.0,
            "segments": [{"start": 0, "end": 10, "text": " CUDA 전사 성공"}],
            "performance": {"elapsed_seconds": 1.0, "compute_type": "int8_float16"},
        }


class CudaTranscriptionTest(unittest.TestCase):
    def test_large_v3_turbo_local_alias(self):
        self.assertEqual(transcribe_cuda.resolve_cuda_model("large-v3-turbo"), "turbo")

    def test_api_model_alias_is_local_turbo(self):
        self.assertEqual(transcribe_cuda.resolve_cuda_model("whisper-large-v3-turbo"), "turbo")

    def test_full_model_id_is_preserved(self):
        model_id = "example-org/custom-faster-whisper"
        self.assertEqual(transcribe_cuda.resolve_cuda_model(model_id), model_id)

    def test_cuda_api_normalizes_openai_response(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            fake_requests = types.SimpleNamespace(post=lambda *args, **kwargs: FakeResponse())
            with patch.dict(sys.modules, {"requests": fake_requests}):
                result = transcribe_cuda.transcribe_via_api(
                    audio.name,
                    server_url="http://gpu.test:8001",
                    api_key="test-key",
                )

        self.assertEqual(result["text"], "CUDA 전사 성공")
        self.assertEqual(result["backend"], "cuda-api")
        self.assertEqual(result["compute_type"], "int8_float16")
        self.assertEqual(result["duration_sec"], 10.0)
        self.assertEqual(result["segments"][0]["text"], "CUDA 전사 성공")

    def test_benchmark_summary_omits_full_transcript(self):
        result = compact_result(
            {"backend": "cuda-api", "text": "테스트", "segments": [{"text": "테스트"}]}
        )
        self.assertNotIn("text", result)
        self.assertNotIn("segments", result)
        self.assertEqual(result["text_characters"], 3)
        self.assertEqual(result["segment_count"], 1)


if __name__ == "__main__":
    unittest.main()
