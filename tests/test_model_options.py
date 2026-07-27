import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from model_options import DEFAULT_MODEL, resolve_model
import transcribe_2pass


class ModelOptionsTest(unittest.TestCase):
    def test_large_v3_turbo_is_default(self):
        self.assertEqual(
            DEFAULT_MODEL,
            "mlx-community/whisper-large-v3-turbo",
        )

    def test_large_v3_turbo_alias(self):
        self.assertEqual(
            resolve_model("large-v3-turbo"),
            "mlx-community/whisper-large-v3-turbo",
        )

    def test_full_model_path_is_preserved(self):
        model_path = "example-org/custom-whisper-model"
        self.assertEqual(resolve_model(model_path), model_path)

    def test_two_pass_resolves_alias_before_transcribing(self):
        calls = []

        def fake_transcribe(audio, **kwargs):
            calls.append(kwargs)
            return {"text": "코난쌤", "language": "ko"}

        fake_module = types.SimpleNamespace(transcribe=fake_transcribe)
        with patch.dict(sys.modules, {"mlx_whisper": fake_module}):
            result = transcribe_2pass.transcribe(
                [0.0] * transcribe_2pass.SAMPLE_RATE,
                model="large-v3-turbo",
                language="ko",
            )

        self.assertEqual(
            calls[0]["path_or_hf_repo"],
            "mlx-community/whisper-large-v3-turbo",
        )
        self.assertEqual(
            result["model"],
            "mlx-community/whisper-large-v3-turbo",
        )


if __name__ == "__main__":
    unittest.main()
