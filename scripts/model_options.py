"""Shared Whisper MLX model aliases and defaults."""

MODELS = {
    "turbo": "mlx-community/whisper-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
}

DEFAULT_MODEL = MODELS["large-v3-turbo"]


def resolve_model(model_name: str) -> str:
    """Resolve a short model alias while preserving full Hugging Face paths."""
    return MODELS.get(model_name, model_name)
