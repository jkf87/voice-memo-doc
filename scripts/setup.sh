#!/bin/bash
# Voice Memo Doc — dependency installer
# Usage: bash scripts/setup.sh [--yes]
# --yes: non-interactive mode (skip all prompts, install everything)

set -e

YES_MODE=false
[[ "$1" == "--yes" || "$1" == "-y" ]] && YES_MODE=true

echo "=== Voice Memo Doc Setup ==="

# Detect Python
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python3 not found. Install Python 3.9+ first."
    exit 1
fi
echo "Python: $($PYTHON --version) ($PYTHON)"

# Check Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "WARNING: This skill is optimized for Apple Silicon (arm64). Current: $ARCH"
fi

# Brew dependencies
echo ""
echo "[1/2] Installing system dependencies (brew)..."
if command -v brew &>/dev/null; then
    brew list portaudio &>/dev/null || brew install portaudio
    echo "  portaudio: OK"
else
    echo "  WARNING: Homebrew not found. Install portaudio manually."
fi

# Core pip dependencies
echo ""
echo "[2/2] Installing Python packages..."
"$PYTHON" -m pip install --quiet mlx-whisper sounddevice soundfile
echo "  mlx-whisper sounddevice soundfile: OK"

echo ""
echo "=== Setup Complete ==="
echo "Test: $PYTHON scripts/transcribe_2pass.py --input audio.wav --pass 1 --language ko"
