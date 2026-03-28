#!/usr/bin/env python3
"""
STT Accuracy Benchmark — compares mlx_whisper transcription against ground truth SRT.
Metrics: CER (Character Error Rate), sentence-level accuracy, hallucination count.
"""
import os
import re
import sys
import time
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# ── Config ──
VIDEO_PATH = Path(os.environ.get("BENCH_VIDEO", "test_video.mp4"))
SRT_PATH = Path(os.environ.get("BENCH_SRT", "test_ground_truth.srt"))
SAMPLE_RATE = 16000

# Models to benchmark
MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "small": "mlx-community/whisper-small-mlx",
}

# ── SRT Parser ──
def parse_srt(path):
    """Parse SRT file → list of (start_sec, end_sec, text)."""
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r'\n\s*\n', content.strip())
    entries = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        # Parse timestamp
        ts_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
            lines[1]
        )
        if not ts_match:
            continue
        g = ts_match.groups()
        start = int(g[0])*3600 + int(g[1])*60 + int(g[2]) + int(g[3])/1000
        end = int(g[4])*3600 + int(g[5])*60 + int(g[6]) + int(g[7])/1000
        text = ' '.join(lines[2:]).strip()
        entries.append((start, end, text))
    return entries


def srt_to_full_text(entries):
    """Concatenate all SRT text into single string."""
    return ' '.join(e[2] for e in entries)


# ── Audio Extraction ──
def extract_audio(video_path, sr=16000):
    """Extract audio from video as float32 numpy array."""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    subprocess.run([
        'ffmpeg', '-y', '-i', str(video_path),
        '-ac', '1', '-ar', str(sr), '-acodec', 'pcm_s16le',
        tmp.name
    ], capture_output=True)
    audio, _ = sf.read(tmp.name, dtype='float32')
    Path(tmp.name).unlink()
    return audio


# ── CER (Character Error Rate) ──
def cer(reference, hypothesis):
    """Compute Character Error Rate using edit distance."""
    ref = reference.replace(' ', '')
    hyp = hypothesis.replace(' ', '')
    if not ref:
        return 0.0 if not hyp else 1.0

    # Levenshtein distance
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / m


# ── Hallucination Detection ──
def count_hallucinations(text):
    """Count hallucination patterns in transcribed text."""
    count = 0
    # Repeated words
    words = text.split()
    for i in range(len(words) - 3):
        if len(set(w.lower() for w in words[i:i+4])) == 1:
            count += 1
    # Known hallucination phrases
    halluc = ["도움이 되었습니다", "감사합니다", "구독과 좋아요", "날씨였습니다"]
    for h in halluc:
        count += text.count(h)
    return count


# ── Main Benchmark ──
def run_benchmark(model_name, model_repo, audio, ground_truth,
                  no_speech_threshold=0.6, compression_ratio_threshold=2.4):
    """Run transcription and compute metrics."""
    from mlx_whisper import transcribe as mlx_transcribe

    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_repo})")
    print(f"  no_speech_threshold={no_speech_threshold}")
    print(f"  compression_ratio_threshold={compression_ratio_threshold}")
    print(f"{'='*60}")

    t0 = time.time()
    result = mlx_transcribe(
        audio,
        path_or_hf_repo=model_repo,
        language="ko",
        verbose=False,
        condition_on_previous_text=True,
        no_speech_threshold=no_speech_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
    )
    t1 = time.time()

    hypothesis = result.get("text", "").strip()

    # Metrics
    c = cer(ground_truth, hypothesis)
    halluc = count_hallucinations(hypothesis)
    duration = len(audio) / SAMPLE_RATE
    speed = duration / (t1 - t0)

    print(f"\n  Duration: {duration:.1f}s audio in {t1-t0:.1f}s ({speed:.1f}x realtime)")
    print(f"  CER: {c*100:.1f}%")
    print(f"  Hallucinations: {halluc}")
    print(f"  Output length: {len(hypothesis)} chars (ref: {len(ground_truth)} chars)")

    # Show first 200 chars of output
    print(f"\n  First 200 chars of transcription:")
    print(f"  {hypothesis[:200]}...")

    # Show first 200 chars of reference
    print(f"\n  First 200 chars of reference:")
    print(f"  {ground_truth[:200]}...")

    return {
        "model": model_name,
        "cer": c,
        "hallucinations": halluc,
        "time": t1 - t0,
        "speed": speed,
        "hypothesis": hypothesis,
        "no_speech_threshold": no_speech_threshold,
        "compression_ratio_threshold": compression_ratio_threshold,
    }


def main():
    print("=== STT Accuracy Benchmark ===\n")

    # Parse ground truth
    entries = parse_srt(SRT_PATH)
    ground_truth = srt_to_full_text(entries)
    print(f"Ground truth: {len(entries)} segments, {len(ground_truth)} chars")

    # Extract audio
    print(f"Extracting audio from {VIDEO_PATH.name}...")
    audio = extract_audio(VIDEO_PATH, SAMPLE_RATE)
    print(f"Audio: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.1f}s)")

    # Run benchmarks
    results = []

    # Test different parameter combinations
    configs = [
        # (model, no_speech_thresh, compression_ratio_thresh)
        ("turbo", 0.3, 2.4),   # baseline (old defaults)
        ("turbo", 0.6, 2.0),   # current settings
        ("turbo", 0.6, 2.4),   # higher compression threshold
        ("turbo", 0.8, 1.8),   # aggressive filtering
        ("small", 0.6, 2.4),   # small model
    ]

    for model_key, nst, crt in configs:
        repo = MODELS[model_key]
        r = run_benchmark(model_key, repo, audio, ground_truth,
                         no_speech_threshold=nst,
                         compression_ratio_threshold=crt)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"{'BENCHMARK SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Model':<8} {'NST':>5} {'CRT':>5} {'CER%':>7} {'Halluc':>7} {'Speed':>7} {'Time':>7}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['model']:<8} {r['no_speech_threshold']:>5.1f} {r['compression_ratio_threshold']:>5.1f} "
              f"{r['cer']*100:>6.1f}% {r['hallucinations']:>7d} {r['speed']:>6.1f}x {r['time']:>6.1f}s")
    print(f"{'='*80}")

    # Best result
    best = min(results, key=lambda r: r['cer'])
    print(f"\nBest: {best['model']} (NST={best['no_speech_threshold']}, CRT={best['compression_ratio_threshold']}) → CER={best['cer']*100:.1f}%")


if __name__ == "__main__":
    main()
