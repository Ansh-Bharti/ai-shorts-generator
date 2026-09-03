"""CLI entry point.

Usage:
    python -m app.main input.mp4
    python -m app.main input.mp4 --clips 10
    python -m app.main input.mp4 --min-score 7.5
    python -m app.main input.mp4 --force-retranscribe
    python -m app.main "https://www.youtube.com/watch?v=XXXXXXXXXXX" --clips 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app import config
from app.pipeline.pipeline import run_pipeline
from app.utils.youtube import YouTubeError, download_video, is_youtube_url


def _resolve_local_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidate = config.INPUT_DIR / raw
    if candidate.exists():
        return candidate
    if p.exists():
        return p
    return candidate  # let downstream validation produce a clear error


def _resolve_input(raw: str) -> Path:
    if is_youtube_url(raw):
        def _on_progress(frac: float, message: str) -> None:
            print(f"\r[Download] {message}" + " " * 10, end="", flush=True)

        print(f"Detected YouTube URL, downloading into {config.INPUT_DIR} ...")
        try:
            path = download_video(raw, on_progress=_on_progress)
        except YouTubeError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)
        print(f"\nDownloaded: {path.name}")
        return path
    return _resolve_local_path(raw)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Fully local AI Shorts Generator (faster-whisper + Qwen3 8B via Ollama).",
    )
    parser.add_argument("input", help="Video filename in input/, a full path, or a YouTube URL.")
    parser.add_argument("--clips", type=int, default=None, help="Number of Shorts to generate (default from config).")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum weighted score 0-10 (default from config).")
    parser.add_argument("--force-retranscribe", action="store_true", help="Ignore cached transcript and re-run Whisper.")
    args = parser.parse_args()

    config.save_default_config()

    input_path = _resolve_input(args.input)
    run_pipeline(
        input_path=input_path,
        num_clips=args.clips,
        min_score=args.min_score,
        force_retranscribe=args.force_retranscribe,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
