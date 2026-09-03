"""Self-calibrating timing estimates.

Rather than guessing a fixed "shorts take N minutes" constant, we record how
fast each pipeline stage actually ran on this machine and use an exponential
moving average to predict future runs. The estimate gets more accurate the
more videos you process.
"""
from __future__ import annotations

from app.config import CACHE_DIR
from app.utils.files import read_json, write_json

STATS_FILE = CACHE_DIR / "timing_stats.json"

# Conservative defaults for an RTX 4060 8GB before any real run has been
# recorded. Overwritten by real measurements after the first run.
_DEFAULTS = {
    "whisper_seconds_per_video_second": 0.25,   # faster-whisper large-v3-turbo, GPU
    "whisper_model_load_seconds": 8.0,           # warm load from local cache
    "llm_seconds_per_candidate": 12.0,           # Qwen3 8B via Ollama
    "render_seconds_per_output_second": 3.0,     # cut+crop+encode+caption-burn per clip
    "samples": 0,
}

_ALPHA = 0.35  # weight given to the newest sample


def load_stats() -> dict:
    data = read_json(STATS_FILE)
    if not data:
        return dict(_DEFAULTS)
    merged = dict(_DEFAULTS)
    merged.update(data)
    return merged


def _update(stats: dict, key: str, sample: float) -> None:
    if sample <= 0:
        return
    current = stats.get(key, _DEFAULTS[key])
    stats[key] = current * (1 - _ALPHA) + sample * _ALPHA


def record_run(
    whisper_seconds: float | None = None,
    whisper_video_seconds: float | None = None,
    whisper_model_load_seconds: float | None = None,
    llm_seconds_per_candidate_samples: list[float] | None = None,
    render_seconds_per_output_second_samples: list[float] | None = None,
) -> None:
    stats = load_stats()

    if whisper_seconds and whisper_video_seconds:
        _update(stats, "whisper_seconds_per_video_second", whisper_seconds / whisper_video_seconds)
    if whisper_model_load_seconds:
        _update(stats, "whisper_model_load_seconds", whisper_model_load_seconds)
    for sample in (llm_seconds_per_candidate_samples or []):
        _update(stats, "llm_seconds_per_candidate", sample)
    for sample in (render_seconds_per_output_second_samples or []):
        _update(stats, "render_seconds_per_output_second", sample)

    stats["samples"] = stats.get("samples", 0) + 1
    write_json(STATS_FILE, stats)


def estimate_total_seconds(
    video_duration_seconds: float,
    num_candidates: int,
    num_clips: int,
    avg_clip_duration_seconds: float,
    whisper_cached: bool,
) -> dict:
    stats = load_stats()

    whisper_est = 0.0 if whisper_cached else (
        stats["whisper_model_load_seconds"]
        + video_duration_seconds * stats["whisper_seconds_per_video_second"]
    )
    scoring_est = num_candidates * stats["llm_seconds_per_candidate"]
    # Each clip is encoded twice (cut+reframe, then caption burn-in).
    render_est = num_clips * avg_clip_duration_seconds * stats["render_seconds_per_output_second"]

    total = whisper_est + scoring_est + render_est
    return {
        "whisper_seconds": round(whisper_est, 1),
        "scoring_seconds": round(scoring_est, 1),
        "rendering_seconds": round(render_est, 1),
        "total_seconds": round(total, 1),
        "calibrated": stats.get("samples", 0) > 0,
    }
