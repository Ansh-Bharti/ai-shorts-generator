"""Optional post-processing effects. All OFF by default — the default
output is video + original audio + captions, per spec. Each effect is a
self-contained step that takes a rendered clip and returns a new one.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

from app.config import Config
from app.utils.ffmpeg import (
    apply_zoom, detect_silence, mix_audio_track, overlay_sfx,
    probe_duration, remove_silence_segments,
)

logger = logging.getLogger("shorts.effects")


def maybe_remove_silence(video_path: Path, work_dir: Path, cfg: Config) -> Path:
    if not cfg.enable_silence_removal:
        return video_path
    duration = probe_duration(video_path)
    silent_ranges = detect_silence(video_path)
    if not silent_ranges:
        return video_path
    out_path = work_dir / f"{video_path.stem}_nosilence.mp4"
    logger.info("Removing %d silent range(s) from %s", len(silent_ranges), video_path.name)
    return remove_silence_segments(
        video_path, out_path, silent_ranges, duration, cfg.video_crf, cfg.video_preset,
    )


def maybe_apply_zoom(video_path: Path, work_dir: Path, cfg: Config) -> Path:
    if not cfg.enable_zoom:
        return video_path
    out_path = work_dir / f"{video_path.stem}_zoom.mp4"
    logger.info("Applying zoom effect to %s", video_path.name)
    return apply_zoom(video_path, out_path, cfg.output_width, cfg.output_height, cfg.video_crf, cfg.video_preset)


def maybe_add_music(video_path: Path, work_dir: Path, cfg: Config) -> Path:
    if not cfg.enable_music:
        return video_path
    music_dir = Path(cfg.music_dir)
    if not music_dir.exists():
        logger.warning("Music enabled but %s does not exist — skipping.", music_dir)
        return video_path
    tracks = [p for p in music_dir.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".m4a")]
    if not tracks:
        logger.warning("Music enabled but no audio files found in %s — skipping.", music_dir)
        return video_path
    track = random.choice(tracks)
    out_path = work_dir / f"{video_path.stem}_music.mp4"
    logger.info("Mixing background music (%s) into %s", track.name, video_path.name)
    return mix_audio_track(video_path, track, out_path)


def maybe_add_sfx(video_path: Path, work_dir: Path, cfg: Config) -> Path:
    if not cfg.enable_sfx:
        return video_path
    sfx_dir = Path(cfg.sfx_dir)
    if not sfx_dir.exists():
        logger.warning("SFX enabled but %s does not exist — skipping.", sfx_dir)
        return video_path
    clips = [p for p in sfx_dir.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".m4a")]
    if not clips:
        logger.warning("SFX enabled but no audio files found in %s — skipping.", sfx_dir)
        return video_path
    sfx = random.choice(clips)
    out_path = work_dir / f"{video_path.stem}_sfx.mp4"
    logger.info("Overlaying SFX (%s) at clip start of %s", sfx.name, video_path.name)
    return overlay_sfx(video_path, sfx, out_path, at_time=0.0)


def apply_optional_effects(video_path: Path, work_dir: Path, cfg: Config) -> Path:
    """Applies enabled effects in order: silence removal -> zoom -> sfx -> music.
    Each step is a no-op passthrough if its flag is off."""
    result = video_path
    result = maybe_remove_silence(result, work_dir, cfg)
    result = maybe_apply_zoom(result, work_dir, cfg)
    result = maybe_add_sfx(result, work_dir, cfg)
    result = maybe_add_music(result, work_dir, cfg)
    return result
