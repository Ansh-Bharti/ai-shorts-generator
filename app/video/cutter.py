"""Cuts a clip from the source video, applies vertical reframing, and
snaps timestamps to word boundaries so cuts never land mid-word."""
from __future__ import annotations

import logging
from pathlib import Path

from app.transcription.whisper import Word
from app.utils.ffmpeg import cut_reframe_and_encode
from app.video.reframer import compute_crop

logger = logging.getLogger("shorts.cutter")

_START_PAD = 0.15
_END_PAD = 0.25


def refine_bounds(start: float, end: float, words: list[Word]) -> tuple[float, float]:
    """Snap to the first/last word actually spoken in range, with small
    padding, so we never cut mid-word or leave dead air at the edges."""
    if not words:
        return start, end
    refined_start = max(0.0, words[0].start - _START_PAD)
    refined_end = words[-1].end + _END_PAD
    return refined_start, refined_end


def render_clip(
    source_video: Path,
    start: float,
    end: float,
    words: list[Word],
    out_path: Path,
    output_width: int,
    output_height: int,
    crf: int,
    preset: str,
) -> Path:
    refined_start, refined_end = refine_bounds(start, end, words)
    crop_filter = compute_crop(source_video, refined_start, refined_end, output_width, output_height)

    logger.info(
        "Rendering clip %.2fs-%.2fs (orig %.2fs-%.2fs) -> %s",
        refined_start, refined_end, start, end, out_path.name,
    )
    return cut_reframe_and_encode(
        video_path=source_video,
        start=refined_start,
        end=refined_end,
        out_path=out_path,
        crop_filter=crop_filter,
        output_width=output_width,
        output_height=output_height,
        crf=crf,
        preset=preset,
    )
