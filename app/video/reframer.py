"""Automatic vertical (9:16) reframing.

v1: samples a handful of frames from the clip, runs OpenCV's Haar-cascade
face detector, and centers a 9:16 crop on the average face position.
Falls back to a plain center crop when no face is found. Deliberately
modular (single `compute_crop`) so a stronger vision model can replace
the detection step later without touching callers.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2

from app import config
from app.utils.ffmpeg import probe_resolution, sample_frames

logger = logging.getLogger("shorts.reframer")

_face_cascade = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _detect_face_center_x(frame_paths: list[Path]) -> float | None:
    """Returns the average normalized (0-1) x-center of the largest detected
    face across frames, or None if no face was found in any frame."""
    cascade = _get_cascade()
    weighted_centers = []
    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            continue
        # Largest face = most likely the main subject.
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        center_x = (fx + fw / 2) / w
        weighted_centers.append((center_x, fw * fh))

    if not weighted_centers:
        return None

    total_weight = sum(weight for _, weight in weighted_centers)
    return sum(cx * weight for cx, weight in weighted_centers) / total_weight


def compute_crop(
    video_path: Path,
    start: float,
    end: float,
    output_width: int,
    output_height: int,
    sample_count: int = 5,
) -> str:
    """Returns an ffmpeg crop filter string, e.g. 'crop=608:1080:236:0'."""
    src_w, src_h = probe_resolution(video_path)
    target_aspect = output_width / output_height  # e.g. 9/16

    if src_w / src_h > target_aspect:
        # Source is wider than target: crop width, keep full height.
        crop_h = src_h
        crop_w = int(round(src_h * target_aspect))
    else:
        # Source is taller/narrower: crop height, keep full width.
        crop_w = src_w
        crop_h = int(round(src_w / target_aspect))

    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)

    center_x_norm = 0.5
    frame_dir = config.TEMP_DIR / f"frames_{round(start * 1000)}"
    try:
        frames = sample_frames(video_path, start, end, sample_count, frame_dir)
        detected = _detect_face_center_x(frames)
        if detected is not None:
            center_x_norm = detected
            logger.info("Face-based crop for %.1fs-%.1fs: center_x=%.2f", start, end, center_x_norm)
        else:
            logger.info("No face detected for %.1fs-%.1fs, using center crop", start, end)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    center_x_px = center_x_norm * src_w
    x = int(round(center_x_px - crop_w / 2))
    x = max(0, min(x, src_w - crop_w))
    y = int(round((src_h - crop_h) / 2))
    y = max(0, min(y, src_h - crop_h))

    return f"crop={crop_w}:{crop_h}:{x}:{y}"
