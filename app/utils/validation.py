"""Input validation and disk-space checks."""
from __future__ import annotations

import shutil
from pathlib import Path

from app.config import SUPPORTED_INPUT_EXTENSIONS


class ValidationError(Exception):
    pass


def validate_input_file(path: Path) -> Path:
    """Raise ValidationError with a clear message if the input file is unusable."""
    if not path.exists():
        raise ValidationError(f"Input file not found: {path}")
    if not path.is_file():
        raise ValidationError(f"Input path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))
        raise ValidationError(
            f"Unsupported file type '{path.suffix}'. Supported: {allowed}"
        )
    if path.stat().st_size == 0:
        raise ValidationError(f"Input file is empty: {path}")
    return path


def check_disk_space(directory: Path, required_bytes: int) -> None:
    """Raise ValidationError if directory's drive doesn't have enough free space."""
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < required_bytes:
        need_gb = required_bytes / 1e9
        have_gb = free / 1e9
        raise ValidationError(
            f"Not enough disk space on {directory}: need ~{need_gb:.1f} GB, "
            f"only {have_gb:.1f} GB free"
        )


def estimate_required_space(source_video: Path) -> int:
    """Rough estimate: source size + 2x headroom for temp audio/clips."""
    return int(source_video.stat().st_size * 2.5) + 500_000_000
