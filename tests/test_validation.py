"""Tests for input validation and disk space checks."""
import pytest
from pathlib import Path

from app.utils.validation import ValidationError, check_disk_space, validate_input_file


def test_rejects_missing_file(tmp_path):
    with pytest.raises(ValidationError, match="not found"):
        validate_input_file(tmp_path / "missing.mp4")


def test_rejects_unsupported_extension(tmp_path):
    f = tmp_path / "video.avi"
    f.write_bytes(b"data")
    with pytest.raises(ValidationError, match="Unsupported file type"):
        validate_input_file(f)


def test_rejects_empty_file(tmp_path):
    f = tmp_path / "video.mp4"
    f.touch()
    with pytest.raises(ValidationError, match="empty"):
        validate_input_file(f)


def test_accepts_valid_mp4(tmp_path):
    f = tmp_path / "video.mp4"
    f.write_bytes(b"fake video bytes")
    assert validate_input_file(f) == f


def test_check_disk_space_raises_when_insufficient(tmp_path):
    with pytest.raises(ValidationError, match="Not enough disk space"):
        check_disk_space(tmp_path, required_bytes=10 ** 18)


def test_check_disk_space_passes_for_small_requirement(tmp_path):
    check_disk_space(tmp_path, required_bytes=1024)
