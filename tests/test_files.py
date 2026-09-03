"""Tests for filesystem helpers: fingerprinting, JSON I/O, safe filenames."""
from pathlib import Path

from app.utils.files import file_fingerprint, read_json, safe_stem, write_json


def test_fingerprint_stable_for_unchanged_file(tmp_path):
    f = tmp_path / "video.mp4"
    f.write_bytes(b"content")
    assert file_fingerprint(f) == file_fingerprint(f)


def test_fingerprint_changes_when_file_size_changes(tmp_path):
    f = tmp_path / "video.mp4"
    f.write_bytes(b"content")
    fp1 = file_fingerprint(f)
    f.write_bytes(b"different content, longer now")
    fp2 = file_fingerprint(f)
    assert fp1 != fp2


def test_write_then_read_json_roundtrip(tmp_path):
    path = tmp_path / "data.json"
    data = {"a": 1, "b": [1, 2, 3]}
    write_json(path, data)
    assert read_json(path) == data


def test_read_json_missing_file_returns_none(tmp_path):
    assert read_json(tmp_path / "missing.json") is None


def test_safe_stem_strips_unsafe_characters():
    assert safe_stem(Path("my video?!.mp4")) == "my video"
