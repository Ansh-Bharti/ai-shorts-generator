"""Tests for YouTube URL detection (network calls are not exercised here)."""
from app.utils.youtube import is_youtube_url


def test_detects_standard_watch_url():
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_detects_short_youtu_be_url():
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")


def test_detects_shorts_url():
    assert is_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")


def test_detects_url_without_scheme():
    assert is_youtube_url("youtube.com/watch?v=dQw4w9WgXcQ")


def test_detects_url_with_extra_params():
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PL123")


def test_rejects_plain_text():
    assert not is_youtube_url("this is not a url at all")


def test_rejects_other_video_sites():
    assert not is_youtube_url("https://vimeo.com/12345678")


def test_rejects_local_filename():
    assert not is_youtube_url("my_video.mp4")


def test_strips_whitespace():
    assert is_youtube_url("  https://youtu.be/dQw4w9WgXcQ  ")
