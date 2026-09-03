"""Tests for word-boundary-aware timestamp refinement."""
from app.transcription.whisper import Word
from app.video.cutter import refine_bounds


def test_refine_bounds_pads_around_first_and_last_word():
    words = [Word(start=10.0, end=10.5, word=" Hello"), Word(start=10.6, end=11.2, word=" world.")]
    start, end = refine_bounds(9.5, 11.5, words)
    assert start == 10.0 - 0.15
    assert end == 11.2 + 0.25


def test_refine_bounds_never_goes_negative():
    words = [Word(start=0.05, end=0.3, word=" Hi")]
    start, end = refine_bounds(0.0, 1.0, words)
    assert start == 0.0


def test_refine_bounds_passthrough_when_no_words():
    start, end = refine_bounds(5.0, 10.0, [])
    assert (start, end) == (5.0, 10.0)
