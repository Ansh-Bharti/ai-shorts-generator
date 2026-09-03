"""Tests for candidate window generation from a transcript."""
from app.analysis.candidate_generator import generate_candidates
from app.transcription.whisper import Transcript, Segment, Word


def make_word(start, end, text):
    return Word(start=start, end=end, word=text, probability=0.99)


def build_transcript(sentences: list[list[tuple[float, float, str]]]) -> Transcript:
    """sentences: list of sentences, each a list of (start, end, word_text)."""
    segments = []
    for sent in sentences:
        words = [make_word(s, e, t) for s, e, t in sent]
        segments.append(Segment(start=words[0].start, end=words[-1].end, text="".join(t for _, _, t in sent), words=words))
    return Transcript(source="test.mp4", duration=segments[-1].end, language="en", model="test", segments=segments)


def _sentence(start, word_count=5, gap=0.3):
    words = []
    t = start
    for i in range(word_count):
        w_end = t + 0.4
        text = f" word{i}." if i == word_count - 1 else f" word{i}"
        words.append((t, w_end, text))
        t = w_end + 0.05
    return words, t + gap


def test_generates_at_least_one_candidate_within_bounds():
    sentences = []
    t = 0.0
    for _ in range(20):
        sent, t = _sentence(t, word_count=8, gap=0.8)
        sentences.append(sent)
    transcript = build_transcript(sentences)

    candidates = generate_candidates(
        transcript, min_seconds=15.0, max_seconds=60.0,
        silence_gap_threshold=0.6, max_silence_ratio=0.5, max_candidates=40,
    )
    assert len(candidates) > 0
    for c in candidates:
        assert c.duration >= 15.0 - 0.01
        assert c.duration <= 60.0 + 0.01


def test_no_candidates_from_empty_transcript():
    transcript = Transcript(source="test.mp4", duration=0, language="en", model="test", segments=[])
    candidates = generate_candidates(
        transcript, min_seconds=15.0, max_seconds=60.0,
        silence_gap_threshold=0.6, max_silence_ratio=0.5, max_candidates=40,
    )
    assert candidates == []


def test_respects_max_candidates_cap():
    sentences = []
    t = 0.0
    for _ in range(200):
        sent, t = _sentence(t, word_count=8, gap=0.8)
        sentences.append(sent)
    transcript = build_transcript(sentences)

    candidates = generate_candidates(
        transcript, min_seconds=15.0, max_seconds=60.0,
        silence_gap_threshold=0.6, max_silence_ratio=0.5, max_candidates=10,
    )
    assert len(candidates) <= 10
