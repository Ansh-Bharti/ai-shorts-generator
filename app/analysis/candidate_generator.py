"""Turn a timestamped transcript into a manageable set of candidate Shorts
windows (15-60s) that start/end on sentence boundaries, instead of sending
the whole transcript to the LLM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.transcription.whisper import Transcript, Word

logger = logging.getLogger("shorts.candidates")

SENTENCE_ENDERS = (".", "!", "?", "…")


@dataclass
class Sentence:
    start: float
    end: float
    text: str
    words: list[Word]
    gap_before: float  # silence gap since previous sentence ended


@dataclass
class Candidate:
    id: str
    start: float
    end: float
    text: str
    words: list[Word]

    @property
    def duration(self) -> float:
        return self.end - self.start


def _build_sentences(transcript: Transcript) -> list[Sentence]:
    """Merge words across segments into sentence-like units using punctuation
    and pause boundaries."""
    all_words: list[Word] = []
    for seg in transcript.segments:
        all_words.extend(seg.words if seg.words else [])

    if not all_words:
        # Fall back to whole segments as "sentences" if no word timestamps.
        sentences = []
        prev_end = 0.0
        for seg in transcript.segments:
            sentences.append(Sentence(
                start=seg.start, end=seg.end, text=seg.text,
                words=[], gap_before=max(0.0, seg.start - prev_end),
            ))
            prev_end = seg.end
        return sentences

    sentences: list[Sentence] = []
    current_words: list[Word] = []
    prev_end = 0.0

    for w in all_words:
        if not current_words:
            gap_before = max(0.0, w.start - prev_end)
        current_words.append(w)
        stripped = w.word.strip()
        if stripped.endswith(SENTENCE_ENDERS):
            text = "".join(x.word for x in current_words).strip()
            sentences.append(Sentence(
                start=current_words[0].start,
                end=current_words[-1].end,
                text=text,
                words=list(current_words),
                gap_before=gap_before,
            ))
            prev_end = current_words[-1].end
            current_words = []

    if current_words:
        text = "".join(x.word for x in current_words).strip()
        sentences.append(Sentence(
            start=current_words[0].start,
            end=current_words[-1].end,
            text=text,
            words=list(current_words),
            gap_before=max(0.0, current_words[0].start - prev_end),
        ))

    return sentences


def _speech_ratio(words: list[Word], duration: float) -> float:
    if duration <= 0:
        return 0.0
    spoken = sum(w.end - w.start for w in words)
    return spoken / duration


def generate_candidates(
    transcript: Transcript,
    min_seconds: float,
    max_seconds: float,
    silence_gap_threshold: float,
    max_silence_ratio: float,
    max_candidates: int,
) -> list[Candidate]:
    sentences = _build_sentences(transcript)
    if not sentences:
        logger.warning("No sentences could be built from transcript — no candidates generated.")
        return []

    # Natural start points: the very first sentence, or any sentence preceded
    # by a pause (likely a fresh thought / good hook).
    start_indices = [
        i for i, s in enumerate(sentences)
        if i == 0 or s.gap_before >= silence_gap_threshold
    ]
    if not start_indices:
        start_indices = list(range(len(sentences)))

    raw_candidates: list[Candidate] = []

    for i in start_indices:
        start_time = sentences[i].start
        acc_words: list[Word] = []
        acc_text_parts: list[str] = []
        last_good_end_idx = None

        for j in range(i, len(sentences)):
            if j > i and sentences[j].gap_before >= silence_gap_threshold * 3:
                # Big pause — likely a topic change, stop extending.
                break

            acc_words.extend(sentences[j].words)
            acc_text_parts.append(sentences[j].text)
            duration = sentences[j].end - start_time

            if duration > max_seconds:
                break

            if duration >= min_seconds:
                last_good_end_idx = j
                # Record a candidate ending here (a valid, complete-thought window).
                ratio = _speech_ratio(acc_words, duration)
                if ratio >= (1.0 - max_silence_ratio):
                    cand_id = f"c_{round(start_time * 1000)}_{round(sentences[j].end * 1000)}"
                    raw_candidates.append(Candidate(
                        id=cand_id,
                        start=start_time,
                        end=sentences[j].end,
                        text=" ".join(acc_text_parts).strip(),
                        words=list(acc_words),
                    ))
                # Only keep the first (shortest complete) and last (fullest)
                # candidate per start to avoid near-duplicate spam.
                if len([c for c in raw_candidates if c.start == start_time]) >= 2:
                    continue

    # Deduplicate exact (start, end) pairs.
    seen = set()
    deduped = []
    for c in raw_candidates:
        key = (round(c.start, 1), round(c.end, 1))
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    if len(deduped) > max_candidates:
        # Prioritize longer, more "complete" candidates spread across the video.
        deduped.sort(key=lambda c: c.start)
        step = len(deduped) / max_candidates
        sampled = [deduped[int(i * step)] for i in range(max_candidates)]
        deduped = sampled

    logger.info("Generated %d candidate clips from %d sentences", len(deduped), len(sentences))
    return deduped
