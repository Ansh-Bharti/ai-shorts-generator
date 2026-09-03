"""Generates burned-in ASS captions from word-level Whisper timestamps:
short phrases, bottom-safe placement, and a karaoke-style highlight on
the word currently being spoken.
"""
from __future__ import annotations

from pathlib import Path

from app.config import CaptionStyle
from app.transcription.whisper import Word

_ALIGNMENT = {"bottom": 2, "center": 5, "top": 8}


def _format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    centis = int(round((s - int(s)) * 100))
    if centis == 100:
        centis = 0
        s += 1
    return f"{h}:{m:02d}:{int(s):02d}.{centis:02d}"


def _escape(text: str) -> str:
    return text.replace("{", "").replace("}", "").replace("\\", "")


def _chunk_words(words: list[Word], max_words: int) -> list[list[Word]]:
    chunks = []
    current: list[Word] = []
    for w in words:
        current.append(w)
        ends_sentence = w.word.strip().endswith((".", "!", "?"))
        if len(current) >= max_words or ends_sentence:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _build_header(style: CaptionStyle, res_x: int, res_y: int) -> str:
    alignment = _ALIGNMENT.get(style.position, 2)
    margin_v = int(res_y * 0.12) if style.position != "center" else 0
    return f"""[Script Info]
Title: AI Shorts Generator Captions
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{style.font_name},{style.font_size},{style.primary_color},{style.highlight_color},{style.outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,{style.outline_width},{style.shadow},{alignment},60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_ass(
    clip_words: list[Word],
    clip_start_offset: float,
    style: CaptionStyle,
    out_path: Path,
    output_width: int = 1080,
    output_height: int = 1920,
) -> Path:
    """clip_words: word timestamps in ORIGINAL video time.
    clip_start_offset: the clip's start time in the original video (subtracted
    to make caption timing relative to the rendered clip)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [_build_header(style, output_width, output_height)]

    chunks = _chunk_words(clip_words, style.max_words_per_caption)
    for chunk in chunks:
        rel_start = chunk[0].start - clip_start_offset
        rel_end = chunk[-1].end - clip_start_offset
        if rel_end <= rel_start:
            continue

        karaoke_text = ""
        for w in chunk:
            duration_cs = max(1, int(round((w.end - w.start) * 100)))
            karaoke_text += f"{{\\k{duration_cs}}}{_escape(w.word.strip())} "

        line = (
            f"Dialogue: 0,{_format_time(rel_start)},{_format_time(rel_end)},"
            f"Caption,,0,0,0,,{karaoke_text.strip()}"
        )
        lines.append(line)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
