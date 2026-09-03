"""Local transcription via faster-whisper (large-v3-turbo), with word-level
timestamps and on-disk caching so the same video is never re-transcribed
unless explicitly forced."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from app import config
from app.utils.ffmpeg import extract_audio
from app.utils.files import file_fingerprint, read_json, write_json

logger = logging.getLogger("shorts.whisper")


def _register_cuda_dll_dirs() -> None:
    """On Windows, CTranslate2 does not add pip-installed CUDA libraries
    (nvidia-cublas-cu12, nvidia-cudnn-cu12) to the DLL search path, so
    `cublas64_12.dll` / cuDNN fail to load even when the wheels are present.
    Register their `bin` folders explicitly before faster_whisper imports."""
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for pkg_root in map(Path, nvidia.__path__):
        for bin_dir in pkg_root.glob("*/bin"):
            if bin_dir.is_dir():
                try:
                    os.add_dll_directory(str(bin_dir))
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                except OSError:
                    pass


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]


@dataclass
class Transcript:
    source: str
    duration: float
    language: str
    model: str
    segments: list[Segment]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "duration": self.duration,
            "language": self.language,
            "model": self.model,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [asdict(w) for w in s.words],
                }
                for s in self.segments
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> "Transcript":
        segments = [
            Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                words=[Word(**w) for w in s.get("words", [])],
            )
            for s in data["segments"]
        ]
        return Transcript(
            source=data["source"],
            duration=data["duration"],
            language=data.get("language", "en"),
            model=data.get("model", "unknown"),
            segments=segments,
        )


def _resolve_device(requested: str) -> tuple[str, str]:
    """Returns (device, compute_type), falling back gracefully if CUDA unavailable."""
    if requested != "auto":
        return requested, config.Config().whisper_compute_type
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    logger.warning("CUDA not available to ctranslate2 — falling back to CPU (much slower).")
    return "cpu", "int8"


def transcript_cache_path(video_path: Path, model_name: str) -> Path:
    fp = file_fingerprint(video_path)
    from app.utils.files import safe_stem
    return config.TRANSCRIPTS_DIR / f"{safe_stem(video_path)}_{model_name}_{fp}.json"


def transcribe_video(
    video_path: Path,
    model_name: str = "large-v3-turbo",
    device: str = "auto",
    compute_type: str = "float16",
    force: bool = False,
) -> Transcript:
    """Transcribe a video, using a cached result keyed on (filename, size,
    mtime, model) when available."""
    cache_path = transcript_cache_path(video_path, model_name)
    if not force:
        cached = read_json(cache_path)
        if cached is not None:
            logger.info("Using cached transcript: %s", cache_path.name)
            return Transcript.from_dict(cached)

    _register_cuda_dll_dirs()
    from faster_whisper import WhisperModel

    resolved_device, resolved_compute = _resolve_device(device)
    if device != "auto":
        resolved_compute = compute_type

    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s (cache: %s)",
        model_name, resolved_device, resolved_compute, config.FASTER_WHISPER_CACHE_DIR,
    )

    audio_path = config.TEMP_DIR / f"{video_path.stem}_audio.wav"
    extract_audio(video_path, audio_path)

    def _load_model(dev: str, comp: str) -> "WhisperModel":
        return WhisperModel(
            model_name,
            device=dev,
            compute_type=comp,
            download_root=str(config.FASTER_WHISPER_CACHE_DIR),
        )

    try:
        try:
            model = _load_model(resolved_device, resolved_compute)
        except RuntimeError as exc:
            if resolved_device != "cuda":
                raise
            logger.warning(
                "CUDA transcription unavailable (%s) — falling back to CPU (much slower). "
                "Install nvidia-cublas-cu12 and nvidia-cudnn-cu12 to use the GPU.",
                exc,
            )
            resolved_device, resolved_compute = "cpu", "int8"
            model = _load_model(resolved_device, resolved_compute)

        segments_iter, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
        )

        segments: list[Segment] = []
        for seg in segments_iter:
            words = [
                Word(start=w.start, end=w.end, word=w.word, probability=w.probability)
                for w in (seg.words or [])
            ]
            segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words))

        transcript = Transcript(
            source=video_path.name,
            duration=info.duration,
            language=info.language,
            model=model_name,
            segments=segments,
        )
    finally:
        # Free GPU memory before the Ollama scoring step runs.
        try:
            del model
        except UnboundLocalError:
            pass
        import gc
        gc.collect()
        audio_path.unlink(missing_ok=True)

    write_json(cache_path, transcript.to_dict())
    logger.info("Transcribed %d segments, %.1fs audio, cached to %s", len(segments), info.duration, cache_path.name)
    return transcript
