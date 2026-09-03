"""
Central configuration for the AI Shorts Generator.

Everything is derived from PROJECT_ROOT so no project data ever touches C:.
HF_HOME / HUGGINGFACE_HUB_CACHE are set as environment variables at import
time, before any library that downloads models (faster-whisper /
huggingface_hub) gets imported elsewhere in the app.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths — everything else is derived from these. Never hardcode C:\ here.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("F:/AI-Shorts-Generator")

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
CAPTIONS_DIR = PROJECT_ROOT / "captions"
TEMP_DIR = PROJECT_ROOT / "temp"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
TOOLS_DIR = PROJECT_ROOT / "tools"

FASTER_WHISPER_CACHE_DIR = MODELS_DIR / "faster-whisper"
HF_HOME_DIR = MODELS_DIR / "huggingface"
OLLAMA_MODELS_DIR = MODELS_DIR / "ollama"

for _d in (
    INPUT_DIR, OUTPUT_DIR, TRANSCRIPTS_DIR, CAPTIONS_DIR, TEMP_DIR,
    CACHE_DIR, LOGS_DIR, MODELS_DIR, TOOLS_DIR, FASTER_WHISPER_CACHE_DIR,
    HF_HOME_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# Force every HF/torch cache onto F: — must happen before huggingface_hub
# or faster-whisper is imported anywhere in the process.
os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME_DIR / "hub"))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME_DIR / "hub"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

# ---------------------------------------------------------------------------
# FFmpeg — prefer the portable build bundled under tools/ffmpeg, fall back
# to whatever is on PATH.
# ---------------------------------------------------------------------------
_bundled_ffmpeg = TOOLS_DIR / "ffmpeg" / "ffmpeg.exe"
_bundled_ffprobe = TOOLS_DIR / "ffmpeg" / "ffprobe.exe"
FFMPEG_BIN = str(_bundled_ffmpeg) if _bundled_ffmpeg.exists() else "ffmpeg"
FFPROBE_BIN = str(_bundled_ffprobe) if _bundled_ffprobe.exists() else "ffprobe"

SUPPORTED_INPUT_EXTENSIONS = {".mp4", ".mkv", ".mov"}

# ---------------------------------------------------------------------------
# Score weights — must sum to 1.0. Validated in load_config().
# ---------------------------------------------------------------------------
DEFAULT_SCORE_WEIGHTS = {
    "hook": 0.20,
    "curiosity": 0.15,
    "emotion": 0.10,
    "payoff": 0.15,
    "standalone": 0.10,
    "shareability": 0.10,
    "clarity": 0.05,
    "retention": 0.15,
}

CATEGORIES = [
    "FUNNY", "SHOCKING", "STORY", "EDUCATIONAL", "EMOTIONAL",
    "CONTROVERSIAL", "INSPIRATIONAL", "UNEXPECTED", "OTHER",
]

CONFIG_FILE = PROJECT_ROOT / "config.json"


@dataclass
class CaptionStyle:
    font_name: str = "Arial Black"
    font_size: int = 84
    primary_color: str = "&H00FFFFFF"      # white
    highlight_color: str = "&H0000D7FF"    # orange/gold (BGR in ASS)
    outline_color: str = "&H00000000"      # black
    outline_width: int = 4
    shadow: int = 2
    position: str = "bottom"               # bottom | center | top
    max_words_per_caption: int = 4


@dataclass
class Config:
    # models
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "auto"           # auto | cuda | cpu
    whisper_compute_type: str = "float16"  # float16 | int8_float16 | int8
    ollama_model: str = "qwen3:8b"
    ollama_host: str = "http://localhost:11434"

    # candidate generation
    min_clip_seconds: float = 15.0
    max_clip_seconds: float = 60.0
    silence_gap_threshold: float = 0.6     # seconds treated as a natural break
    max_silence_ratio: float = 0.35        # skip candidates mostly silence
    max_candidates: int = 40               # cap sent to the LLM per video (cost control)

    # scoring / selection
    num_clips: int = 10
    min_score: float = 6.5
    score_weights: dict = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    overlap_iou_threshold: float = 0.5     # candidates overlapping more than this are duplicates
    prefer_category_diversity: bool = True
    llm_max_retries: int = 3
    llm_request_timeout: int = 120

    # output video
    output_width: int = 1080
    output_height: int = 1920
    video_crf: int = 18
    video_preset: str = "medium"

    # captions
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)

    # optional effects — OFF by default (spec: default = video + audio + captions)
    enable_zoom: bool = False
    enable_silence_removal: bool = False
    enable_music: bool = False
    enable_sfx: bool = False
    music_dir: str = str(PROJECT_ROOT / "assets" / "music")
    sfx_dir: str = str(PROJECT_ROOT / "assets" / "sfx")

    # paths (overridable)
    temp_dir: str = str(TEMP_DIR)
    cache_dir: str = str(CACHE_DIR)
    output_dir: str = str(OUTPUT_DIR)

    def validate(self) -> None:
        total = sum(self.score_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"score_weights must sum to 1.0, got {total}")
        if self.min_clip_seconds <= 0 or self.max_clip_seconds <= self.min_clip_seconds:
            raise ValueError("min_clip_seconds must be > 0 and < max_clip_seconds")
        if not (0.0 <= self.min_score <= 10.0):
            raise ValueError("min_score must be between 0 and 10")


def load_config() -> Config:
    """Load config.json if present, overlaying it on top of defaults."""
    cfg = Config()
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {CONFIG_FILE}: {e}") from e
        caption_data = data.pop("caption_style", None)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        if caption_data:
            cfg.caption_style = CaptionStyle(**caption_data)
    cfg.validate()
    return cfg


def save_default_config() -> None:
    """Write out the current defaults to config.json (does not overwrite)."""
    if CONFIG_FILE.exists():
        return
    cfg = Config()
    data = asdict(cfg)
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
