"""Fetches YouTube videos locally via yt-dlp so the rest of the pipeline can
treat them exactly like any other file dropped into input/. No cloud AI
service is involved — yt-dlp just downloads the public video stream.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from app import config
from app.utils.files import list_input_videos

YOUTUBE_URL_RE = re.compile(
    r"(https?://)?(www\.)?(m\.)?(youtube\.com/(watch\?v=|shorts/|embed/|live/)|youtu\.be/)[\w-]{6,}",
    re.IGNORECASE,
)

_ID_PREFIX = "yt_"


class YouTubeError(Exception):
    pass


def is_youtube_url(text: str) -> bool:
    return bool(YOUTUBE_URL_RE.search(text.strip()))


def _ffmpeg_dir() -> str | None:
    ffmpeg_dir = Path(config.FFMPEG_BIN).parent
    return str(ffmpeg_dir) if ffmpeg_dir.exists() else None


def _base_opts() -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "restrictfilenames": True,
        "ffmpeg_location": _ffmpeg_dir(),
    }


def fetch_metadata(url: str) -> dict:
    """Info-only fetch (no video download) — used for the pre-run preview
    and ETA so we know real duration before committing to a download."""
    import yt_dlp

    if not is_youtube_url(url):
        raise YouTubeError(f"Not a recognized YouTube URL: {url}")
    try:
        opts = _base_opts()
        opts["skip_download"] = True
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise YouTubeError(f"Could not fetch video info: {e}") from e

    return {
        "id": info.get("id"),
        "title": info.get("title") or "video",
        "duration": info.get("duration") or 0,
        "thumbnail": info.get("thumbnail"),
        "uploader": info.get("uploader"),
    }


def _find_existing(video_id: str) -> Path | None:
    prefix = f"{_ID_PREFIX}{video_id}_"
    for existing in list_input_videos(config.INPUT_DIR, config.SUPPORTED_INPUT_EXTENSIONS):
        if existing.stem.startswith(prefix):
            return existing
    return None


def download_video(
    url: str,
    on_progress: Callable[[float, str], None] | None = None,
    max_height: int = 1080,
) -> Path:
    """Downloads (or reuses a cached copy of) a YouTube video into input/,
    merged to mp4 with the project's own bundled ffmpeg. Returns the local
    path so callers can feed it straight into the normal pipeline."""
    import yt_dlp

    meta = fetch_metadata(url)
    video_id = meta["id"]
    if not video_id:
        raise YouTubeError("Could not determine a video id for this URL.")

    cached = _find_existing(video_id)
    if cached is not None:
        if on_progress:
            on_progress(1.0, f"Already downloaded: {cached.name}")
        return cached

    def _hook(d: dict) -> None:
        if not on_progress:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            frac = (downloaded / total) if total else 0.0
            on_progress(frac, f"Downloading: {frac * 100:.0f}%")
        elif d.get("status") == "finished":
            on_progress(1.0, "Download finished, merging with ffmpeg...")

    out_template = str(config.INPUT_DIR / f"{_ID_PREFIX}{video_id}_%(title).60s.%(ext)s")
    opts = _base_opts()
    opts.update({
        "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}]",
        "merge_output_format": "mp4",
        "outtmpl": out_template,
        "progress_hooks": [_hook],
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
    })

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise YouTubeError(f"Download failed: {e}") from e

    result = _find_existing(video_id)
    if result is None:
        raise YouTubeError("Download reported success but the output file was not found.")
    return result
