"""Thin, safe wrappers around the FFmpeg/FFprobe binaries.

All commands are built as argument lists and run with subprocess (no
shell=True), so filenames are never interpolated into a shell string.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from app.config import FFMPEG_BIN, FFPROBE_BIN

logger = logging.getLogger("shorts.ffmpeg")


class FFmpegError(Exception):
    pass


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise FFmpegError(
            f"Could not find executable '{cmd[0]}'. Is FFmpeg installed at "
            f"tools/ffmpeg or available on PATH?"
        ) from e
    if result.returncode != 0:
        raise FFmpegError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr[-4000:]}"
        )
    return result


def probe_duration(video_path: Path) -> float:
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    result = _run(cmd)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def probe_resolution(video_path: Path) -> tuple[int, int]:
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ]
    result = _run(cmd)
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def extract_audio(video_path: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    """Extract mono 16kHz WAV audio, the format faster-whisper expects."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    _run(cmd, timeout=1800)
    return out_wav


def sample_frames(video_path: Path, start: float, end: float, count: int, out_dir: Path) -> list[Path]:
    """Grab `count` evenly spaced JPEG frames between start/end for face detection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = max(end - start, 0.1)
    paths = []
    for i in range(count):
        t = start + duration * (i + 0.5) / count
        out_path = out_dir / f"frame_{i:02d}.jpg"
        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{t:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ]
        try:
            _run(cmd, timeout=60)
            paths.append(out_path)
        except FFmpegError:
            logger.warning("Failed to sample frame at %.2fs", t)
    return paths


def cut_reframe_and_encode(
    video_path: Path,
    start: float,
    end: float,
    out_path: Path,
    crop_filter: str,
    output_width: int,
    output_height: int,
    crf: int,
    preset: str,
    extra_video_filters: list[str] | None = None,
) -> Path:
    """Cut [start, end), apply crop_filter (e.g. 'crop=608:1080:120:0'), scale to
    output resolution, and encode with original audio preserved."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(end - start, 0.1)

    filters = [crop_filter, f"scale={output_width}:{output_height}"]
    if extra_video_filters:
        filters.extend(extra_video_filters)
    vf = ",".join(filters)

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run(cmd, timeout=1800)
    return out_path


def burn_subtitles(
    video_in: Path,
    ass_path: Path,
    video_out: Path,
    crf: int,
    preset: str,
) -> Path:
    """Burn an ASS subtitle file into the video (re-encode)."""
    video_out.parent.mkdir(parents=True, exist_ok=True)
    # FFmpeg's subtitles filter needs escaped colons/backslashes on Windows paths.
    escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_in),
        "-vf", f"ass='{escaped}'",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(video_out),
    ]
    _run(cmd, timeout=1800)
    return video_out


def detect_silence(
    video_path: Path,
    noise_db: float = -35.0,
    min_duration: float = 0.4,
) -> list[tuple[float, float]]:
    """Returns a list of (start, end) silent ranges using ffmpeg's silencedetect."""
    cmd = [
        FFMPEG_BIN, "-i", str(video_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    ranges: list[tuple[float, float]] = []
    start = None
    for line in result.stderr.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                start = float(line.split("silence_start:")[1].strip())
            except (ValueError, IndexError):
                start = None
        elif "silence_end:" in line and start is not None:
            try:
                end_part = line.split("silence_end:")[1].strip()
                end = float(end_part.split(" ")[0])
                ranges.append((start, end))
            except (ValueError, IndexError):
                pass
            start = None
    return ranges


def remove_silence_segments(
    video_in: Path,
    video_out: Path,
    silent_ranges: list[tuple[float, float]],
    total_duration: float,
    crf: int,
    preset: str,
) -> Path:
    """Cuts out the given silent ranges from both video and audio (kept in
    sync since both streams are trimmed from the same segment boundaries),
    and concatenates what remains."""
    video_out.parent.mkdir(parents=True, exist_ok=True)

    keep_ranges = []
    cursor = 0.0
    for s, e in sorted(silent_ranges):
        if s > cursor:
            keep_ranges.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_duration:
        keep_ranges.append((cursor, total_duration))
    keep_ranges = [(s, e) for s, e in keep_ranges if e - s > 0.05]

    if not keep_ranges:
        video_out.write_bytes(video_in.read_bytes())
        return video_out

    filter_parts = []
    v_labels, a_labels = [], []
    for i, (s, e) in enumerate(keep_ranges):
        filter_parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
        v_labels.append(f"[v{i}]")
        a_labels.append(f"[a{i}]")

    concat_inputs = "".join(f"{v}{a}" for v, a in zip(v_labels, a_labels))
    filter_parts.append(f"{concat_inputs}concat=n={len(keep_ranges)}:v=1:a=1[vout][aout]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_in),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(video_out),
    ]
    _run(cmd, timeout=1800)
    return video_out


def apply_zoom(
    video_in: Path,
    video_out: Path,
    output_width: int,
    output_height: int,
    crf: int,
    preset: str,
    zoom_to: float = 1.08,
) -> Path:
    """Slow, steady zoom-in over the clip's duration."""
    video_out.parent.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video_in)
    fps = 30
    total_frames = max(1, int(duration * fps))
    zoom_step = (zoom_to - 1.0) / total_frames
    zoompan = (
        f"zoompan=z='min(zoom+{zoom_step:.6f},{zoom_to})':d=1:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s={output_width}x{output_height}:fps={fps}"
    )
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_in),
        "-vf", zoompan,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(video_out),
    ]
    _run(cmd, timeout=1800)
    return video_out


def overlay_sfx(
    video_in: Path,
    sfx_path: Path,
    video_out: Path,
    at_time: float = 0.0,
    volume: float = 0.6,
) -> Path:
    """Overlays a one-shot sound effect at `at_time` seconds into the clip."""
    video_out.parent.mkdir(parents=True, exist_ok=True)
    delay_ms = int(at_time * 1000)
    filter_complex = (
        f"[1:a]adelay={delay_ms}|{delay_ms},volume={volume}[sfx];"
        f"[0:a][sfx]amix=inputs=2:duration=first:dropout_transition=0[aout]"
    )
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_in),
        "-i", str(sfx_path),
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(video_out),
    ]
    _run(cmd, timeout=600)
    return video_out


def mix_audio_track(
    video_in: Path,
    music_path: Path,
    video_out: Path,
    music_volume: float = 0.15,
) -> Path:
    """Mix a background music file under the existing audio track."""
    video_out.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (
        f"[1:a]volume={music_volume}[music];"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_in),
        "-stream_loop", "-1",
        "-i", str(music_path),
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(video_out),
    ]
    _run(cmd, timeout=1800)
    return video_out
