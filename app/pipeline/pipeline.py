"""Orchestrates the full VIDEO -> SHORTS pipeline.

Reports progress through an optional `progress_callback(event: dict)` so
both the CLI and the Streamlit UI can show live status. Every event has a
"type" key: "step" (major stage change), "substep" (per-candidate /
per-clip progress within a stage), or "log" (a printable note). If no
callback is given, a console printer reproduces the original CLI output.
"""
from __future__ import annotations

import logging
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from app import config
from app.analysis.candidate_generator import Candidate, generate_candidates
from app.analysis.llm_scorer import ScoredClip, score_all_candidates
from app.transcription.whisper import Transcript, Word, transcribe_video, transcript_cache_path
from app.utils import stats as timing_stats
from app.utils.ffmpeg import burn_subtitles, probe_duration
from app.utils.files import file_fingerprint, safe_stem, write_json
from app.utils.validation import ValidationError, check_disk_space, estimate_required_space, validate_input_file
from app.video.captions import generate_ass
from app.video.cutter import refine_bounds, render_clip
from app.video.effects import apply_optional_effects

logger = logging.getLogger("shorts.pipeline")

TOTAL_STEPS = 7
ProgressCallback = Callable[[dict], None]


def _console_progress(event: dict) -> None:
    if event["type"] == "step":
        print(f"[{event['step']}/{TOTAL_STEPS}] {event['message']}")
    elif event["type"] in ("log", "clip_ready"):
        print(event["message"])


def _emit(cb: ProgressCallback | None, **event) -> None:
    _console_progress(event)
    logger.info("%s: %s", event.get("type"), event.get("message", ""))
    if cb:
        cb(event)


def _iou(a: Candidate | ScoredClip, b: Candidate | ScoredClip) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union > 0 else 0.0


def _greedy_non_overlapping(sorted_clips: list[ScoredClip], threshold: float) -> list[ScoredClip]:
    selected: list[ScoredClip] = []
    for clip in sorted_clips:
        if all(_iou(clip, s) < threshold for s in selected):
            selected.append(clip)
    return selected


def _diversity_round_robin(pool: list[ScoredClip], n: int) -> list[ScoredClip]:
    """Round-robins across categories (highest-scoring category first each
    round) so we don't return N near-identical clips of the same type."""
    by_cat: dict[str, list[ScoredClip]] = defaultdict(list)
    for clip in pool:
        by_cat[clip.category].append(clip)

    cats = sorted(by_cat.keys(), key=lambda c: by_cat[c][0].final_score, reverse=True)
    idx = {c: 0 for c in cats}
    selected: list[ScoredClip] = []

    while len(selected) < n:
        progressed = False
        for cat in cats:
            if idx[cat] < len(by_cat[cat]):
                selected.append(by_cat[cat][idx[cat]])
                idx[cat] += 1
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break
    return selected


def select_final_clips(
    scored_clips: list[ScoredClip], cfg: config.Config
) -> tuple[list[ScoredClip], str | None]:
    eligible = [c for c in scored_clips if c.final_score >= cfg.min_score]
    eligible.sort(key=lambda c: c.final_score, reverse=True)

    non_overlapping = _greedy_non_overlapping(eligible, cfg.overlap_iou_threshold)

    if cfg.prefer_category_diversity:
        selected = _diversity_round_robin(non_overlapping, cfg.num_clips)
    else:
        selected = non_overlapping[: cfg.num_clips]

    note = None
    if len(selected) < cfg.num_clips:
        note = (
            f"Requested {cfg.num_clips} clips, but only {len(selected)} candidates met the "
            f"minimum score of {cfg.min_score} after removing overlaps "
            f"({len(scored_clips)} scored, {len(eligible)} above threshold, "
            f"{len(non_overlapping)} non-overlapping). Quality was not lowered to hit the count."
        )
    return selected, note


def _words_in_range(transcript: Transcript, start: float, end: float) -> list[Word]:
    words = []
    for seg in transcript.segments:
        for w in seg.words:
            if w.start >= start - 0.05 and w.end <= end + 0.05:
                words.append(w)
    return words


def estimate_run_time_from_duration(duration: float, cfg: config.Config, whisper_cached: bool = False) -> dict:
    """Pre-run ETA from a known duration — used both for local files (probed)
    and for a not-yet-downloaded YouTube video (duration from its metadata)."""
    approx_candidates = min(cfg.max_candidates, max(1, int(duration / cfg.min_clip_seconds))) if duration else 1
    avg_clip_duration = (cfg.min_clip_seconds + cfg.max_clip_seconds) / 2
    est = timing_stats.estimate_total_seconds(
        video_duration_seconds=duration,
        num_candidates=approx_candidates,
        num_clips=cfg.num_clips,
        avg_clip_duration_seconds=avg_clip_duration,
        whisper_cached=whisper_cached,
    )
    est["video_duration_seconds"] = round(duration, 1)
    est["approx_candidates"] = approx_candidates
    return est


def estimate_run_time(input_path: Path, cfg: config.Config) -> dict:
    """Pre-run ETA for a local file already on disk: probes real duration
    and checks whether it's already transcribed."""
    try:
        duration = probe_duration(input_path)
    except Exception:
        duration = 0.0
    whisper_cached = transcript_cache_path(input_path, cfg.whisper_model).exists()
    return estimate_run_time_from_duration(duration, cfg, whisper_cached=whisper_cached)


def run_pipeline(
    input_path: Path,
    num_clips: int | None = None,
    min_score: float | None = None,
    force_retranscribe: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Returns a summary dict: {"clips": [...], "output_dir": ..., "error": str|None}."""
    cfg = config.load_config()
    if num_clips is not None:
        cfg.num_clips = num_clips
    if min_score is not None:
        cfg.min_score = min_score
    cfg.validate()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f"run_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    logger.info("=== AI Shorts Generator run %s ===", run_id)
    logger.info("Input: %s | clips=%d | min_score=%.1f", input_path, cfg.num_clips, cfg.min_score)

    try:
        validate_input_file(input_path)
        check_disk_space(config.TEMP_DIR, estimate_required_space(input_path))
    except ValidationError as e:
        _emit(progress_callback, type="log", message=f"ERROR: {e}")
        logger.error(str(e))
        return {"clips": [], "output_dir": str(config.OUTPUT_DIR), "error": str(e)}

    video_stem = safe_stem(input_path)
    work_dir = config.TEMP_DIR / f"{video_stem}_{run_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    whisper_seconds = 0.0
    llm_timings: list[float] = []
    render_timings: list[float] = []

    try:
        _emit(progress_callback, type="step", step=1, total_steps=TOTAL_STEPS,
              message="Extracting audio + transcribing (cached if unchanged)...")
        whisper_cache_hit = transcript_cache_path(input_path, cfg.whisper_model).exists() and not force_retranscribe
        t0 = time.time()
        transcript = transcribe_video(
            input_path,
            model_name=cfg.whisper_model,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
            force=force_retranscribe,
        )
        whisper_seconds = time.time() - t0 if not whisper_cache_hit else 0.0

        _emit(progress_callback, type="step", step=2, total_steps=TOTAL_STEPS,
              message="Generating candidate clips from transcript...")
        candidates = generate_candidates(
            transcript,
            min_seconds=cfg.min_clip_seconds,
            max_seconds=cfg.max_clip_seconds,
            silence_gap_threshold=cfg.silence_gap_threshold,
            max_silence_ratio=cfg.max_silence_ratio,
            max_candidates=cfg.max_candidates,
        )
        if not candidates:
            msg = "No viable candidate clips could be generated from this video."
            _emit(progress_callback, type="log", message=msg)
            return {"clips": [], "output_dir": str(config.OUTPUT_DIR), "error": msg}

        _emit(progress_callback, type="step", step=3, total_steps=TOTAL_STEPS,
              message=f"Scoring {len(candidates)} candidates with {cfg.ollama_model}...")
        cache_namespace = f"{video_stem}_{file_fingerprint(input_path)}"

        def _on_score_progress(i, total, elapsed, was_cached):
            _emit(progress_callback, type="substep", stage="scoring", index=i, total=total,
                  elapsed=elapsed, cached=was_cached,
                  message=f"Scored candidate {i}/{total}")

        scored, llm_timings = score_all_candidates(
            candidates, cfg, cache_namespace, on_progress=_on_score_progress,
        )
        if not scored:
            msg = "No candidates could be scored (Ollama unreachable or all requests failed). Check logs."
            _emit(progress_callback, type="log", message=msg)
            return {"clips": [], "output_dir": str(config.OUTPUT_DIR), "error": msg}

        _emit(progress_callback, type="step", step=4, total_steps=TOTAL_STEPS,
              message="Removing overlaps/duplicates and selecting final clips...")
        selected, note = select_final_clips(scored, cfg)
        if note:
            _emit(progress_callback, type="log", message=note)
        if not selected:
            msg = "No clips met the minimum score threshold. Try lowering --min-score."
            _emit(progress_callback, type="log", message=msg)
            return {"clips": [], "output_dir": str(config.OUTPUT_DIR), "error": msg}

        _emit(progress_callback, type="step", step=5, total_steps=TOTAL_STEPS,
              message=f"Refining timestamps for {len(selected)} selected clip(s)...")
        for clip in selected:
            words = _words_in_range(transcript, clip.start, clip.end)
            clip_start, clip_end = refine_bounds(clip.start, clip.end, words)
            clip.start, clip.end = clip_start, clip_end

        _emit(progress_callback, type="step", step=6, total_steps=TOTAL_STEPS,
              message=f"Rendering {len(selected)} Shorts (cut, reframe, captions)...")
        metadata = []
        for i, clip in enumerate(selected, 1):
            clip_id = f"short_{i:02d}"
            words = _words_in_range(transcript, clip.start, clip.end)
            clip_duration = max(0.1, clip.end - clip.start)
            render_t0 = time.time()

            raw_out = work_dir / f"{clip_id}_raw.mp4"
            render_clip(
                source_video=input_path,
                start=clip.start,
                end=clip.end,
                words=words,
                out_path=raw_out,
                output_width=cfg.output_width,
                output_height=cfg.output_height,
                crf=cfg.video_crf,
                preset=cfg.video_preset,
            )

            ass_path = config.CAPTIONS_DIR / f"{video_stem}_{clip_id}.ass"
            generate_ass(
                clip_words=words,
                clip_start_offset=clip.start,
                style=cfg.caption_style,
                out_path=ass_path,
                output_width=cfg.output_width,
                output_height=cfg.output_height,
            )

            captioned = work_dir / f"{clip_id}_captioned.mp4"
            burn_subtitles(raw_out, ass_path, captioned, cfg.video_crf, cfg.video_preset)

            final_video = apply_optional_effects(captioned, work_dir, cfg)

            final_out = config.OUTPUT_DIR / f"{clip_id}.mp4"
            shutil.copy2(final_video, final_out)

            render_timings.append((time.time() - render_t0) / clip_duration)

            clip_meta = {
                "clip": clip_id,
                "source": input_path.name,
                "start": round(clip.start, 2),
                "end": round(clip.end, 2),
                "duration": round(clip.end - clip.start, 2),
                "score": clip.final_score,
                "category": clip.category,
                "hook": clip.hook,
                "reason": clip.reason,
                "score_breakdown": clip.scores,
                "caption_file": str(ass_path),
                "output_file": str(final_out),
            }
            metadata.append(clip_meta)

            # Written after every clip (not just at the end) so a viewer —
            # the Streamlit UI, or metadata.json itself — can pick up each
            # finished Short as soon as it's ready, instead of waiting for
            # the whole batch to render.
            write_json(config.OUTPUT_DIR / "metadata.json", {
                "source": str(input_path),
                "run_id": run_id,
                "requested_clips": cfg.num_clips,
                "generated_clips": len(metadata),
                "clips": metadata,
            })

            _emit(progress_callback, type="clip_ready", index=i, total=len(selected), clip=clip_meta,
                  message=f"  -> {final_out.name} [{clip.category}] score={clip.final_score} \"{clip.hook}\"")

        timing_stats.record_run(
            whisper_seconds=whisper_seconds,
            whisper_video_seconds=transcript.duration if whisper_seconds else None,
            llm_seconds_per_candidate_samples=llm_timings,
            render_seconds_per_output_second_samples=render_timings,
        )

        _emit(progress_callback, type="step", step=7, total_steps=TOTAL_STEPS, message="Complete.")
        _emit(progress_callback, type="log", message=f"{len(metadata)} Short(s) written to {config.OUTPUT_DIR}")

        return {"clips": metadata, "output_dir": str(config.OUTPUT_DIR), "error": None}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
