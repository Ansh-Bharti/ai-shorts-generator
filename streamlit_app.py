"""Local Streamlit front-end for the AI Shorts Generator.

Run with:
    F:\\AI-Shorts-Generator\\venv\\Scripts\\streamlit run streamlit_app.py

Everything here calls the same pipeline the CLI uses (app.pipeline.pipeline) —
this is a UI on top of it, not a separate implementation. Runs fully local,
no external services besides the local Ollama server.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests
import streamlit as st

from app import config
from app.pipeline.pipeline import estimate_run_time_from_duration, run_pipeline
from app.utils.ffmpeg import FFmpegError, probe_duration
from app.utils.files import list_input_videos, read_json
from app.utils.youtube import YouTubeError, download_video, fetch_metadata, is_youtube_url
from app.transcription.whisper import transcript_cache_path

st.set_page_config(page_title="AI Shorts Generator", layout="wide", page_icon="🎬")

CATEGORY_COLORS = {
    "FUNNY": "#f5a623", "SHOCKING": "#d0021b", "STORY": "#4a90d9",
    "EDUCATIONAL": "#7ed321", "EMOTIONAL": "#bd10e0", "CONTROVERSIAL": "#e35b5b",
    "INSPIRATIONAL": "#50e3c2", "UNEXPECTED": "#9013fe", "OTHER": "#9b9b9b",
}


# --------------------------------------------------------------------------- #
# System status
# --------------------------------------------------------------------------- #
def check_ollama(host: str) -> tuple[bool, list[str]]:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, models
    except requests.RequestException:
        return False, []


def check_gpu() -> tuple[bool, str]:
    try:
        import ctranslate2
        count = ctranslate2.get_cuda_device_count()
        return count > 0, f"{count} CUDA device(s) visible to ctranslate2"
    except Exception as e:
        return False, str(e)


# --------------------------------------------------------------------------- #
# Sidebar — inputs and settings
# --------------------------------------------------------------------------- #
st.sidebar.title("🎬 AI Shorts Generator")
st.sidebar.caption("Fully local — no cloud AI APIs.")

with st.sidebar.expander("System status", expanded=False):
    cfg_preview = config.load_config()
    ollama_ok, ollama_models = check_ollama(cfg_preview.ollama_host)
    gpu_ok, gpu_msg = check_gpu()
    st.write("Ollama:", "🟢 running" if ollama_ok else "🔴 not reachable")
    st.write("GPU:", "🟢 " + gpu_msg if gpu_ok else "🟡 " + gpu_msg)
    ffmpeg_path = Path(config.FFMPEG_BIN)
    st.write("FFmpeg:", "🟢 bundled" if ffmpeg_path.exists() else "🟡 using system PATH")

st.sidebar.subheader("1. Video")
source_mode = st.sidebar.radio("Source", ["Local file", "YouTube URL"], horizontal=True)

selected_name = None
youtube_url = ""
youtube_meta = None
youtube_error = None

if source_mode == "Local file":
    existing_videos = list_input_videos(config.INPUT_DIR, config.SUPPORTED_INPUT_EXTENSIONS)
    video_names = [p.name for p in existing_videos]

    uploaded = st.sidebar.file_uploader("Upload a video into input/", type=["mp4", "mkv", "mov"])
    if uploaded is not None:
        dest = config.INPUT_DIR / uploaded.name
        if not dest.exists():
            dest.write_bytes(uploaded.getvalue())
            st.sidebar.success(f"Saved to input/{uploaded.name}")
            video_names = [p.name for p in list_input_videos(config.INPUT_DIR, config.SUPPORTED_INPUT_EXTENSIONS)]

    if not video_names:
        st.sidebar.info("No videos in input/ yet. Upload one above.")
    else:
        selected_name = st.sidebar.selectbox("Choose video from input/", video_names)

else:
    youtube_url = st.sidebar.text_input("Paste a YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    st.sidebar.caption("Only use videos you have the rights to repurpose.")
    if youtube_url.strip():
        if not is_youtube_url(youtube_url):
            youtube_error = "Doesn't look like a YouTube URL."
        else:
            cache_key = youtube_url.strip()
            if st.session_state.get("_yt_meta_url") == cache_key:
                youtube_meta = st.session_state.get("_yt_meta")
            else:
                with st.sidebar.status("Fetching video info...", expanded=False):
                    try:
                        youtube_meta = fetch_metadata(cache_key)
                        st.session_state["_yt_meta_url"] = cache_key
                        st.session_state["_yt_meta"] = youtube_meta
                    except YouTubeError as e:
                        youtube_error = str(e)
                        st.session_state["_yt_meta_url"] = None
                        st.session_state["_yt_meta"] = None
    if youtube_error:
        st.sidebar.error(youtube_error)

st.sidebar.subheader("2. Generation settings")
num_clips = st.sidebar.number_input("Number of Shorts", min_value=1, max_value=30, value=cfg_preview.num_clips)
min_score = st.sidebar.slider("Minimum score", 0.0, 10.0, cfg_preview.min_score, 0.5)
col_a, col_b = st.sidebar.columns(2)
min_clip_len = col_a.number_input("Min length (s)", min_value=5, max_value=120, value=int(cfg_preview.min_clip_seconds))
max_clip_len = col_b.number_input("Max length (s)", min_value=10, max_value=180, value=int(cfg_preview.max_clip_seconds))

model_options = ollama_models if ollama_models else [cfg_preview.ollama_model]
default_idx = model_options.index(cfg_preview.ollama_model) if cfg_preview.ollama_model in model_options else 0
ollama_model = st.sidebar.selectbox("Ollama model", model_options, index=default_idx)

caption_position = st.sidebar.selectbox("Caption position", ["bottom", "center", "top"], index=0)

st.sidebar.subheader("3. Optional effects (off by default)")
enable_zoom = st.sidebar.checkbox("Slow zoom")
enable_silence_removal = st.sidebar.checkbox("Silence removal")
enable_music = st.sidebar.checkbox("Background music (needs files in assets/music)")
enable_sfx = st.sidebar.checkbox("Sound effects (needs files in assets/sfx)")
force_retranscribe = st.sidebar.checkbox("Force re-transcription (ignore cache)")


def build_config() -> config.Config:
    cfg = config.load_config()
    cfg.num_clips = int(num_clips)
    cfg.min_score = float(min_score)
    cfg.min_clip_seconds = float(min_clip_len)
    cfg.max_clip_seconds = float(max_clip_len)
    cfg.ollama_model = ollama_model
    cfg.caption_style.position = caption_position
    cfg.enable_zoom = enable_zoom
    cfg.enable_silence_removal = enable_silence_removal
    cfg.enable_music = enable_music
    cfg.enable_sfx = enable_sfx
    cfg.validate()
    return cfg


def render_clip_card(clip: dict) -> None:
    with st.container(border=True):
        cols = st.columns([1, 1, 1.2])
        out_file = Path(clip["output_file"])
        with cols[0]:
            if out_file.exists():
                st.video(str(out_file))
                with open(out_file, "rb") as f:
                    st.download_button(
                        "⬇ Download", f.read(), file_name=out_file.name,
                        mime="video/mp4", key=f"dl_{clip['clip']}_{clip['start']}",
                    )
            else:
                st.warning("Output file missing (was it moved or deleted?).")

        with cols[1]:
            st.metric("Virality score", f"{clip['score']:.1f} / 10")
            color = CATEGORY_COLORS.get(clip["category"], "#9b9b9b")
            st.markdown(
                f"<span style='background:{color};color:white;padding:2px 10px;"
                f"border-radius:12px;font-size:0.85em'>{clip['category']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"{clip['duration']:.1f}s clip · {clip['start']:.1f}s–{clip['end']:.1f}s in source")
            st.markdown(f"**Hook:** {clip['hook']}")
            st.markdown(f"*{clip['reason']}*")

        with cols[2]:
            st.caption("Score breakdown")
            st.bar_chart(clip["score_breakdown"], horizontal=True)


def fmt_seconds(s: float) -> str:
    s = max(0, int(round(s)))
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.title("AI Shorts Generator")

generated_this_run = False
has_local = selected_name is not None
has_youtube = youtube_meta is not None

if not has_local and not has_youtube:
    st.info("Add a video via the sidebar to get started — upload a file or paste a YouTube URL.")
else:
    cfg = build_config()
    input_path: Path | None = None
    duration = 0.0
    whisper_cached = False

    left, right = st.columns([2, 1])

    if has_local:
        input_path = config.INPUT_DIR / selected_name
        with left:
            st.video(str(input_path))
        try:
            duration = probe_duration(input_path)
        except FFmpegError as e:
            st.error(f"Could not read video: {e}")
        whisper_cached = transcript_cache_path(input_path, cfg.whisper_model).exists()
    else:
        duration = youtube_meta["duration"]
        with left:
            if youtube_meta.get("thumbnail"):
                st.image(youtube_meta["thumbnail"], width="stretch")
            st.markdown(f"**{youtube_meta['title']}**")
            st.caption(f"by {youtube_meta.get('uploader') or 'unknown'} — will be downloaded into input/ when you click Generate")

    with right:
        st.metric("Video duration", fmt_seconds(duration))
        estimate = estimate_run_time_from_duration(duration, cfg, whisper_cached=whisper_cached)
        calibrated_note = "calibrated from your past runs" if estimate["calibrated"] else "rough default — improves after your first run"
        time_label = "Estimated processing time" if has_local else "Estimated processing time (excludes download)"
        st.metric(time_label, fmt_seconds(estimate["total_seconds"]), help=calibrated_note)
        with st.expander("Time estimate breakdown"):
            st.write(f"Transcription: {fmt_seconds(estimate['whisper_seconds'])}")
            st.write(f"LLM scoring (~{estimate['approx_candidates']} candidates): {fmt_seconds(estimate['scoring_seconds'])}")
            st.write(f"Rendering {cfg.num_clips} clip(s): {fmt_seconds(estimate['rendering_seconds'])}")
            st.caption(calibrated_note)

    generate_disabled = not ollama_ok
    if not ollama_ok:
        st.error("Ollama isn't reachable at " + cfg.ollama_host + ". Start it before generating.")

    button_label = "🚀 Download & Generate Shorts" if has_youtube else "🚀 Generate Shorts"
    if st.button(button_label, type="primary", disabled=generate_disabled, width="stretch"):
        generated_this_run = True
        st.session_state["last_result"] = []

        if has_youtube and input_path is None:
            dl_status = st.status("Downloading video from YouTube...", expanded=True)
            dl_progress = st.progress(0.0)

            def _dl_progress(frac: float, message: str) -> None:
                dl_progress.progress(min(1.0, max(0.0, frac)))
                dl_status.update(label=message)

            try:
                input_path = download_video(youtube_url.strip(), on_progress=_dl_progress)
            except YouTubeError as e:
                dl_status.update(label="Download failed", state="error")
                st.error(str(e))
                st.stop()
            dl_status.update(label=f"Downloaded: {input_path.name}", state="complete")

        status_box = st.status("Starting pipeline...", expanded=True)
        progress_bar = st.progress(0.0)
        timer_placeholder = st.empty()
        log_lines: list[str] = []
        log_box = st.empty()

        st.divider()
        st.subheader("Results — each Short appears here as soon as it finishes rendering")
        results_container = st.container()

        run_state = {
            "current_step": 0,
            "scoring_index": 0, "scoring_total": 0,
            "rendering_index": 0, "rendering_total": 0,
        }
        start_time = time.time()

        def phase_weights() -> dict:
            total = max(1.0, estimate["whisper_seconds"] + estimate["scoring_seconds"] + estimate["rendering_seconds"])
            return {
                "whisper": estimate["whisper_seconds"] / total,
                "scoring": estimate["scoring_seconds"] / total,
                "rendering": estimate["rendering_seconds"] / total,
            }

        weights = phase_weights()

        def overall_fraction() -> float:
            frac = 0.0
            if run_state["current_step"] >= 2:
                frac += weights["whisper"]
            if run_state["scoring_total"]:
                scoring_frac = run_state["scoring_index"] / run_state["scoring_total"]
                frac += weights["scoring"] * (1.0 if run_state["current_step"] > 3 else scoring_frac)
            if run_state["rendering_total"]:
                rendering_frac = run_state["rendering_index"] / run_state["rendering_total"]
                frac += weights["rendering"] * rendering_frac
            if run_state["current_step"] >= 7:
                frac = 1.0
            return min(1.0, frac)

        def on_event(event: dict) -> None:
            if event["type"] == "step":
                run_state["current_step"] = event["step"]
                status_box.update(label=f"[{event['step']}/7] {event['message']}")
                log_lines.append(f"**Step {event['step']}/7** — {event['message']}")
            elif event["type"] == "substep" and event.get("stage") == "scoring":
                run_state["scoring_index"] = event["index"]
                run_state["scoring_total"] = event["total"]
            elif event["type"] == "clip_ready":
                run_state["rendering_index"] = event["index"]
                run_state["rendering_total"] = event["total"]
                log_lines.append(event["message"])
                st.session_state["last_result"].append(event["clip"])
                with results_container:
                    render_clip_card(event["clip"])
            elif event["type"] == "log":
                log_lines.append(event["message"])

            frac = overall_fraction()
            progress_bar.progress(frac)
            elapsed = time.time() - start_time
            remaining = max(0.0, estimate["total_seconds"] * (1 - frac))
            timer_placeholder.markdown(
                f"⏱️ Elapsed: **{fmt_seconds(elapsed)}** &nbsp;|&nbsp; "
                f"Estimated remaining: **{fmt_seconds(remaining)}**"
            )
            log_box.markdown("\n\n".join(log_lines[-12:]))

        result = run_pipeline(
            input_path=input_path,
            num_clips=cfg.num_clips,
            min_score=cfg.min_score,
            force_retranscribe=force_retranscribe,
            progress_callback=on_event,
        )

        if result["error"]:
            status_box.update(label="Failed", state="error")
            st.error(result["error"])
        else:
            status_box.update(label="Done", state="complete")
            st.success(f"Generated {len(result['clips'])} Short(s) in {fmt_seconds(time.time() - start_time)}.")

        if st.session_state["last_result"]:
            st.caption(
                "Virality score is Qwen3 8B's own estimate across hook, curiosity, emotion, "
                "payoff, standalone clarity, shareability, clarity, and retention — a useful "
                "ranking signal, not a guarantee of real-world performance."
            )

# --------------------------------------------------------------------------- #
# Results gallery for page loads where nothing was just generated — shows the
# last completed run's output (from session state, or metadata.json on disk).
# --------------------------------------------------------------------------- #
if not generated_this_run:
    st.divider()
    st.header("Results")

    clips = st.session_state.get("last_result")
    if not clips:
        meta = read_json(config.OUTPUT_DIR / "metadata.json")
        clips = meta["clips"] if meta else []
        if clips:
            st.caption("Showing the most recent completed run's output.")

    if not clips:
        st.info("No Shorts generated yet.")
    else:
        for clip in clips:
            render_clip_card(clip)
        st.caption(
            "Virality score is Qwen3 8B's own estimate across hook, curiosity, emotion, "
            "payoff, standalone clarity, shareability, clarity, and retention — a useful "
            "ranking signal, not a guarantee of real-world performance."
        )
