# AI Shorts Generator

Turn a long‑form video (or a YouTube URL) into a handful of **ranked, captioned, vertical
9:16 Shorts** — running **entirely on your own machine**. No OpenAI, no Gemini, no
Anthropic, no cloud transcription. The only network calls are to a **local** Ollama
server and (optionally) `yt-dlp` fetching a public video you point it at.

```
long video  ──▶  transcribe  ──▶  find candidate windows  ──▶  score with a local LLM
            ──▶  pick the best, de‑duplicate  ──▶  cut · reframe · burn captions
            ──▶  output/short_01.mp4 … + metadata.json
```

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running it](#running-it)
- [Configuration](#configuration)
- [Output](#output)
- [Scoring rubric](#scoring-rubric)
- [Caching & resumability](#caching--resumability)
- [Project layout](#project-layout)
- [Storage layout](#storage-layout)
- [Troubleshooting](#troubleshooting)
- [Limitations (v1)](#limitations-v1)
- [Tests](#tests)
- [License & responsible use](#license--responsible-use)

---

## Why this exists

Most "long video → Shorts" tools upload your footage to a SaaS backend and bill per
minute. This one keeps everything local:

| Stage | Tool | Runs where |
|---|---|---|
| Speech‑to‑text | [faster‑whisper](https://github.com/SYSTRAN/faster-whisper) `large‑v3‑turbo` | Your GPU (or CPU) |
| Clip scoring | [Qwen3 8B](https://ollama.com/library/qwen3) via [Ollama](https://ollama.com) | Your machine, `localhost:11434` |
| Cut / reframe / captions | [FFmpeg](https://ffmpeg.org) | Your machine |
| Face‑aware crop | OpenCV Haar cascade | Your machine |
| YouTube fetch (optional) | [yt‑dlp](https://github.com/yt-dlp/yt-dlp) | Downloads the public stream only |

Nothing about your video leaves the box.

---

## Features

- **CLI and Streamlit UI** over the exact same pipeline (`app/pipeline/pipeline.py`) — the
  web app is a front‑end, not a fork.
- **Word‑level captions** burned in as styled ASS, with a karaoke‑style highlight on the
  word currently being spoken. Configurable font, size, colours, position, words‑per‑line.
- **Automatic 9:16 reframing** — samples frames, detects the dominant face, and centres a
  vertical crop on it. Falls back to a centre crop when no face is found.
- **LLM clip scoring** on 8 axes (hook, curiosity, emotion, payoff, standalone,
  shareability, clarity, retention) with a weighted final score and a one‑line reason.
- **Category‑diverse selection** — round‑robins across FUNNY / SHOCKING / STORY /
  EDUCATIONAL / … so you don't get 10 near‑identical clips.
- **Overlap removal** — candidates that overlap too much (IoU) are de‑duplicated before
  selection.
- **Word‑boundary snapping** so cuts never land mid‑word or leave dead air at the edges.
- **Aggressive caching** — transcripts keyed on `(filename, size, mtime, model)`; LLM
  scores cached per candidate. Re‑runs are cheap.
- **Self‑calibrating ETA** — records how fast each stage actually ran on *your* hardware
  and uses an EMA to predict the next run.
- **Optional effects, all off by default**: slow zoom, silence removal, background music,
  SFX. Default output is video + original audio + captions.
- **YouTube URL support** — paste a link; `yt-dlp` pulls it into `input/` and the rest of
  the pipeline treats it like any local file.
- Graceful degradation: no GPU → falls back to CPU with a warning; malformed LLM JSON →
  repaired / retried / the candidate is dropped, never a crash.

---

## How it works

Seven stages (`TOTAL_STEPS = 7` in `pipeline.py`):

1. **Transcribe** — extract audio with FFmpeg, run faster‑whisper with
   `word_timestamps=True` and VAD filtering. Result cached to `transcripts/*.json`.
2. **Generate candidates** — split the transcript into sentences, then build every
   `min_clip_seconds…max_clip_seconds` window that starts and ends on a sentence
   boundary. Windows that are mostly silence are skipped. Capped at `max_candidates`.
3. **Score** — each candidate's transcript excerpt is sent to Qwen3 8B via Ollama with a
   strict JSON‑only prompt. Scores are cached per candidate (keyed on a content hash).
4. **Select** — drop anything below `min_score`, greedily remove overlaps
   (`overlap_iou_threshold`), then round‑robin across categories up to `num_clips`. If
   fewer clips clear the bar than you asked for, it says so rather than lowering quality.
5. **Refine timestamps** — snap each clip's start/end to the first/last spoken word,
   with small padding.
6. **Render** — for each clip: cut → 9:16 reframe → encode (`libx264`, configurable CRF /
   preset) → generate ASS captions → burn them in → apply optional effects → copy to
   `output/short_NN.mp4`. `metadata.json` is rewritten after *every* clip so a UI can
   show results as they land.
7. **Done** — timing stats recorded for future estimates.

---

## Requirements

**Hardware** — developed and tuned on **Windows 11, RTX 4060 Laptop (8 GB VRAM), 16 GB
RAM**. An NVIDIA GPU is strongly recommended; CPU transcription works but is much slower.

**Software**

| Need | Notes |
|---|---|
| **Python 3.10+** | Developed on 3.14. |
| **NVIDIA driver + CUDA 12‑capable GPU** | For GPU transcription. The CUDA *libraries* are installed via pip (see below) — you do **not** need the full CUDA Toolkit. |
| **[Ollama](https://ollama.com)** running locally | `ollama pull qwen3:8b` (or set `ollama_model` to something you have). |
| **FFmpeg** | Either on `PATH`, or drop `ffmpeg`/`ffprobe` binaries into `tools/ffmpeg/` (that folder is git‑ignored — binaries are too large for the repo). |

> **Path note:** `app/config.py` currently hardcodes
> `PROJECT_ROOT = Path("F:/AI-Shorts-Generator")` — every model cache, temp dir, and
> output dir is derived from it, deliberately keeping all data off `C:`. If you clone
> elsewhere, **edit that one line** (or point it at `Path(__file__).resolve().parents[1]`).

---

## Installation

```bash
git clone https://github.com/Ansh-Bharti/ai-shorts-generator.git
cd ai-shorts-generator

# 1. Virtual env
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

# 2. Core dependencies
pip install -r requirements.txt

# 3. GPU libraries (skip for CPU-only). Installs cuBLAS + cuDNN 9 for CUDA 12.
pip install -r requirements-gpu.txt

# 4. Local LLM
ollama pull qwen3:8b

# 5. FFmpeg — if it isn't already on PATH, put the binaries here:
#    tools/ffmpeg/ffmpeg.exe, tools/ffmpeg/ffprobe.exe
```

If you cloned outside `F:\AI-Shorts-Generator`, edit `PROJECT_ROOT` in `app/config.py`
now (see the path note above). All working directories are created automatically on first
import.

### Why `requirements-gpu.txt` is separate

CTranslate2 (faster‑whisper's backend) ships wheels built against **CUDA 12 + cuDNN 9**
but bundles neither on Windows. Without `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` you get
`RuntimeError: Library cublas64_12.dll is not found`. They total ~1.3 GB, so CPU‑only
users can skip them and set `whisper_device: "cpu"`.

---

## Running it

### Streamlit UI

```bash
venv\Scripts\streamlit run streamlit_app.py
# opens http://localhost:8501
```

Sidebar: pick a local file or paste a YouTube URL, set number of Shorts / minimum score /
Ollama model / caption position, toggle optional effects, hit **Generate**. A System
Status panel shows whether Ollama, the GPU, and FFmpeg are detected. Finished Shorts
appear with their score breakdown, category, and hook as they render.

### CLI

```bash
python -m app.main input.mp4
python -m app.main input.mp4 --clips 10
python -m app.main input.mp4 --min-score 7.5
python -m app.main input.mp4 --force-retranscribe
python -m app.main "https://www.youtube.com/watch?v=XXXXXXXXXXX" --clips 8
```

`input` can be a bare filename (resolved against `input/`), an absolute path, or a
YouTube URL. Flags override `config.json` for that run only.

| Flag | Effect |
|---|---|
| `--clips N` | Number of Shorts to produce (default from config). |
| `--min-score X` | Minimum weighted score, 0–10 (default from config). |
| `--force-retranscribe` | Ignore any cached transcript and re‑run Whisper. |

---

## Configuration

On first run, `config.json` is written with all defaults. Edit it, or pass CLI flags.
Full list and defaults live in `app/config.py`.

| Key | Default | Meaning |
|---|---|---|
| `whisper_model` | `large-v3-turbo` | Any faster‑whisper model name. |
| `whisper_device` | `auto` | `auto` → CUDA if available else CPU; or force `cuda` / `cpu`. |
| `whisper_compute_type` | `float16` | `float16` \| `int8_float16` \| `int8`. |
| `ollama_model` | `qwen3:8b` | Must be pulled in Ollama. |
| `ollama_host` | `http://localhost:11434` | Local Ollama endpoint. |
| `min_clip_seconds` / `max_clip_seconds` | `15` / `60` | Candidate window length bounds. |
| `silence_gap_threshold` | `0.6` | Silence (s) treated as a natural sentence break. |
| `max_silence_ratio` | `0.35` | Candidates more silent than this are skipped. |
| `max_candidates` | `40` | Cap on candidates sent to the LLM per video. |
| `num_clips` | `10` | How many Shorts to select. |
| `min_score` | `6.5` | Weighted‑score threshold for selection. |
| `score_weights` | see below | Per‑axis weights; **must sum to 1.0** (validated). |
| `overlap_iou_threshold` | `0.5` | Candidates overlapping more than this are duplicates. |
| `prefer_category_diversity` | `true` | Round‑robin across categories when selecting. |
| `llm_max_retries` / `llm_request_timeout` | `3` / `120` | LLM robustness knobs. |
| `output_width` / `output_height` | `1080` / `1920` | Output frame size (9:16). |
| `video_crf` / `video_preset` | `18` / `medium` | `libx264` quality / speed. |
| `caption_style` | see `CaptionStyle` | `font_name`, `font_size`, `primary_color`, `highlight_color`, `outline_color`, `outline_width`, `shadow`, `position` (`bottom`\|`center`\|`top`), `max_words_per_caption`. Colours are ASS `&HAABBGGRR`. |
| `enable_zoom` / `enable_silence_removal` / `enable_music` / `enable_sfx` | `false` | Optional effects. Music/SFX need files in `assets/music` / `assets/sfx`. |

---

## Output

```
output/
  short_01.mp4
  short_02.mp4
  ...
  metadata.json
```

`metadata.json` (rewritten after each clip finishes):

```jsonc
{
  "source": "F:\\AI-Shorts-Generator\\input\\my_video.mp4",
  "run_id": "20260903_192909",
  "requested_clips": 10,
  "generated_clips": 3,
  "clips": [
    {
      "clip": "short_01",
      "source": "my_video.mp4",
      "start": 412.83,
      "end": 447.11,
      "duration": 34.28,
      "score": 8.4,
      "category": "STORY",
      "hook": "The one email that changed everything",
      "reason": "Strong cold open, clear payoff within 30s, works with zero context.",
      "score_breakdown": {
        "hook": 9, "curiosity": 8, "emotion": 7, "payoff": 9,
        "standalone": 8, "shareability": 8, "clarity": 9, "retention": 8
      },
      "caption_file": ".../captions/my_video_short_01.ass",
      "output_file": ".../output/short_01.mp4"
    }
  ]
}
```

Generated `.ass` caption files are kept in `captions/`.

---

## Scoring rubric

Each candidate is scored 0–10 (integers) by the LLM on eight axes; the final score is the
weighted sum (default weights):

| Axis | Weight | Question |
|---|---|---|
| `hook` | 0.20 | Does it grab attention in the first 1–2 seconds? |
| `curiosity` | 0.15 | Does it make you want to know what happens next? |
| `payoff` | 0.15 | Does the clip actually deliver (answer / punchline / reveal)? |
| `retention` | 0.15 | Does the structure pull you to the end? |
| `emotion` | 0.10 | Funny, shocking, emotional, exciting, controversial? |
| `standalone` | 0.10 | Understandable with zero surrounding context? |
| `shareability` | 0.10 | Would someone send it to a friend? |
| `clarity` | 0.05 | Is the message easy to follow? |

Categories: `FUNNY`, `SHOCKING`, `STORY`, `EDUCATIONAL`, `EMOTIONAL`, `CONTROVERSIAL`,
`INSPIRATIONAL`, `UNEXPECTED`, `OTHER`.

---

## Caching & resumability

- **Transcripts** — `transcripts/<stem>_<model>_<fingerprint>.json`, fingerprint =
  `(filename, size, mtime)`. Same video + same model ⇒ no re‑transcription unless
  `--force-retranscribe`.
- **LLM scores** — cached per candidate under `cache/llm_scores/`, keyed on a stable hash
  of the candidate text + rubric. Change code, re‑run, and only *new* candidates hit
  Ollama.
- **Timing stats** — `cache/timing_stats.json`, an EMA of real per‑stage timings used for
  the pre‑run ETA.

---

## Project layout

```
app/
  config.py                 all paths + the Config dataclass (+ config.json load/save)
  main.py                   CLI entry point
  pipeline/pipeline.py      the 7-stage orchestrator (used by CLI and UI alike)
  transcription/whisper.py  faster-whisper wrapper, caching, CUDA DLL registration
  analysis/
    candidate_generator.py  transcript → sentence-aligned candidate windows
    llm_scorer.py           Ollama scoring + tolerant JSON parsing/repair
    prompts.py              system + scoring prompt templates
  video/
    cutter.py               refine bounds, cut + reframe orchestration
    reframer.py             OpenCV face detection → 9:16 crop filter
    captions.py             word-level ASS caption generation (karaoke highlight)
    effects.py              optional zoom / silence-removal / music / SFX
  utils/
    ffmpeg.py               every FFmpeg/ffprobe invocation
    youtube.py              yt-dlp download + URL detection + metadata
    files.py                fingerprints, hashing, safe stems, JSON IO
    validation.py           input-file and disk-space checks
    stats.py                self-calibrating timing estimates
streamlit_app.py            local web UI
tests/                      41 pytest tests (pure logic, no GPU/network needed)
requirements.txt            core deps
requirements-gpu.txt        CUDA 12 libs for GPU transcription
```

---

## Storage layout

Everything is derived from `PROJECT_ROOT`, so no project data touches `C:` (by design):

```
<PROJECT_ROOT>/
  input/         source videos you drop in (git-ignored)
  output/        short_NN.mp4 + metadata.json (git-ignored)
  transcripts/   cached Whisper JSON (git-ignored)
  captions/      generated .ass files (git-ignored)
  temp/          per-run working dirs, deleted afterward (git-ignored)
  cache/         LLM score cache + timing_stats.json (git-ignored)
  logs/          one log file per run (git-ignored)
  models/
    faster-whisper/   faster-whisper download_root
    huggingface/      HF_HOME (if any HF download happens)
    ollama/           Ollama model store (if OLLAMA_MODELS points here)
  tools/ffmpeg/  portable ffmpeg/ffprobe/ffplay (git-ignored — provide your own)
  venv/          virtual environment (git-ignored)
```

`app/config.py` sets `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE`, `TORCH_HOME`,
and `XDG_CACHE_HOME` at import time (before any model‑downloading library loads) so those
caches also land under `models/` / `cache/` instead of your home directory.

---

## Troubleshooting

**`RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`**
CUDA 12 runtime libraries aren't installed. `pip install -r requirements-gpu.txt`. On
Windows these pip packages aren't auto‑added to the DLL search path, so
`app/transcription/whisper.py` registers their `bin/` folders via `os.add_dll_directory`
before importing faster‑whisper — make sure you're on the current version of that file.
No GPU? Set `whisper_device: "cpu"` in `config.json` instead.

**Transcription hangs at 0 % GPU right after "VAD filter removed"**
Usually a stale process: the Python/Streamlit process was started *before* the CUDA
libraries were installed, and a module hot‑reload can't re‑initialise native CUDA state.
Fully stop and relaunch the process.

**`AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`**
OpenCV 5 removed the Haar cascade API and bundled XML data. Pin to the 4.x line:
`pip install "opencv-python<5"` (already constrained in `requirements.txt`).

**`Ollama unreachable` / no candidates scored**
Start Ollama and confirm the model: `ollama list` should show `qwen3:8b` (or whatever
`ollama_model` is set to). Check `ollama_host` in `config.json`.

**`score_weights must sum to 1.0`**
Your edited weights in `config.json` don't add up. Fix them to total exactly 1.0.

**FFmpeg not found**
Put `ffmpeg.exe` / `ffprobe.exe` in `tools/ffmpeg/`, or install FFmpeg on `PATH`. The
Streamlit System Status panel shows which one is in use.

**Everything writes to the wrong drive / `F:` doesn't exist**
Edit `PROJECT_ROOT` in `app/config.py`.

---

## Limitations (v1)

- Reframing is **one static crop per clip** (average face position across a few sampled
  frames), not per‑frame tracking.
- Selection is **transcript‑only** — no vision/multimodal model looks at the pixels.
- **No generated voiceover/narration** — the original audio is always preserved.
- Face detection is a classic Haar cascade: fast, but weaker than a modern detector on
  profiles / poor lighting. The `compute_crop` interface is deliberately isolated so a
  better model can drop in without touching callers.
- Windows‑first. The code is mostly portable, but paths and the bundled‑FFmpeg
  convention assume Windows.

---

## Tests

```bash
venv\Scripts\pytest tests
```

41 tests covering candidate generation, selection/overlap logic, tolerant LLM‑JSON
parsing, timestamp refinement, validation, file utilities, and YouTube URL handling. No
GPU, network, or Ollama required.

---

## License & responsible use

No license file is included yet — all rights reserved by default until one is added. If
you want others to reuse this, add a `LICENSE` (MIT is a common choice for a project like
this).

Only process videos you have the right to repurpose. Downloading and re‑cutting YouTube
content may violate YouTube's Terms of Service and/or copyright depending on the video
and your jurisdiction — that's on you.
