# AI Shorts Generator (Fully Local)

Turns a long-form video into ranked, captioned, vertical YouTube Shorts —
entirely on-device. No cloud AI APIs (no OpenAI, no Gemini, no Anthropic).

Everything — code, models, caches, temp files, logs, and output — lives under
`F:\AI-Shorts-Generator\`. Nothing project-related is written to `C:\`.

## Stack

| Stage | Tool |
|---|---|
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo`, CUDA |
| Clip scoring | Qwen3 8B via [Ollama](https://ollama.com) (local HTTP API) |
| Cutting / reframing / captions | FFmpeg (portable build in `tools/ffmpeg`) |
| Face-aware crop | OpenCV Haar cascade (v1; modular, swappable for a vision model later) |

Hardware target: Windows, RTX 4060 Laptop (8GB VRAM), 16GB RAM.

## One-time setup

```powershell
# 1. Create the venv (already done if you're reading this after initial setup)
python -m venv F:\AI-Shorts-Generator\venv

# 2. Install dependencies
F:\AI-Shorts-Generator\venv\Scripts\pip install -r F:\AI-Shorts-Generator\requirements.txt

# 3. Make sure Ollama's model store points at F: (see "Storage layout" below),
#    then pull the LLM:
ollama pull qwen3:8b
```

FFmpeg is already bundled at `tools\ffmpeg\` — nothing to install separately.

## Usage

Drop a video into `input\`, then from the project root:

```powershell
F:\AI-Shorts-Generator\venv\Scripts\python -m app.main input.mp4
F:\AI-Shorts-Generator\venv\Scripts\python -m app.main input.mp4 --clips 10
F:\AI-Shorts-Generator\venv\Scripts\python -m app.main input.mp4 --min-score 7.5
F:\AI-Shorts-Generator\venv\Scripts\python -m app.main input.mp4 --force-retranscribe
```

Output lands in `output\short_01.mp4`, `short_02.mp4`, ... plus
`output\metadata.json` describing every clip (timestamps, score breakdown,
category, hook, reason, caption file).

## Pipeline

```
video -> audio extraction -> faster-whisper transcription (cached)
      -> candidate window generation (15-60s, sentence-aligned)
      -> Qwen3 8B scoring (hook/curiosity/emotion/payoff/standalone/
         shareability/clarity/retention, cached per candidate)
      -> overlap/duplicate removal -> category-diverse final selection
      -> word-boundary timestamp refinement
      -> cut + vertical (9:16) reframe -> ASS captions burned in
      -> optional effects (zoom / silence removal / SFX / music — off by default)
      -> output/short_NN.mp4 + metadata.json
```

## Configuration

Edit `config.json` (created on first run) or pass CLI flags. Notable fields —
see `app/config.py` for the full list and defaults:

- `whisper_model`, `whisper_device`, `whisper_compute_type`
- `ollama_model`, `ollama_host`
- `num_clips`, `min_score`, `score_weights` (must sum to 1.0)
- `min_clip_seconds`, `max_clip_seconds`, `max_candidates`
- `output_width`, `output_height`, `video_crf`, `video_preset`
- `caption_style` (font, size, colors, position, words-per-caption)
- `enable_zoom`, `enable_silence_removal`, `enable_music`, `enable_sfx`
  (all `false` by default — default output is video + original audio + captions)

## Storage layout (everything on F:)

```
F:\AI-Shorts-Generator\
  app\                 pipeline source code
  input\                you place source videos here
  output\               short_01.mp4 ... + metadata.json
  transcripts\          cached Whisper JSON transcripts (skip re-transcription)
  captions\             generated .ass subtitle files
  temp\                 working files, cleaned up after each run
  cache\                LLM scoring cache
  logs\                 one log file per run
  models\
    ollama\             Ollama model store (OLLAMA_MODELS env var points here)
    huggingface\        HF_HOME (used if any HF downloads occur)
    faster-whisper\     faster-whisper's own model cache (download_root)
  tools\ffmpeg\         portable ffmpeg/ffprobe/ffplay binaries
  venv\                 Python virtual environment
```

`OLLAMA_MODELS` is a systemwide (not per-project) environment variable — it
was repointed from its previous location to `F:\AI-Shorts-Generator\models\ollama`
during setup, so **all** Ollama models on this machine now live on F:.

## Caching / resumability

- Transcripts are cached by (filename, size, mtime, model) — re-running the
  same video reuses the cached transcript unless `--force-retranscribe` is passed.
- LLM scores are cached per candidate — re-running scoring/selection after a
  code change doesn't re-hit Ollama for candidates already scored.

## Known v1 limitations (by design — see "Future extension")

- Reframing is a single static crop per clip (average face position across a
  few sampled frames), not per-frame dynamic tracking.
- No vision/multimodal model is used yet — selection is transcript-only.
- No voiceover/narration is generated; original audio is always preserved.

## Tests

```powershell
F:\AI-Shorts-Generator\venv\Scripts\pytest F:\AI-Shorts-Generator\tests
```
