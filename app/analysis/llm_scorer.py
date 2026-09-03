"""Score candidate clips with Qwen3 8B via the local Ollama HTTP API.

Robust to malformed/partial JSON from the LLM: attempts direct parsing,
then fence-stripping + brace-extraction, then json_repair, then retries
the request. Never raises just because the model returned bad JSON —
a candidate that can't be scored after all retries is dropped with a
warning, not a crash.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

from app import config
from app.analysis.candidate_generator import Candidate
from app.analysis.prompts import SYSTEM_PROMPT, build_scoring_prompt
from app.config import CATEGORIES
from app.utils.files import read_json, stable_hash, write_json

logger = logging.getLogger("shorts.llm_scorer")

REQUIRED_SCORE_KEYS = [
    "hook", "curiosity", "emotion", "payoff",
    "standalone", "shareability", "clarity", "retention",
]


@dataclass
class ScoredClip:
    candidate_id: str
    start: float
    end: float
    text: str
    category: str
    hook: str
    reason: str
    scores: dict
    final_score: float


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _extract_braces(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _parse_llm_json(raw_text: str) -> dict | None:
    """Best-effort recovery of a JSON object from raw LLM output."""
    candidates_to_try = [raw_text, _strip_markdown_fences(raw_text)]

    braces = _extract_braces(_strip_markdown_fences(raw_text))
    if braces:
        candidates_to_try.append(braces)

    for candidate in candidates_to_try:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Last resort: try json_repair, which tolerates trailing commas,
    # unquoted keys, truncated objects, etc.
    try:
        from json_repair import repair_json
        target = braces or _strip_markdown_fences(raw_text)
        repaired = repair_json(target)
        return json.loads(repaired)
    except Exception:
        return None


def _validate_llm_result(data: dict) -> dict | None:
    if not isinstance(data, dict):
        return None
    scores = data.get("scores")
    if not isinstance(scores, dict):
        return None
    clean_scores = {}
    for key in REQUIRED_SCORE_KEYS:
        val = scores.get(key)
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        clean_scores[key] = max(0.0, min(10.0, val))

    category = str(data.get("category", "OTHER")).upper().strip()
    if category not in CATEGORIES:
        category = "OTHER"

    return {
        "category": category,
        "hook": str(data.get("hook", "")).strip(),
        "reason": str(data.get("reason", "")).strip(),
        "scores": clean_scores,
    }


def _call_ollama(prompt: str, host: str, model: str, timeout: int) -> str:
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def _weighted_score(scores: dict, weights: dict) -> float:
    return round(sum(scores[k] * weights[k] for k in REQUIRED_SCORE_KEYS), 2)


def score_candidate(
    candidate: Candidate,
    cfg: "config.Config",
    cache_dir_file,
) -> ScoredClip | None:
    cache_key = stable_hash({"id": candidate.id, "model": cfg.ollama_model})
    cache_path = cache_dir_file(cache_key)
    cached = read_json(cache_path)
    if cached is not None:
        return ScoredClip(**cached)

    prompt = build_scoring_prompt(candidate.start, candidate.end, candidate.text)

    last_error = None
    for attempt in range(1, cfg.llm_max_retries + 1):
        try:
            raw = _call_ollama(prompt, cfg.ollama_host, cfg.ollama_model, cfg.llm_request_timeout)
        except requests.RequestException as e:
            last_error = e
            logger.warning("Ollama request failed (attempt %d/%d): %s", attempt, cfg.llm_max_retries, e)
            continue

        parsed = _parse_llm_json(raw)
        if parsed is None:
            last_error = f"Could not parse JSON from LLM output: {raw[:200]!r}"
            logger.warning("Attempt %d/%d: %s", attempt, cfg.llm_max_retries, last_error)
            continue

        validated = _validate_llm_result(parsed)
        if validated is None:
            last_error = f"LLM JSON missing required fields: {parsed!r}"
            logger.warning("Attempt %d/%d: %s", attempt, cfg.llm_max_retries, last_error)
            continue

        final_score = _weighted_score(validated["scores"], cfg.score_weights)
        result = ScoredClip(
            candidate_id=candidate.id,
            start=candidate.start,
            end=candidate.end,
            text=candidate.text,
            category=validated["category"],
            hook=validated["hook"],
            reason=validated["reason"],
            scores=validated["scores"],
            final_score=final_score,
        )
        write_json(cache_path, result.__dict__)
        return result

    logger.error(
        "Giving up scoring candidate %s after %d attempts: %s",
        candidate.id, cfg.llm_max_retries, last_error,
    )
    return None


def score_all_candidates(
    candidates: list[Candidate],
    cfg: "config.Config",
    cache_namespace: str,
    on_progress: Callable[[int, int, float, bool], None] | None = None,
) -> tuple[list[ScoredClip], list[float]]:
    """Returns (scored_clips, per_candidate_seconds) — the timing list excludes
    cache hits so it reflects real LLM latency for calibration purposes."""
    cache_subdir = config.CACHE_DIR / "llm_scores" / cache_namespace

    def cache_path_for(key: str):
        return cache_subdir / f"{key}.json"

    results = []
    timings = []
    for i, candidate in enumerate(candidates, 1):
        logger.info("Scoring candidate %d/%d (%.1fs-%.1fs)", i, len(candidates), candidate.start, candidate.end)
        cache_key = stable_hash({"id": candidate.id, "model": cfg.ollama_model})
        was_cached = cache_path_for(cache_key).exists()

        t0 = time.time()
        scored = score_candidate(candidate, cfg, cache_path_for)
        elapsed = time.time() - t0

        if scored:
            results.append(scored)
        if not was_cached:
            timings.append(elapsed)
        if on_progress:
            on_progress(i, len(candidates), elapsed, was_cached)
    return results, timings
