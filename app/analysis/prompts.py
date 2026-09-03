"""Prompt templates for Qwen3 8B candidate scoring."""
from __future__ import annotations

from app.config import CATEGORIES

_CATEGORIES_STR = ", ".join(CATEGORIES)

SYSTEM_PROMPT = """You are an expert short-form video editor who has produced viral \
YouTube Shorts / TikTok clips for years. You evaluate transcript excerpts and judge \
whether they would work as a standalone vertical short video.

You MUST respond with ONLY a single JSON object. No markdown fences, no commentary, \
no explanation before or after the JSON."""

SCORING_INSTRUCTIONS = """Evaluate the CANDIDATE CLIP transcript below as a potential short-form video.

Score each of these from 0-10 (integers):
- hook: does it grab attention in the first 1-2 seconds?
- curiosity: does it make the viewer want to know what happens next?
- emotion: is it funny, shocking, emotional, exciting, or controversial?
- payoff: does the clip actually deliver something (an answer, a punchline, a reveal)?
- standalone: can someone understand it with zero context from the rest of the video?
- shareability: would someone send this to a friend?
- clarity: is the message easy to follow?
- retention: does the structure encourage watching to the very end?

Classify the clip into exactly one category: {categories}

Respond with ONLY this JSON structure (no other text):
{{
  "category": "ONE_OF_THE_CATEGORIES",
  "hook": "a short 5-10 word description of the hook/premise",
  "reason": "one sentence on why this would or wouldn't work as a short",
  "scores": {{
    "hook": 0,
    "curiosity": 0,
    "emotion": 0,
    "payoff": 0,
    "standalone": 0,
    "shareability": 0,
    "clarity": 0,
    "retention": 0
  }}
}}

CANDIDATE CLIP TRANSCRIPT (timestamps {start:.1f}s - {end:.1f}s, duration {duration:.1f}s):
\"\"\"
{text}
\"\"\"
"""


def build_scoring_prompt(start: float, end: float, text: str) -> str:
    return SCORING_INSTRUCTIONS.format(
        start=start, end=end, duration=end - start, text=text, categories=_CATEGORIES_STR,
    )
