"""Tests for robust LLM JSON parsing/repair — must never crash on bad output."""
from app.analysis.llm_scorer import _parse_llm_json, _validate_llm_result

GOOD_JSON = """{
  "category": "STORY",
  "hook": "He reveals why he quit his $200k job",
  "reason": "Strong curiosity gap with a clear payoff.",
  "scores": {"hook": 9, "curiosity": 9, "emotion": 8, "payoff": 9,
             "standalone": 8, "shareability": 9, "clarity": 9, "retention": 9}
}"""


def test_parses_clean_json():
    data = _parse_llm_json(GOOD_JSON)
    assert data is not None
    assert data["category"] == "STORY"


def test_strips_markdown_fences():
    wrapped = f"```json\n{GOOD_JSON}\n```"
    data = _parse_llm_json(wrapped)
    assert data is not None
    assert data["category"] == "STORY"


def test_extracts_json_with_surrounding_prose():
    wrapped = f"Sure, here's the analysis:\n{GOOD_JSON}\nLet me know if you need more."
    data = _parse_llm_json(wrapped)
    assert data is not None
    assert data["category"] == "STORY"


def test_repairs_trailing_comma():
    broken = GOOD_JSON.replace('"retention": 9}', '"retention": 9,}')
    data = _parse_llm_json(broken)
    assert data is not None


def test_totally_invalid_returns_none_not_crash():
    data = _parse_llm_json("not json at all, just plain text response")
    assert data is None


def test_validate_rejects_missing_scores():
    assert _validate_llm_result({"category": "STORY"}) is None


def test_validate_clamps_out_of_range_scores():
    data = {
        "category": "FUNNY",
        "scores": {"hook": 15, "curiosity": -3, "emotion": 5, "payoff": 5,
                   "standalone": 5, "shareability": 5, "clarity": 5, "retention": 5},
    }
    result = _validate_llm_result(data)
    assert result["scores"]["hook"] == 10.0
    assert result["scores"]["curiosity"] == 0.0


def test_validate_defaults_unknown_category_to_other():
    data = {
        "category": "NOT_A_REAL_CATEGORY",
        "scores": {k: 5 for k in ["hook", "curiosity", "emotion", "payoff",
                                   "standalone", "shareability", "clarity", "retention"]},
    }
    result = _validate_llm_result(data)
    assert result["category"] == "OTHER"
