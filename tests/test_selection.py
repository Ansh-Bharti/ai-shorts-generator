"""Tests for overlap removal, diversity selection, and score-based filtering."""
from app.analysis.llm_scorer import ScoredClip
from app.pipeline.pipeline import _diversity_round_robin, _greedy_non_overlapping, _iou, select_final_clips
from app.config import Config


def make_clip(start, end, score, category="STORY", cid=None):
    return ScoredClip(
        candidate_id=cid or f"c_{start}_{end}",
        start=start, end=end, text="text",
        category=category, hook="hook", reason="reason",
        scores={"hook": score}, final_score=score,
    )


def test_iou_no_overlap():
    a, b = make_clip(0, 10, 5), make_clip(20, 30, 5)
    assert _iou(a, b) == 0.0


def test_iou_full_overlap():
    a, b = make_clip(0, 10, 5), make_clip(0, 10, 5)
    assert _iou(a, b) == 1.0


def test_iou_partial_overlap():
    a, b = make_clip(0, 10, 5), make_clip(5, 15, 5)
    assert abs(_iou(a, b) - (5 / 15)) < 1e-6


def test_greedy_non_overlapping_keeps_highest_score_first():
    clips = [make_clip(0, 20, 9.0), make_clip(5, 25, 8.0), make_clip(100, 120, 7.0)]
    result = _greedy_non_overlapping(clips, threshold=0.3)
    assert result == [clips[0], clips[2]]


def test_diversity_round_robin_spreads_categories():
    clips = [
        make_clip(0, 10, 9.0, category="FUNNY"),
        make_clip(20, 30, 8.5, category="FUNNY"),
        make_clip(40, 50, 8.0, category="FUNNY"),
        make_clip(60, 70, 7.0, category="STORY"),
    ]
    result = _diversity_round_robin(clips, n=2)
    categories = {c.category for c in result}
    assert categories == {"FUNNY", "STORY"}


def test_select_final_clips_respects_min_score():
    cfg = Config(min_score=7.0, num_clips=10, overlap_iou_threshold=0.5)
    clips = [make_clip(0, 20, 9.0), make_clip(100, 120, 3.0)]
    selected, note = select_final_clips(clips, cfg)
    assert len(selected) == 1
    assert note is not None  # fewer than requested, should explain why


def test_select_final_clips_never_exceeds_requested_count():
    cfg = Config(min_score=0.0, num_clips=2, overlap_iou_threshold=0.9)
    clips = [make_clip(i * 100, i * 100 + 20, 9.0 - i * 0.1) for i in range(5)]
    selected, _ = select_final_clips(clips, cfg)
    assert len(selected) == 2
