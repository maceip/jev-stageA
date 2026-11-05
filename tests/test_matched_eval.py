"""Tests that CONSTRUCT the v1 failure, not a clean happy path."""
import pytest

from matched_eval import (free_lunch_check, honesty_probe_rate, matched_pair_deltas,
                          mcnemar_exact, publication_gate, wilson)


def _separable(n=100, pos_hi=0.9, neg_lo=0.1):
    return [{"score": pos_hi if i % 2 == 0 else neg_lo, "label": i % 2 == 0} for i in range(n)]


def test_wilson_rejects_empty():
    with pytest.raises(ValueError):
        wilson(0, 0)
    lo, hi = wilson(20, 20)
    assert lo > 0.8 and hi == 1.0


def test_free_lunch_detects_untrained_head_that_already_solves():
    """The exact v1 defect: untrained AUC 0.98 -> trained 1.00 is not evidence."""
    untrained = _separable(pos_hi=0.95, neg_lo=0.05)
    trained = _separable(pos_hi=0.99, neg_lo=0.01)
    r = free_lunch_check(untrained, trained)
    assert r["untrained_auc"] >= 0.98
    assert r["passed"] is False
    assert "FAIL" in r["verdict"]


def test_matched_pairs_tag_everything_firing_as_blind():
    rows = []
    for i in range(20):
        rows.append({"pair_id": f"p{i}", "score": 0.9, "label": True})
        rows.append({"pair_id": f"p{i}", "score": 0.9, "label": False})
    r = matched_pair_deltas(rows, 0.5)
    assert r["mean_delta"] == 0.0
    assert r["n_blind"] == 20


def test_matched_pairs_require_real_twins():
    with pytest.raises(ValueError):
        matched_pair_deltas([{"pair_id": "x", "score": 0.9, "label": True}], 0.5)
    with pytest.raises(ValueError):
        matched_pair_deltas([{"score": 0.9, "label": True}], 0.5)


def test_honesty_probe_is_scored_separately():
    rows = [{"score": 0.9, "label": True, "honesty_probe": True} for _ in range(10)]
    r = honesty_probe_rate(rows, 0.5)
    assert r["excluded_from_headline"] is True
    assert r["fire_rate"] == 1.0  # a BAD result, reported honestly


def test_mcnemar_nonsignificance_is_stated_not_rounded():
    # 16 vs 6 discordant -> p = 0.052, directionally supported, formally unproven
    pairs = [(True, False)] * 16 + [(False, True)] * 6
    r = mcnemar_exact(pairs)
    assert r["n_discordant"] == 22
    assert r["p_exact"] > 0.05
    assert r["significant_at_0.05"] is False


def test_publication_gate_fails_closed_on_v1_inconsistency():
    untrained = _separable(pos_hi=0.95, neg_lo=0.05)
    trained = _separable(pos_hi=0.99, neg_lo=0.01)
    pairs = []
    for i in range(10):
        pairs.append({"pair_id": f"p{i}", "score": 0.9, "label": True})
        pairs.append({"pair_id": f"p{i}", "score": 0.9, "label": False})
    honesty = [{"score": 0.9, "label": True, "honesty_probe": True} for _ in range(10)]
    with pytest.raises(ValueError, match="PUBLICATION_GATE_FAILED"):
        publication_gate(untrained, trained, pairs, honesty, 0.5)


def test_publication_gate_passes_only_on_genuine_discrimination():
    untrained = [{"score": 0.5, "label": i % 2 == 0} for i in range(100)]
    trained = _separable(pos_hi=0.9, neg_lo=0.1)
    pairs = []
    for i in range(10):
        pairs.append({"pair_id": f"p{i}", "score": 0.9, "label": True})
        pairs.append({"pair_id": f"p{i}", "score": 0.1, "label": False})
    honesty = [{"score": 0.1, "label": True, "honesty_probe": True} for _ in range(10)]
    r = publication_gate(untrained, trained, pairs, honesty, 0.5)
    assert r["passed"] is True
    assert r["matched_pairs"]["mean_delta"] > 0.3
