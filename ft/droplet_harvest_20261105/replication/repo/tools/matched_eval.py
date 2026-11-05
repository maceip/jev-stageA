"""Publication-grade evaluation guards for the Tuesday-v2 redo.

These close the failure documented in the Tuesday assessment: an *untrained*
head scored dev AUC 0.9752/0.9829 on the small curated CLEF sets, so a trained
AUC of 1.000 was not evidence of learning. A clean-looking number is not a
result; every headline must survive a matched twin, a free baseline, a
confidence interval, and an honesty probe.

All functions are pure and fail closed.
"""
from __future__ import annotations

import math
from collections import defaultdict

from redo_eval import evaluate_scores

WEAK_DELTA = 0.30          # delta < 0.30 -> weak
MAX_UNTRAINED_AUC = 0.80   # untrained head above this => eval set cannot discriminate


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Empty samples raise."""
    if n <= 0:
        raise ValueError("Wilson interval undefined for n=0")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def _fired(row: dict, threshold: float) -> bool:
    return float(row["score"]) >= threshold


def _label(row: dict) -> bool:
    return bool(row["label"])


def matched_pair_deltas(rows: list[dict], threshold: float) -> dict:
    """delta = P(fire | positive) - P(fire | its matched benign twin).

    `pair_id` joins the twins. A detector that fires on everything scores
    delta ~ 0 and is tagged BLIND. Never pool a pair with its twin in a
    cross-case analysis: twins share literals by construction.
    """
    by_pair: dict[str, dict[bool, list[dict]]] = defaultdict(lambda: {True: [], False: []})
    for r in rows:
        pid = r.get("pair_id")
        if pid is None:
            raise ValueError("matched_pair_deltas requires pair_id on every row")
        by_pair[str(pid)][_label(r)].append(r)
    deltas, table = [], []
    for pid, arms in sorted(by_pair.items()):
        if not arms[True] or not arms[False]:
            raise ValueError(f"pair {pid} is missing a positive or a matched twin")
        p_fire = sum(_fired(r, threshold) for r in arms[True]) / len(arms[True])
        p_twin = sum(_fired(r, threshold) for r in arms[False]) / len(arms[False])
        d = round(p_fire - p_twin, 4)
        tag = "BLIND" if d <= 0 else "weak" if d < WEAK_DELTA else "ok"
        deltas.append(d)
        table.append({"pair_id": pid, "n_pos": len(arms[True]), "n_twin": len(arms[False]),
                      "p_fire": round(p_fire, 4), "p_twin_fire": round(p_twin, 4),
                      "delta": d, "tag": tag})
    if not table:
        raise ValueError("no matched pairs supplied")
    return {"threshold": threshold, "n_pairs": len(table),
            "mean_delta": round(sum(deltas) / len(deltas), 4),
            "n_blind": sum(t["tag"] == "BLIND" for t in table),
            "n_weak": sum(t["tag"] == "weak" for t in table),
            "pairs": table,
            "footer": "Read the deltas, not the recall column."}


def mcnemar_exact(pairs: list[tuple[bool, bool]]) -> dict:
    """Exact binomial test on discordant pairs of two detectors on identical inputs."""
    b = sum(1 for a, c in pairs if a and not c)
    c = sum(1 for a, d in pairs if not a and d)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_exact": 1.0, "note": "no discordant pairs"}
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b": b, "c": c, "n_discordant": n, "p_exact": round(min(1.0, p), 6),
            "significant_at_0.05": bool(p < 0.05)}


def honesty_probe_rate(rows: list[dict], threshold: float) -> dict:
    """Cases that contain the phenomenon but nothing legitimately citable.

    Firing here is confabulation. A LOW number is the good number. Scored
    separately and excluded from precision/recall.
    """
    probe = [r for r in rows if r.get("honesty_probe") is True]
    if not probe:
        raise ValueError("no honesty-probe rows supplied")
    fired = sum(_fired(r, threshold) for r in probe)
    lo, hi = wilson(fired, len(probe))
    return {"n": len(probe), "fired": fired, "fire_rate": round(fired / len(probe), 4),
            "wilson95": [lo, hi], "excluded_from_headline": True}


def free_lunch_check(untrained_rows: list[dict], trained_rows: list[dict],
                     threshold: float = 0.5, max_untrained_auc: float = MAX_UNTRAINED_AUC) -> dict:
    """Fail the build when the UNTRAINED scorer already solves the eval set.

    This is the invariant that v1 violated. It constructs the inconsistency
    deliberately: if a random/untrained head is already at AUC >= 0.80, the
    corpus cannot discriminate and no trained metric is interpretable.
    """
    u = evaluate_scores(untrained_rows, threshold)
    t = evaluate_scores(trained_rows, threshold)
    ok = u["auc"] < max_untrained_auc
    return {"untrained_auc": u["auc"], "trained_auc": t["auc"],
            "max_untrained_auc": max_untrained_auc, "passed": ok,
            "verdict": "PASS" if ok else "FAIL: eval set is trivially separable before training",
            "note": "Run the free/no-model baseline on IDENTICAL rows."}


def publication_gate(untrained_rows, trained_rows, paired_rows, honesty_rows,
                     threshold: float, max_untrained_auc: float = MAX_UNTRAINED_AUC) -> dict:
    """Single fail-closed gate. Raises rather than returning a partial pass."""
    free = free_lunch_check(untrained_rows, trained_rows, threshold, max_untrained_auc)
    pairs = matched_pair_deltas(paired_rows, threshold)
    honest = honesty_probe_rate(honesty_rows, threshold)
    failures = []
    if not free["passed"]:
        failures.append("FREE_LUNCH")
    if pairs["n_blind"] > 0 or pairs["mean_delta"] < WEAK_DELTA:
        failures.append("BLIND_OR_WEAK_MATCHED_PAIRS")
    report = {"free_lunch": free, "matched_pairs": pairs, "honesty_probe": honest,
              "failures": failures, "passed": not failures}
    if failures:
        raise ValueError("PUBLICATION_GATE_FAILED " + ",".join(failures) + " :: " + str(
            {"untrained_auc": free["untrained_auc"], "mean_delta": pairs["mean_delta"]}))
    return report


if __name__ == "__main__":
    # Self-test that CONSTRUCTS the v1 inconsistency, not a clean fixture.
    untrained = [{"score": 0.9 if i % 2 == 0 else 0.1, "label": i % 2 == 0} for i in range(100)]
    trained = [{"score": 0.95 if i % 2 == 0 else 0.05, "label": i % 2 == 0} for i in range(100)]
    pairs = []
    for i in range(50):
        pairs.append({"pair_id": f"p{i}", "score": 0.9, "label": True})
        pairs.append({"pair_id": f"p{i}", "score": 0.9, "label": False})  # twin fires too -> BLIND
    honesty = [{"score": 0.95, "label": True, "honesty_probe": True} for _ in range(20)]
    try:
        publication_gate(untrained, trained, pairs, honesty, 0.5)
        print("SELF_TEST_FAIL: gate accepted the v1 inconsistency")
        raise SystemExit(1)
    except ValueError as e:
        print("SELF_TEST_OK: gate rejected the v1 inconsistency ->", str(e)[:160])
