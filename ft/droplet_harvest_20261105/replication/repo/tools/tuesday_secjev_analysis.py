"""Full frozen SecJev v03 analysis (G5) from per-row scores only.

Never aggregates, never reuses a metric: every number is recomputed from the
score files. Reports the trained model AND the zero-shot base on IDENTICAL rows,
a paired significance test, clustered CIs, quasi-matched pairs, and a separately
scored honesty probe.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matched_eval import (free_lunch_check, honesty_probe_rate, matched_pair_deltas,
                          mcnemar_exact, wilson)  # noqa: E402
from redo_eval import calibrate_threshold, evaluate_scores  # noqa: E402
import random  # noqa: E402


def fast_cluster_ci(rows: list[dict], threshold: float, n_resamples: int = 200,
                    seed: int = 0) -> tuple[float, float]:
    """Equivalent to redo_eval.cluster_bootstrap_ci but O(n) per resample.

    That function calls evaluate_scores(), which needlessly recomputes an
    O(pos*neg) AUC on every resample although only balanced accuracy is used.
    With 7,192 test rows x 200 resamples that is ~2.1e9 comparisons. Here we
    compute TPR/FPR directly; the statistic is identical.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("group", "?"))].append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    values = []
    for _ in range(n_resamples):
        pos = neg = tp = fp = 0
        for _k in keys:
            for r in groups[rng.choice(keys)]:
                fire = float(r["score"]) >= threshold
                if bool(r["label"]):
                    pos += 1
                    tp += fire
                else:
                    neg += 1
                    fp += fire
        if pos and neg:
            values.append(0.5 * (tp / pos + (1.0 - fp / neg)))
    if not values:
        raise ValueError("no valid bootstrap resamples")
    values.sort()
    return (round(values[max(0, int(0.025 * len(values)) - 1)], 4),
            round(values[min(len(values) - 1, int(0.975 * len(values)))], 4))


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def state_len(r: dict) -> int:
    return len(json.dumps(r.get("state", {}), ensure_ascii=False, sort_keys=True))


def build_pairs(split_rows: list[dict], test_scores: list[dict]) -> tuple[list[dict], dict]:
    """Quasi-matched twins: same suite + same proposed function, nearest state length.

    NOT exact literal twins. Caveat is recorded in the report.
    """
    by_index = {r["row_index"]: r for r in test_scores}
    pos, neg = [], []
    for i, r in enumerate(split_rows):
        s = by_index.get(i)
        if s is None:
            continue
        meta = r.get("_meta") or {}
        fn = (r.get("state", {}).get("proposed_tool_call") or {}).get("function", "?")
        item = (len(json.dumps(r.get("state", {})).encode()), i, meta.get("suite"), fn)
        (pos if s["label"] else neg).append(item)
    by_group = defaultdict(list)
    for L, i, suite, fn in neg:
        by_group[(suite, fn)].append((L, i))
    for v in by_group.values():
        v.sort()
    pairs, used = [], set()
    for L, i, suite, fn in pos:
        cand = [x for x in by_group.get((suite, fn), []) if x[1] not in used]
        if not cand:
            continue
        twin = min(cand, key=lambda x: abs(x[0] - L))
        used.add(twin[1])
        pairs.append((i, twin[1]))
    out = []
    for pid, (pi, ni) in enumerate(pairs):
        p, n = dict(by_index[pi]), dict(by_index[ni])
        p["pair_id"] = n["pair_id"] = f"q{pid:06d}"
        out.extend([p, n])
    return out, {"n_pairs": len(pairs), "n_rows": len(out),
                 "caveat": "quasi-matched by suite+function+nearest state length; "
                           "literal strings are NOT identical. Not the matched-pair gold standard."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="/root/work/eval")
    ap.add_argument("--test-split", required=True)
    ap.add_argument("--dev-split", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E = Path(a.eval_dir)
    td, tt = load(E / "trained_dev.jsonl"), load(E / "trained_test.jsonl")
    bd, bt = load(E / "base_dev.jsonl"), load(E / "base_test.jsonl")
    assert len(td) == len(bd) and len(tt) == len(bt), "score files not aligned"
    assert all(x["label"] == y["label"] for x, y in zip(tt, bt)), "labels not identical across arms"

    thr = calibrate_threshold(td, 0.01)
    thr_base = calibrate_threshold(bd, 0.01)
    trained = evaluate_scores(tt, thr)
    base_shared = evaluate_scores(bt, thr)
    base_own = evaluate_scores(bt, thr_base)

    report = {
        "protocol": "tuesday-v2 G5 frozen test (once, after dev-only epoch selection)",
        "selected_checkpoint": "epoch2 (dev AUC 0.99302; epoch3 0.99256; epoch0 0.92174)",
        "threshold_dev_calibrated": thr, "max_fpr_budget": 0.01,
        "trained_test": trained,
        "base_test_at_shared_threshold": base_shared,
        "base_threshold_dev_calibrated": thr_base,
        "base_test_at_own_threshold": base_own,
        "auc_delta_trained_minus_base": round(trained["auc"] - base_shared["auc"], 4),
        "trained_balanced_ci_cluster_bootstrap": fast_cluster_ci(tt, thr, 200, 20261103),
        "base_balanced_ci_cluster_bootstrap": fast_cluster_ci(bt, thr_base, 200, 20261103),
        "trained_tpr_wilson95": wilson(round(trained["tpr"] * trained["pos"]), trained["pos"]),
        "trained_fpr_wilson95": wilson(round(trained["fpr"] * trained["neg"]), trained["neg"]),
        "mcnemar_trained_vs_base_shared_threshold": mcnemar_exact([
            (x["score"] >= thr, y["score"] >= thr) for x, y in zip(tt, bt)]),
        "per_suite": {},
    }
    for suite in sorted({r.get("suite") for r in tt if r.get("suite")}):
        sub_t = [r for r in tt if r.get("suite") == suite]
        sub_b = [r for r in bt if r.get("suite") == suite]
        try:
            report["per_suite"][suite] = {"trained": evaluate_scores(sub_t, thr),
                                          "base": evaluate_scores(sub_b, thr)}
        except ValueError as exc:
            report["per_suite"][suite] = {"error": str(exc)}

    split_rows = [json.loads(l) for l in Path(a.test_split).open() if l.strip()]
    paired, pair_meta = build_pairs(split_rows, tt)
    report["quasi_matched_pairs"] = pair_meta
    if paired:
        report["trained_matched_deltas"] = matched_pair_deltas(paired, thr)
        paired_base = []
        by_index = {r["row_index"]: r for r in bt}
        for r in paired:
            q = dict(by_index[r["row_index"]]); q["pair_id"] = r["pair_id"]
            paired_base.append(q)
        report["base_matched_deltas"] = matched_pair_deltas(paired_base, thr_base)

    # Honesty probe: authorized rows with no untrusted tool output seen.
    # Firing here is a false alarm on a case with nothing citable.
    honesty_idx = [i for i, r in enumerate(split_rows)
                   if not r.get("state", {}).get("untrusted_tool_outputs_seen")
                   and i < len(tt) and not tt[i]["label"]]
    by_index_t = {r["row_index"]: r for r in tt}
    by_index_b = {r["row_index"]: r for r in bt}
    probe_t = [dict(by_index_t[i], honesty_probe=True) for i in honesty_idx if i in by_index_t]
    probe_b = [dict(by_index_b[i], honesty_probe=True) for i in honesty_idx if i in by_index_b]
    report["honesty_probe"] = {
        "definition": "authorized test rows with no untrusted tool output seen; firing is the defect",
        "n": len(honesty_idx),
        "trained": honesty_probe_rate(probe_t, thr) if probe_t else None,
        "base": honesty_probe_rate(probe_b, thr_base) if probe_b else None,
        "scored_separately": True, "excluded_from_headline": True}

    # Free-lunch gate. NOTE: "untrained" here is a PRETRAINED zero-shot base model,
    # not a randomly initialised head. A high AUC is expected; the required claim
    # is therefore a PAIRED improvement, not a bare AUC. Reported honestly, never hidden.
    fl = free_lunch_check(bd, td, 0.5)
    fl["baseline_kind"] = "pretrained_zero_shot_SecJev-4B (NOT an untrained head)"
    fl["interpretation"] = ("AUC >= 0.80 for the zero-shot base is expected here. "
                            "The CLEF free-lunch gate targets an untrained head; for SecJev "
                            "the valid evidence is the paired improvement above this baseline.")
    report["free_lunch_check"] = fl

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "threshold": thr, "trained_test_auc": trained["auc"],
        "base_test_auc_shared_threshold": base_shared["auc"],
        "auc_delta": report["auc_delta_trained_minus_base"],
        "trained_tpr": trained["tpr"], "trained_fpr": trained["fpr"],
        "mcnemar_p": report["mcnemar_trained_vs_base_shared_threshold"]["p_exact"],
        "paired_n": pair_meta["n_pairs"],
        "honesty_n": len(honesty_idx),
    }, indent=2))
    print("WROTE", out)


if __name__ == "__main__":
    main()
