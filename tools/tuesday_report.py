"""Assemble the Tuesday-v2 frozen report and run the fail-closed publication gate.

Inputs are PER-ROW score files, never aggregates. Every headline must survive:
  - an untrained-head / keyword baseline on identical rows,
  - a matched-pair discrimination delta,
  - Wilson + clustered-bootstrap confidence intervals,
  - a paired significance test,
  - a separately-scored honesty probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from matched_eval import (honesty_probe_rate, matched_pair_deltas, mcnemar_exact,
                          publication_gate, wilson)
from redo_eval import cluster_bootstrap_ci, evaluate_scores, calibrate_threshold


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(p: str) -> list[dict]:
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--untrained-dev", required=True)
    ap.add_argument("--trained-dev", required=True)
    ap.add_argument("--trained-test", required=True)
    ap.add_argument("--keyword-test", required=True)
    ap.add_argument("--paired", required=True, help="per-row scores carrying pair_id")
    ap.add_argument("--honesty", required=True)
    ap.add_argument("--checkpoint-manifest", default="")
    ap.add_argument("--training-log", default="")
    ap.add_argument("--dataset-manifest", default="")
    ap.add_argument("--replication-report", default="")
    ap.add_argument("--load-check", default="")
    ap.add_argument("--out", default="docs/redo-v2/tuesday_frozen_report.json")
    ap.add_argument("--chain-out", default="docs/redo-v2/frozen_chain_manifest.json")
    a = ap.parse_args()

    u_dev, t_dev, t_test = load(a.untrained_dev), load(a.trained_dev), load(a.trained_test)
    kw_test, paired, honesty = load(a.keyword_test), load(a.paired), load(a.honesty)

    thr = calibrate_threshold(t_dev, 0.01)  # dev-only, FPR budget 1%
    headline = evaluate_scores(t_test, thr)
    keyword = evaluate_scores(kw_test, thr)
    ci = cluster_bootstrap_ci(t_test, thr, n_resamples=200, seed=20261103)
    wilson_ci = wilson(round(headline["tpr"] * headline["pos"]), headline["pos"])

    pair_rows = [dict(r, honesty_probe=False) for r in paired]
    pair_report = matched_pair_deltas(pair_rows, thr)
    honesty_report = honesty_probe_rate(
        [dict(r, honesty_probe=True) for r in honesty], thr)
    mcnemar = mcnemar_exact([
        (float(r["score"]) >= thr, float(k["score"]) >= thr)
        for r, k in zip(t_test, kw_test)
        if r.get("label") is not None
    ])

    report = {
        "threshold_dev_calibrated": thr,
        "headline_test": headline,
        "keyword_baseline_test": keyword,
        "test_balanced_ci_cluster_bootstrap": ci,
        "tpr_wilson95": wilson_ci,
        "mcnemar_vs_keyword": mcnemar,
        "matched_pairs": pair_report,
        "honesty_probe": honesty_report,
        "free_lunch_gate": None,
        "publication_gate": None,
    }
    try:
        gate = publication_gate(u_dev, t_dev, pair_rows,
                                [dict(r, honesty_probe=True) for r in honesty], thr)
        report["publication_gate"] = gate
        report["free_lunch_gate"] = gate["free_lunch"]
        passed = True
    except ValueError as e:
        report["publication_gate"] = {"passed": False, "error": str(e)}
        report["free_lunch_gate"] = {"passed": False}
        passed = False

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"threshold": thr, "test_auc": headline["auc"],
                      "keyword_auc": keyword["auc"], "ci": ci,
                      "mean_delta": pair_report["mean_delta"],
                      "gate_passed": passed}, indent=2))
    print("WROTE", out)

    if passed:
        chain = {"artifacts": {}}
        for link, path in (("dataset_manifest", a.dataset_manifest),
                           ("training_log", a.training_log),
                           ("checkpoint_manifest", a.checkpoint_manifest),
                           ("frozen_score_file", a.trained_test),
                           ("analysis_output", str(out)),
                           ("replication_report", a.replication_report),
                           ("load_check_record", a.load_check)):
            p = Path(path) if path else None
            chain["artifacts"][link] = ({"path": str(p), "sha256": sha256(p)}
                                        if p and p.exists() else None)
        Path(a.chain_out).write_text(json.dumps(chain, indent=2) + "\n")
        print("WROTE CHAIN", a.chain_out)
    else:
        print("PUBLICATION GATE FAILED - chain manifest NOT written (fail closed)")


if __name__ == "__main__":
    main()
