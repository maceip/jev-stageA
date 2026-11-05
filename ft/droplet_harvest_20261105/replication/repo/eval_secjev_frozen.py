"""Frozen SecJev evaluation over redo-v2 Spot-2 splits (Phase 6).

Two stages so expensive scoring happens once:

  score   --weights DIR --split ft_spot2v2_dev.jsonl --out scores_dev.jsonl
  analyze --dev scores_dev.jsonl --test scores_test.jsonl [--out report.json]

Score per row = P("2") + 0.5 * P("1") from the authorization question,
matching ft_action.py's training-time dev metric. `analyze` calibrates
the operating threshold on dev only (max FPR budget), then reports test
metrics via redo_eval plus a suite-clustered bootstrap CI. Test rows are
never used for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redo_eval import calibrate_threshold, cluster_bootstrap_ci, evaluate_scores

ROOT = Path(__file__).resolve().parent


def cmd_score(a: argparse.Namespace) -> None:
    from secjev_scorer import SecJev  # heavy import only when scoring

    scorer = SecJev(a.weights, device=a.device)
    rows = [json.loads(line) for line in open(a.split, encoding="utf-8") if line.strip()]
    written = failed = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            request = {"state": row["state"], "questions": row["questions"]}
            try:
                probs = scorer.score(request)["authorization"]
            except Exception as exc:  # oversized/strict failures are counted, never hidden
                failed += 1
                print(f"row {i} FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                continue
            meta = row.get("_meta") or {}
            f.write(json.dumps({
                "row_id": meta.get("row_id"),
                "score": float(probs.get("2", 0.0) + 0.5 * probs.get("1", 0.0)),
                "label": int(row["questions"]["authorization"]["label"]) != 0,
                "group": meta.get("suite") or meta.get("source") or "?",
                "suite": meta.get("suite"),
            }) + "\n")
            written += 1
            if a.progress and (i + 1) % a.progress == 0:
                print(f"scored {i + 1}/{len(rows)} (failed {failed})", flush=True)
    print(f"wrote {written} scores, {failed} failures -> {a.out}")


def load_scores(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def cmd_analyze(a: argparse.Namespace) -> None:
    dev, test = load_scores(a.dev), load_scores(a.test)
    thr = calibrate_threshold(dev, max_fpr=a.max_fpr)
    report = {
        "threshold_from_dev": thr, "max_fpr_budget": a.max_fpr,
        "dev": evaluate_scores(dev, thr),
        "test": evaluate_scores(test, thr),
        "test_at_fixed_0.5": evaluate_scores(test, 0.5),
        "test_balanced_acc_ci95": cluster_bootstrap_ci(test, thr),
        "test_by_suite": {},
    }
    for suite in sorted({r.get("suite") for r in test if r.get("suite")}):
        sub = [r for r in test if r.get("suite") == suite]
        try:
            report["test_by_suite"][suite] = evaluate_scores(sub, thr)
        except ValueError as exc:
            report["test_by_suite"][suite] = {"error": str(exc)}
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--weights", required=True)
    s.add_argument("--device", default=None)
    s.add_argument("--split", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--progress", type=int, default=250)
    s.set_defaults(fn=cmd_score)
    an = sub.add_parser("analyze")
    an.add_argument("--dev", required=True)
    an.add_argument("--test", required=True)
    an.add_argument("--max-fpr", type=float, default=0.01)
    an.add_argument("--out", default="")
    an.set_defaults(fn=cmd_analyze)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
