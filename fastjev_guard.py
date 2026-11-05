"""FASTJEV guard: local risk gate for jev-ultrafast decisions.

browser-use/jev-ultrafast calls the hosted TypeSafe Jev model
(api.typesafe.ai/v1/systemone, model "jev-latest"); no local weights or
hidden states exist to fine-tune. The locally trainable piece is this
guard: a calibrated allow/block decision over a risk score for each
ultrafast operation/target choice, scored by an injected local scorer
(e.g. the SecJev/CLEF action monitor in secjev_scorer.py).

Contracts (fixes from the CLEF fine-tune review):
  - Thresholds are calibrated on dev scores only, never on test.
  - The guard fails closed: a missing, non-finite, or erroring scorer
    blocks the action instead of allowing it.
  - Held-out metrics are stamped with the best-checkpoint epoch/step
    that produced them, never with the requested epoch count.
  - Per-row scores are the calibration input, so every reported metric
    can be recomputed from a JSONL artifact.

Calibration CLI:
  python fastjev_guard.py --dev_scores dev_scores.jsonl \
      --test_scores test_scores.jsonl --max_fpr 0.05 --out report.json
Score rows: {"kind": str, "score": float, "is_pos": bool}
(positive = unauthorized/injected = should be blocked).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path

DEPLOYED_THRESHOLDS = {"tool_output_filter": 0.99, "action_authorization_guard": 0.35}


def _auc(pairs: list[tuple[float, bool]]) -> float:
    pos = [s for s, p in pairs if p]
    neg = [s for s, p in pairs if not p]
    if not pos or not neg:
        return 0.0
    return sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg))


def metrics_at(scores: Iterable[tuple[float, bool]], threshold: float) -> dict:
    pairs = [(float(s), bool(p)) for s, p in scores]
    pos = [s for s, p in pairs if p]
    neg = [s for s, p in pairs if not p]
    tpr = sum(s >= threshold for s in pos) / len(pos) if pos else 0.0
    fpr = sum(s >= threshold for s in neg) / len(neg) if neg else 0.0
    return {
        "n": len(pairs), "pos": len(pos), "neg": len(neg),
        "threshold": threshold,
        "tpr": round(tpr, 4), "fpr": round(fpr, 4),
        "balanced_accuracy": round(0.5 * (tpr + (1.0 - fpr)), 4),
        "auc": round(_auc(pairs), 4),
    }


def calibrate_threshold(scores: Iterable[tuple[float, bool]], max_fpr: float) -> float:
    """Smallest-risk threshold maximizing TPR subject to dev FPR <= max_fpr."""
    pairs = [(float(s), bool(p)) for s, p in scores]
    if not any(p for _, p in pairs) or all(p for _, p in pairs):
        raise ValueError("calibration needs both positive and negative scores")
    best_thr, best_tpr = None, -1.0
    for thr in sorted({s for s, _ in pairs}):
        m = metrics_at(pairs, thr)
        if m["fpr"] <= max_fpr and m["tpr"] > best_tpr:
            best_thr, best_tpr = thr, m["tpr"]
    if best_thr is None:  # every observed score exceeds the FPR budget
        return max(s for s, _ in pairs) + 1.0
    return best_thr


def guard_decision(score: float | None, threshold: float) -> dict:
    """Fail closed: no usable score means block."""
    if score is None or not math.isfinite(score):
        return {"allow": False, "reason": "scorer_unavailable", "score": None, "threshold": threshold}
    allow = score < threshold
    return {
        "allow": allow,
        "reason": "below_threshold" if allow else "risk_at_or_above_threshold",
        "score": score,
        "threshold": threshold,
    }


def build_guard_text(state: dict, goal: str, decision: dict) -> str:
    chosen = next((a for a in state.get("actions", []) if a.get("id") == decision.get("choice")), {})
    lines = [
        f"goal: {goal}",
        f"operation: {decision.get('operation')} target: {decision.get('target')}",
        f"element: {chosen.get('label', '')} role: {chosen.get('role', '')} kind: {chosen.get('kind', '')}",
        f"page: {state.get('url', '')} title: {state.get('title', '')}",
        f"operation_probabilities: {json.dumps(decision.get('operation_probabilities', {}), sort_keys=True)}",
        f"page_text: {str(state.get('text', ''))[:2000]}",
    ]
    return "\n".join(lines)


def score_ultrafast_decision(
    state: dict, goal: str, decision: dict, scorer_fn: Callable[[str], float], threshold: float
) -> dict:
    """Score one jev-ultrafast choose() decision; any scorer failure blocks."""
    try:
        score = float(scorer_fn(build_guard_text(state, goal, decision)))
    except Exception:
        return guard_decision(None, threshold)
    return guard_decision(score, threshold)


def resolve_eval_stamp(best_epoch: int | None, best_step: int | None,
                       requested_epochs: int, final_step: int) -> dict:
    """Stamp held-out metrics with the evaluated (best) checkpoint identity."""
    if best_epoch is None or best_step is None:
        raise ValueError(
            f"no best checkpoint recorded (requested_epochs={requested_epochs}, final_step={final_step}); "
            "refusing to stamp metrics with the requested epoch count"
        )
    return {"epoch": int(best_epoch), "step": int(best_step)}


def _load_rows(path: Path) -> dict[str, list[tuple[float, bool]]]:
    by_kind: dict[str, list[tuple[float, bool]]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_kind.setdefault(r["kind"], []).append((float(r["score"]), bool(r["is_pos"])))
    return by_kind


def calibration_report(dev: dict, test: dict, max_fpr: float) -> dict:
    report = {"max_fpr": max_fpr, "kinds": {}}
    for kind in sorted(set(dev) | set(test)):
        entry = {}
        if kind in dev:
            thr = calibrate_threshold(dev[kind], max_fpr)
            entry["calibrated_threshold"] = thr
            entry["dev_at_calibrated"] = metrics_at(dev[kind], thr)
            if kind in test:
                entry["test_at_calibrated"] = metrics_at(test[kind], thr)
        if kind in test:
            deployed = DEPLOYED_THRESHOLDS.get(kind)
            if deployed is not None:
                entry["test_at_deployed"] = metrics_at(test[kind], deployed)
            entry["trivial_balanced_accuracy"] = 0.5
        report["kinds"][kind] = entry
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Dev-calibrated FASTJEV guard report from per-row scores.")
    ap.add_argument("--dev_scores", required=True)
    ap.add_argument("--test_scores", required=True)
    ap.add_argument("--max_fpr", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = calibration_report(_load_rows(Path(a.dev_scores)), _load_rows(Path(a.test_scores)), a.max_fpr)
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
