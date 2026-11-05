"""Redo-v2 evaluation harness over per-row score artifacts.

Phase-3 contracts from the CLEF review:
  * empty or single-class eval arms raise instead of reporting a number;
  * every reported metric recomputes from per-row scores, never aggregates;
  * score-cache keys bind checkpoint + code + data + config hashes;
  * thresholds are calibrated on dev only (calibrate_threshold);
  * trivial and keyword baselines ship beside any learned scorer.

CLI (keyword baseline over canonical JSONL):
  python redo_eval.py --dev canonical_dev.jsonl --test canonical_test.jsonl \
      --max_fpr 0.05 --out baseline_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


def _pairs(rows):
    pairs = [(float(r["score"]), bool(r["label"])) for r in rows]
    if not pairs:
        raise ValueError("empty eval arm: no scored rows")
    if len({p for _, p in pairs}) < 2:
        raise ValueError("single-class eval arm: metrics are undefined")
    return pairs


def _auc(pairs) -> float:
    pos = [s for s, p in pairs if p]
    neg = [s for s, p in pairs if not p]
    return sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg))


def evaluate_scores(rows, threshold: float) -> dict:
    pairs = _pairs(rows)
    pos = [s for s, p in pairs if p]
    neg = [s for s, p in pairs if not p]
    tpr = sum(s >= threshold for s in pos) / len(pos)
    fpr = sum(s >= threshold for s in neg) / len(neg)
    brier = sum((s - (1.0 if p else 0.0)) ** 2 for s, p in pairs) / len(pairs)
    bins: dict[int, list[tuple[float, bool]]] = {}
    for s, p in pairs:
        bins.setdefault(min(9, int(s * 10)), []).append((s, p))
    ece = 0.0
    for members in bins.values():
        conf = sum(s for s, _ in members) / len(members)
        acc = sum(1 for _, p in members if p) / len(members)
        ece += len(members) / len(pairs) * abs(conf - acc)
    return {
        "n": len(pairs), "pos": len(pos), "neg": len(neg), "threshold": threshold,
        "auc": round(_auc(pairs), 4), "tpr": round(tpr, 4), "fpr": round(fpr, 4),
        "balanced_accuracy": round(0.5 * (tpr + (1.0 - fpr)), 4),
        "brier": round(brier, 4), "ece": round(ece, 4),
    }


def calibrate_threshold(rows, max_fpr: float) -> float:
    pairs = _pairs(rows)
    best_thr, best_tpr = None, -1.0
    for thr in sorted({s for s, _ in pairs}):
        m = evaluate_scores(rows, thr)
        if m["fpr"] <= max_fpr and m["tpr"] > best_tpr:
            best_thr, best_tpr = thr, m["tpr"]
    if best_thr is None:
        return max(s for s, _ in pairs) + 1.0
    return best_thr


def cluster_bootstrap_ci(rows, threshold: float, n_resamples: int = 200, seed: int = 0):
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get("group", "?")), []).append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    values = []
    for _ in range(n_resamples):
        sample = [r for _ in keys for r in groups[rng.choice(keys)]]
        try:
            m = evaluate_scores(sample, threshold)
        except ValueError:
            continue
        values.append(0.5 * (m["tpr"] + (1.0 - m["fpr"])))
    if not values:
        raise ValueError("no valid bootstrap resamples")
    values.sort()
    lo = values[max(0, int(0.025 * len(values)) - 1)]
    hi = values[min(len(values) - 1, int(0.975 * len(values)))]
    return round(lo, 4), round(hi, 4)


def cache_key(checkpoint_sha: str, code_sha: str, data_sha: str, config: dict) -> str:
    payload = {"checkpoint": checkpoint_sha, "code": code_sha, "data": data_sha, "config": config}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


KEYWORDS = (
    "ignore previous", "ignore the previous", "disregard", "forget your instructions",
    "system prompt", "reveal the password", "reveal the secret", "send the password",
    "you are now", "new instructions", "override", "jailbreak", "do not tell the user",
    "send $", "send money", "exfiltrate", "access code", "secret key",
)


def keyword_score(text: str) -> float:
    low = text.lower()
    hits = sum(1 for k in KEYWORDS if k in low)
    return 1.0 - math.exp(-hits) if hits else 0.0


def _load_scored(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "score": keyword_score(str(r["text"])),
                "label": bool(r["label"]),
                "group": str((r.get("group_keys") or {}).get("participant", "?")),
                "kind": r.get("kind"), "source": r.get("source"),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--max_fpr", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev, test = _load_scored(Path(a.dev)), _load_scored(Path(a.test))
    report = {"scorer": "keyword_baseline", "max_fpr": a.max_fpr, "kinds": {}}
    for kind in sorted({r["kind"] for r in dev} | {r["kind"] for r in test}):
        d = [r for r in dev if r["kind"] == kind]
        t = [r for r in test if r["kind"] == kind]
        entry = {}
        if d and len({r["label"] for r in d}) == 2:
            thr = calibrate_threshold(d, a.max_fpr)
            entry["calibrated_threshold"] = thr
            entry["dev"] = evaluate_scores(d, thr)
            if t and len({r["label"] for r in t}) == 2:
                entry["test"] = evaluate_scores(t, thr)
                entry["test_balanced_ci"] = cluster_bootstrap_ci(t, thr, n_resamples=100, seed=11)
        entry["trivial_balanced_accuracy"] = 0.5
        entry["counts"] = {"dev": len(d), "test": len(t)}
        report["kinds"][kind] = entry
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
