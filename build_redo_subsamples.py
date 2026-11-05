"""Predeclared compute-feasible subsamples for SecJev redo training (Phase 4).

Measured MPS throughput (~614 padded tok/s bf16, backbone-bound) makes
full-data LoRA training a 10-16 h run. Fixed protocol, declared before
any model result exists:

- train_sub: ALL 430 unauthorized positives + 1,290 negatives (1:3),
  negatives sampled per-suite proportionally, seed 20261103.
- dev_sub (training-time signal only): ALL 85 dev positives + 427
  negatives, same sampling rule. Final epoch selection re-scores the
  FULL dev split with eval_secjev_frozen.py before test is touched.
Row-id manifests are written next to the outputs.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external/redo_v2_out"
SEED = 20261103


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (OUT / name).open(encoding="utf-8") if line.strip()]


def is_pos(r: dict) -> bool:
    return int(r["questions"]["authorization"]["label"]) != 0


def suite_of(r: dict) -> str:
    return str((r.get("_meta") or {}).get("suite") or "?")


def sample_negatives(negatives: list[dict], k: int, rng: random.Random) -> list[dict]:
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for r in negatives:
        by_suite[suite_of(r)].append(r)
    total = len(negatives)
    picked: list[dict] = []
    for suite in sorted(by_suite):
        pool = by_suite[suite][:]
        rng.shuffle(pool)
        take = min(len(pool), round(k * len(pool) / total))
        picked.extend(pool[:take])
    if len(picked) < k:  # rounding remainder from largest pools
        rest = [r for r in negatives if r not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: k - len(picked)])
    return picked[:k]


def build(split: str, n_neg: int, out_name: str) -> dict:
    rng = random.Random(SEED)
    rows = load(f"ft_spot2v2_{split}.jsonl")
    pos = [r for r in rows if is_pos(r)]
    neg = sample_negatives([r for r in rows if not is_pos(r)], n_neg, rng)
    picked = pos + neg
    rng.shuffle(picked)
    dst = OUT / out_name
    with dst.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "source": f"ft_spot2v2_{split}.jsonl", "seed": SEED,
        "positives": len(pos), "negatives": len(neg), "rows": len(picked),
        "suites": dict(sorted(Counter(suite_of(r) for r in picked).items())),
        "row_ids": [str((r.get("_meta") or {}).get("row_id")) for r in picked],
    }
    (OUT / f"{out_name}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return {k: v for k, v in manifest.items() if k != "row_ids"}


def main() -> None:
    print(json.dumps({
        "train_sub": build("train", 1290, "ft_spot2v2_train_sub.jsonl"),
        "dev_sub": build("dev", 427, "ft_spot2v2_dev_sub.jsonl"),
    }, indent=2))


if __name__ == "__main__":
    main()
