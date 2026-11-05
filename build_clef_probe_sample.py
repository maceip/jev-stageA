"""Fixed 50-row stratified sample for the Phase 5 CLEF encode probe.

Per docs/redo-v2/phase5-clef-plan.md: seed 20261103, stratified by
kind x label from clef_spot1v2_train.jsonl. Built ahead of GPU
availability so the probe is a single timing run later.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external/redo_v2_out"
SEED = 20261103
N = 50


def main() -> None:
    rows = [json.loads(line) for line in (OUT / "clef_spot1v2_train.jsonl").open(encoding="utf-8") if line.strip()]
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["kind"], bool(r["label"]))].append(r)
    rng = random.Random(SEED)
    picked: list[dict] = []
    keys = sorted(strata, key=str)
    base, rem = divmod(N, len(keys))
    for i, key in enumerate(keys):
        pool = strata[key][:]
        rng.shuffle(pool)
        picked.extend(pool[: base + (1 if i < rem else 0)])
    rng.shuffle(picked)
    dst = OUT / "clef_encode_probe_50.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {f"{k[0]}:{k[1]}": min(len(strata[k]), base + (1 if i < rem else 0)) for i, k in enumerate(keys)}
    print(json.dumps({"rows": len(picked), "strata": counts, "file": str(dst.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
