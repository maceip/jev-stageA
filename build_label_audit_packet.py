"""Blind human label-audit packet over the frozen redo-v2 datasets.

Writes a blind sample (no labels) plus a separate key file under
data/external/redo_v2_out/ (gitignored; corpus text must not enter docs).
Stratified by file x kind x label with a fixed seed. Human adjudication
fills the verdict fields; agreement is computed against the key afterwards.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external/redo_v2_out"
FILES = ["canonical_v2_train.jsonl", "canonical_v2_dev.jsonl", "canonical_v2_test.jsonl",
         "canonical_spot2v2_train.jsonl", "canonical_spot2v2_dev.jsonl", "canonical_spot2v2_test.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-stratum", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for name in FILES:
        with (OUT / name).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    strata[(name, r["kind"], bool(r["label"]))].append(r)
    blind, key = [], []
    for stratum in sorted(strata, key=str):
        rows = strata[stratum]
        rng.shuffle(rows)
        for r in rows[: a.per_stratum]:
            blind.append({
                "row_id": r["row_id"], "kind": r["kind"], "source": r["source"],
                "text": str(r["text"])[:1500],
                "verdict_label": None, "verdict_confidence": None, "verdict_notes": "",
            })
            key.append({"row_id": r["row_id"], "stratum": list(stratum), "label": bool(r["label"])})
    rng.shuffle(blind)
    with (OUT / "label_audit_blind.jsonl").open("w", encoding="utf-8") as f:
        for b in blind:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    (OUT / "label_audit_key.json").write_text(json.dumps(
        {"seed": a.seed, "per_stratum": a.per_stratum, "rows": key}, indent=2))
    counts = Counter(f"{k[1]}:{k[2]}" for k in
                     [(tuple(entry["stratum"])) for entry in key])
    print(json.dumps({"blind_rows": len(blind), "strata_rows": dict(sorted(counts.items()))}, indent=2))


if __name__ == "__main__":
    main()
