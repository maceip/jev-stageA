"""Convert canonical Spot-1 v0.2 rows into ft_clef.py input format ({kind, text, label})."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external/redo_v2_out"


def main() -> None:
    summary = {}
    for split in ("train", "dev", "test"):
        src = OUT / f"canonical_v2_{split}.jsonl"
        dst = OUT / f"clef_spot1v2_{split}.jsonl"
        counts: Counter = Counter()
        with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                r = json.loads(line)
                fout.write(json.dumps({"kind": r["kind"], "text": r["text"],
                                       "label": bool(r["label"])}, ensure_ascii=False) + "\n")
                counts[(r["kind"], bool(r["label"]))] += 1
        summary[split] = {f"{k}:{v}": n for (k, v), n in sorted(counts.items())}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
