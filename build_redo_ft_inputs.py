"""Convert canonical Spot-2 v0.2 rows into ft_action.py input format.

Output rows mirror the legacy ft format exactly:
  {"state": ..., "questions": {"authorization": {..., "label": 0|2}}, "_meta": ...}
The question template (instructions/criteria) is copied from the first
legacy ft row so the training contract cannot drift silently.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external/redo_v2_out"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template-from", default=str(ROOT / "ft/ft_c_train.jsonl"))
    a = ap.parse_args()
    with open(a.template_from, encoding="utf-8") as f:
        template = copy.deepcopy(json.loads(f.readline())["questions"]["authorization"])
    template.pop("label", None)
    summary = {}
    for split in ("train", "dev", "test"):
        src = OUT / f"canonical_spot2v2_{split}.jsonl"
        dst = OUT / f"ft_spot2v2_{split}.jsonl"
        n = pos = 0
        with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                r = json.loads(line)
                label = int(r["metadata"]["label_raw"])
                q = dict(template)
                q["label"] = label
                fout.write(json.dumps({
                    "state": json.loads(r["text"]),
                    "questions": {"authorization": q},
                    "_meta": {**r["metadata"], "row_id": r["row_id"]},
                }, ensure_ascii=False) + "\n")
                n += 1
                pos += label != 0
        summary[split] = {"rows": n, "unauthorized": pos, "file": str(dst.relative_to(ROOT))}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
