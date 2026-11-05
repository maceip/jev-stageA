"""Benign expansion pack: tool outputs seen in no-attack AgentDojo episodes.

Addendum only — the frozen v0.1/v0.2 canonical files are not modified.
Rows already present (by normalized text) in either frozen Spot-1 build are
dropped so the pack contains only new negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from redo_v2 import convert_benign_tool_output, normalized_text_key
from score import decision_points

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external/redo_v2_out"
EPISODE_FILES = ["run3/benign.jsonl", "run3/mini/benign.jsonl"]
EXISTING = ["canonical_v2_train.jsonl", "canonical_v2_dev.jsonl", "canonical_v2_test.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT / "canonical_benign_agentdojo.jsonl"))
    a = ap.parse_args()
    existing: set[str] = set()
    for name in EXISTING:
        with (OUT / name).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing.add(normalized_text_key(str(json.loads(line)["text"])))
    stats: Counter = Counter()
    seen: set[str] = set()
    kept = 0
    with open(a.out, "w", encoding="utf-8") as out_f:
        for fname in EPISODE_FILES:
            with (ROOT / fname).open(encoding="utf-8") as f:
                for lineno, line in enumerate(f):
                    if not line.strip():
                        continue
                    ep = json.loads(line)
                    if ep.get("error") or ep.get("injection_task"):
                        stats["episodes_skipped"] += 1
                        continue
                    stats["episodes"] += 1
                    for _kind, i, _k, p in decision_points(ep["messages"]):
                        for j, seen_out in enumerate(p.get("seen") or []):
                            text = str(seen_out.get("output") or "")
                            if not text.strip():
                                continue
                            key = normalized_text_key(text)
                            if key in existing:
                                stats["already_in_frozen"] += 1
                                continue
                            if key in seen:
                                stats["duplicates_dropped"] += 1
                                continue
                            seen.add(key)
                            row = convert_benign_tool_output(
                                text,
                                source_id=f"{Path(fname).parent.name}-{Path(fname).stem}:{lineno}:{i}:{j}",
                                participant=f"{ep.get('suite')}:{ep.get('user_task')}",
                            )
                            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            kept += 1
    path = Path(a.out)
    print(json.dumps({"kept": kept, "stats": {k: v for k, v in sorted(stats.items())},
                      "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
