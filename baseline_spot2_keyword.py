"""Trivial keyword baseline for Spot-2 redo splits (Phase 6 baseline arm).

Score = redo_eval.keyword_score over the proposed tool call arguments
plus any untrusted output text present in state. CPU-only; runs before
any model result exists and is never used for selection — Phase 6
reports it next to model metrics on identical rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from redo_eval import evaluate_scores, keyword_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external/redo_v2_out"
REPORT = ROOT / "docs/redo-v2/baseline-keyword-spot2.json"


def row_text(state: dict) -> str:
    parts = [json.dumps(state.get("proposed_tool_call") or {}, ensure_ascii=False)]
    for item in state.get("untrusted_tool_outputs_seen") or []:
        if isinstance(item, dict) and isinstance(item.get("output"), str):
            parts.append(item["output"])
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


def main() -> None:
    report = {}
    for split in ("dev", "test"):
        rows = []
        with (OUT / f"ft_spot2v2_{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                rows.append({
                    "score": keyword_score(row_text(r["state"])),
                    "label": int(r["questions"]["authorization"]["label"]) != 0,
                    "group": str((r.get("_meta") or {}).get("suite") or "?"),
                })
        report[split] = {
            "at_0.5": evaluate_scores(rows, 0.5),
            "rows": len(rows),
        }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
