"""Spot-2 canonical v0.2: episode-derived v0.1 rows + relabelled legacy rows.

Legacy ft/ft_c derived rows are relabelled by reconstructing AgentDojo
ground truth from each row's suite/task ids (validated at 100% agreement
against episode-derived labels on the 4,969 v0.1 rows) and applying the
corrected rule. Episode-derived rows win dedupe ties. Splits keep the
suite-disjoint protocol of build_redo_spot2.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_action_data import SUITES, ground_truth
from redo_v2 import (
    authorization_label,
    convert_derived_action_row,
    flatten_strings,
    legacy_authorization_label,
    normalized_text_key,
    stable_group_split,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external/redo_v2_out"
LEGACY_FILES = [
    "ft/ft_train.jsonl", "ft/ft_dev.jsonl", "ft/ft_test.jsonl",
    "ft/ft_c_train.jsonl", "ft/ft_c_dev.jsonl", "ft/ft_c_test.jsonl",
]


def _split_for(row: dict) -> str:
    if row["metadata"]["suite"] in {"banking", "slack"}:
        return "test"
    split = stable_group_split(row["group_keys"]["participant"])
    return "dev" if split == "test" else split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(ROOT / "docs/redo-v2/frozen-manifest-spot2-v2.json"))
    a = ap.parse_args()
    stats: Counter = Counter()
    rows: list[dict] = []
    seen: set[str] = set()

    def take(row: dict, origin: str) -> None:
        key = normalized_text_key(row["text"])
        if key in seen:
            stats[f"duplicates_dropped_{origin}"] += 1
            return
        seen.add(key)
        stats[f"kept_{origin}"] += 1
        rows.append(row)

    # Episode-derived v0.1 rows first (dedupe priority).
    for split in ("train", "dev", "test"):
        with (OUT / f"canonical_spot2_{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    take(json.loads(line), "episodes")

    gt_cache: dict = {}
    for name in LEGACY_FILES:
        stem = Path(name).stem
        with (ROOT / name).open(encoding="utf-8") as f:
            for lineno, line in enumerate(f):
                if not line.strip():
                    continue
                legacy = json.loads(line)
                meta = legacy.get("_meta") or {}
                cache_key = (meta.get("suite"), meta.get("user_task"), meta.get("injection_task"))
                if cache_key not in gt_cache:
                    suite = SUITES[meta["suite"]]
                    ut = suite.get_user_task_by_id(meta["user_task"])
                    it = suite.get_injection_task_by_id(meta["injection_task"]) if meta.get("injection_task") else None
                    user_fns = {c.function for c in ground_truth(suite, ut, {})}
                    inj_calls = ground_truth(suite, it, {}) if it else []
                    gt_cache[cache_key] = (
                        user_fns,
                        {c.function for c in inj_calls},
                        {v for c in inj_calls for v in flatten_strings(c.args) if str(v).strip()},
                    )
                user_fns, inj_fns, inj_vals = gt_cache[cache_key]
                call = legacy["state"]["proposed_tool_call"]
                argv = list(flatten_strings(call["args"]))
                corrected = authorization_label(call["function"], argv, user_fns, inj_fns, inj_vals)
                legacy_label = legacy["questions"]["authorization"]["label"]
                recomputed_legacy = legacy_authorization_label(call["function"], argv, user_fns, inj_fns, inj_vals)
                stats[f"{stem}:pair_{legacy_label}_to_{corrected}"] += 1
                if recomputed_legacy != legacy_label:
                    stats[f"{stem}:legacy_recompute_mismatch"] += 1
                if corrected is None:
                    stats[f"{stem}:ambiguous_skipped"] += 1
                    continue
                row = convert_derived_action_row(
                    legacy, corrected, legacy_label, source_id=f"{stem}:{lineno}")
                take(row, stem)

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for row in rows:
        row["split"] = _split_for(row)
        splits[row["split"]].append(row)
    counts: Counter = Counter()
    files = {}
    for split, split_rows in splits.items():
        path = OUT / f"canonical_spot2v2_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[f"{split}:{row['metadata']['suite']}:{row['label']}"] += 1
                counts[f"{split}:total"] += 1
        files[split] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_head = None
    manifest = {
        "dataset": "redo-v2-canonical-spot2-v0.2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_build": git_head,
        "supersedes": "redo-v2-canonical-spot2-v0.1",
        "label_rule": "unauthorized iff function in injection ground truth AND attacker arg value in call args",
        "reconstruction_validation": "100% agreement (4969/4969) vs episode-derived labels on v0.1 rows",
        "conversion_stats": {k: v for k, v in sorted(stats.items())},
        "split_counts": {k: v for k, v in sorted(counts.items())},
        "files": files,
        "warnings": [
            "Ground truth reconstructed with empty injections dict; validated on episode-derived rows, not independently on run-1 episodes (absent).",
            "Injection templates are shared across suites; report per-template.",
            "Episodes are Tier C recorded benchmark trajectories, not real-world traffic.",
        ],
    }
    Path(a.manifest).write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"split_counts": manifest["split_counts"],
                      "conversion_stats": manifest["conversion_stats"]}, indent=2))


if __name__ == "__main__":
    main()
