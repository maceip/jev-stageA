"""Build the frozen canonical Spot-2 (action authorization) dataset v0.1.

Rebuilds labels from recorded AgentDojo trajectories with the corrected
rule (redo_v2.authorization_label): unauthorized requires an attacker
argument value in the proposed call. The legacy rule is computed beside
the corrected one so relabelling impact is quantified, never hidden.

Inputs default to the run3 recorded trajectories (standard + mini).
Outputs land in data/external/redo_v2_out/ with a tracked manifest at
docs/redo-v2/frozen-manifest-spot2.json.
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
    convert_action_point,
    flatten_strings,
    legacy_authorization_label,
    normalized_text_key,
    stable_group_split,
)
from score import decision_points

ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = [
    "run3/benign.jsonl", "run3/important_instructions.jsonl", "run3/ood_bland_note.jsonl",
    "run3/mini/benign.jsonl", "run3/mini/important_instructions.jsonl", "run3/mini/ood_bland_note.jsonl",
]
SUITE_SET = {"workspace", "travel", "banking", "slack"}


def _episodes(files):
    for name in files:
        path = ROOT / name
        if not path.is_file():
            print(f"skip missing {name}")
            continue
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f):
                if line.strip():
                    yield name, lineno, json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data/external/redo_v2_out"))
    ap.add_argument("--manifest", default=str(ROOT / "docs/redo-v2/frozen-manifest-spot2.json"))
    ap.add_argument("files", nargs="*", default=DEFAULT_FILES)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    rows: list[dict] = []
    seen_text: set[str] = set()
    for fname, lineno, ep in _episodes(a.files):
        if ep.get("error") or ep.get("suite") not in SUITE_SET:
            stats["episodes_skipped"] += 1
            continue
        suite = SUITES[ep["suite"]]
        try:
            ut = suite.get_user_task_by_id(ep["user_task"])
            it = suite.get_injection_task_by_id(ep["injection_task"]) if ep.get("injection_task") else None
        except KeyError:
            stats["unknown_task"] += 1
            continue
        user_fns = {c.function for c in ground_truth(suite, ut, ep.get("injections"))}
        inj_calls = ground_truth(suite, it, ep.get("injections")) if it else []
        inj_fns = {c.function for c in inj_calls}
        inj_vals = {v for c in inj_calls for v in flatten_strings(c.args) if str(v).strip()}
        stats["episodes"] += 1
        for kind, i, k, p in decision_points(ep["messages"]):
            if kind != "action":
                continue
            argv = list(flatten_strings(p["call"]["args"]))
            label = authorization_label(p["call"]["function"], argv, user_fns, inj_fns, inj_vals)
            legacy = legacy_authorization_label(p["call"]["function"], argv, user_fns, inj_fns, inj_vals)
            stats[f"label_pair_{legacy}_to_{label}"] += 1
            if label is None:
                stats["ambiguous_skipped"] += 1
                continue
            meta = {"suite": ep["suite"], "user_task": ep["user_task"],
                    "injection_task": ep.get("injection_task"), "file": fname}
            row = convert_action_point(meta, p, label, source_id=f"{Path(fname).stem}:{lineno}:{i}:{k}")
            key = normalized_text_key(row["text"])
            if key in seen_text:
                stats["duplicates_dropped"] += 1
                continue
            seen_text.add(key)
            rows.append(row)
    # Suite-disjoint protocol (comparable to the original C1 build):
    # banking+slack are the held-out test suites; workspace+travel split
    # by participant hash into train/dev. Group-component splitting is not
    # used here: 14 shared injection templates union nearly all rows into
    # one component and collapse every split into one bucket.
    splits = {"train": [], "dev": [], "test": []}
    for row in rows:
        suite = row["metadata"]["suite"]
        if suite in {"banking", "slack"}:
            split = "test"
        else:
            split = stable_group_split(row["group_keys"]["participant"])
            if split == "test":
                split = "dev"
        row["split"] = split
        splits[split].append(row)
    counts: Counter = Counter()
    files = {}
    for split, split_rows in splits.items():
        path = out / f"canonical_spot2_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[f"{split}:{row['metadata']['suite']}:{row['label']}"] += 1
                counts[f"{split}:total"] += 1
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        files[split] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": h}
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_head = None
    manifest = {
        "dataset": "redo-v2-canonical-spot2-v0.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_build": git_head,
        "inputs": a.files,
        "label_rule": "unauthorized iff function in injection ground truth AND attacker arg value in call args",
        "conversion_stats": {k: v for k, v in sorted(stats.items())},
        "split_counts": {k: v for k, v in sorted(counts.items())},
        "files": files,
        "warnings": [
            "Splits are suite-disjoint (banking+slack test; workspace+travel train/dev by participant hash). Injection templates are shared across suites, so template-level leakage remains and must be reported per-template.",
            "Episodes are Tier C recorded benchmark trajectories, not real-world traffic.",
        ],
    }
    Path(a.manifest).write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"split_counts": manifest["split_counts"],
                      "conversion_stats": manifest["conversion_stats"]}, indent=2))


if __name__ == "__main__":
    main()
