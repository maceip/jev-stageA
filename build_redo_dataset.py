"""Build the frozen canonical redo-v2 dataset (Spot-1 + direct injection).

Two deterministic streaming passes over approved MIT sources:
  pass 1 collects lightweight keys and assigns connected-component splits;
  pass 2 regenerates rows in the identical order and writes split JSONL.

Covered tasks: tool_output_filter (LLMail labelled + benign FP emails) and
direct_prompt_injection (SaTML, Gandalf). Action-authorization (Spot-2) rows
are NOT built here yet: the legacy ft/ splits remain quarantined until the
label rule is rebuilt (see docs/redo-v2/phase0-claim-audit.md).

Inputs must already exist (see docs/redo-v2/source-registry.md):
  data/external/redo_v2_out/llmail_labelled_phase{1,2}.jsonl  (jq-converted)
  data/external/llmail-inject-challenge/data/emails_for_fp_tests.json
  data/external/ctf-satml24/chat.json
  data/external/gandalf_ignore_instructions/data/*.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from redo_v2 import (
    SOURCE_REGISTRY,
    assign_splits_from_keys,
    assign_splits_stratified,
    convert_benign_email,
    convert_gandalf,
    convert_llmail_labelled,
    convert_satml_chat,
    normalized_text_key,
)

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "data/external"
INTERMEDIATE = EXT / "redo_v2_out"


def _iter_source_rows(stats: Counter):
    # LLMail labelled-unique (gold attack_attempt labels; Unclear dropped).
    for phase in (1, 2):
        path = INTERMEDIATE / f"llmail_labelled_phase{phase}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                row = convert_llmail_labelled(rec["text"], rec, phase=phase)
                if row is None:
                    stats[f"llmail_phase{phase}_skipped_unclear"] += 1
                    continue
                stats[f"llmail_phase{phase}_converted"] += 1
                yield row
    # LLMail benign emails for false-positive testing.
    benign = json.loads((EXT / "llmail-inject-challenge/data/emails_for_fp_tests.json").read_text())
    for i, text in enumerate(benign):
        stats["benign_converted"] += 1
        yield convert_benign_email(str(text), source_id=f"fp-{i}")
    # SaTML direct-injection chats (attacker-side turns only). Convert per
    # record here: an exception inside a generator would silently end it.
    with (EXT / "ctf-satml24/chat.json").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = convert_satml_chat(json.loads(line))
            except Exception as e:
                stats["satml_errors"] += 1
                stats["satml_first_error"] = stats.get("satml_first_error") or f"{type(e).__name__}: {e}"
                continue
            stats["satml_converted"] += 1
            yield row
    # Gandalf direct-injection sample (all published splits; regrouped here).
    import pyarrow.parquet as pq

    for path in sorted((EXT / "gandalf_ignore_instructions/data").glob("*.parquet")):
        table = pq.read_table(path)
        for i, rec in enumerate(table.to_pylist()):
            stats["gandalf_converted"] += 1
            yield convert_gandalf(rec, source_id=f"{path.stem}-{i}")


def _safe(iterator, stats: Counter, name: str):
    while True:
        try:
            row = next(iterator)
        except StopIteration:
            return
        except Exception as e:
            stats[f"{name}_errors"] += 1
            stats[f"{name}_first_error"] = stats.get(f"{name}_first_error") or f"{type(e).__name__}: {e}"
            continue
        stats[f"{name}_converted"] += 1
        yield row


def _deduped(stats: Counter):
    seen: set[str] = set()
    for row in _iter_source_rows(stats):
        key = normalized_text_key(row["text"])
        if key in seen:
            stats["duplicates_dropped"] += 1
            continue
        seen.add(key)
        yield row, key


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(INTERMEDIATE))
    ap.add_argument("--manifest", default=str(ROOT / "docs/redo-v2/frozen-manifest.json"))
    ap.add_argument("--prefix", default="canonical")
    ap.add_argument("--dataset-name", default="redo-v2-canonical-spot1-direct-v0.1")
    ap.add_argument("--stratified", action="store_true",
                    help="balance components per (kind,label) instead of hash-only assignment")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    keys = [
        {"row_id": row["row_id"], "source": row["source"], "kind": row["kind"],
         "label": row["label"], "group_keys": row["group_keys"], "text_key": key}
        for row, key in _deduped(stats)
    ]
    homes = assign_splits_stratified(keys) if a.stratified else assign_splits_from_keys(keys)
    del keys

    counts: Counter = Counter()
    handles = {s: (out / f"{a.prefix}_{s}.jsonl").open("w", encoding="utf-8")
               for s in ("train", "dev", "test")}
    try:
        for row, _key in _deduped(Counter()):
            split = homes[row["row_id"]]
            row["split"] = split
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[f"{split}:{row['source']}:{row['label']}"] += 1
            counts[f"{split}:total"] += 1
    finally:
        for h in handles.values():
            h.close()

    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_head = None
    manifest = {
        "dataset": a.dataset_name,
        "split_assignment": "stratified" if a.stratified else "hash",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_build": git_head,
        "tasks_covered": ["tool_output_filter", "direct_prompt_injection"],
        "tasks_not_covered": ["action_authorization_guard"],
        "sources": [s for s in SOURCE_REGISTRY if s["name"] in {
            "llmail-inject-challenge", "ctf-satml24", "gandalf_ignore_instructions"}],
        "conversion_stats": {k: v for k, v in sorted(stats.items())},
        "split_counts": {k: v for k, v in sorted(counts.items())},
        "files": {
            s: {"path": str((out / f"{a.prefix}_{s}.jsonl").relative_to(ROOT)),
                "bytes": (out / f"{a.prefix}_{s}.jsonl").stat().st_size,
                "sha256": _sha256(out / f"{a.prefix}_{s}.jsonl")}
            for s in ("train", "dev", "test")
        },
        "warnings": (["Stratified mode splits direct_prompt_injection rows at chat level (identical texts stay together); SaTML participants can appear in multiple splits for that kind. Participant-clustered bootstrap CIs are required for direct-kind metrics."] if a.stratified else []) + [
            "LLMail labelled rows carry no team id; participant is a per-text pseudo-group (participant_known=false).",
            "LLMail Unclear labels were dropped, not relabelled.",
            "SaTML/Gandalf rows are direct-injection only; no action-authorization labels were derived.",
            "Spot-2 action-authorization training data is not built yet; legacy ft/ splits remain quarantined.",
        ],
    }
    Path(a.manifest).write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"split_counts": manifest["split_counts"],
                      "conversion_stats": manifest["conversion_stats"]}, indent=2))


if __name__ == "__main__":
    main()
