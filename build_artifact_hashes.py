"""SHA-256 inventory of redo-v2 artifacts for the Phase 7 claim chain.

Covers frozen manifests, canonical/fine-tune input files, subsample
manifests, baseline reports, and audit outputs. Missing files are
reported as missing, never skipped silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external/redo_v2_out"
DOCS = ROOT / "docs/redo-v2"

FILES = [
    DOCS / "frozen-manifest.json", DOCS / "frozen-manifest-v2.json",
    DOCS / "frozen-manifest-spot2.json", DOCS / "frozen-manifest-spot2-v2.json",
    DOCS / "dataset-audit.json", DOCS / "baseline-keyword-report-v2.json",
    DOCS / "baseline-keyword-spot2.json",
    OUT / "canonical_v2_train.jsonl", OUT / "canonical_v2_dev.jsonl", OUT / "canonical_v2_test.jsonl",
    OUT / "canonical_spot2v2_train.jsonl", OUT / "canonical_spot2v2_dev.jsonl", OUT / "canonical_spot2v2_test.jsonl",
    OUT / "ft_spot2v2_train.jsonl", OUT / "ft_spot2v2_dev.jsonl", OUT / "ft_spot2v2_test.jsonl",
    OUT / "ft_spot2v2_train_sub.jsonl", OUT / "ft_spot2v2_train_sub.jsonl.manifest.json",
    OUT / "ft_spot2v2_dev_sub.jsonl", OUT / "ft_spot2v2_dev_sub.jsonl.manifest.json",
    OUT / "clef_spot1v2_train.jsonl", OUT / "clef_spot1v2_dev.jsonl", OUT / "clef_spot1v2_test.jsonl",
    OUT / "clef_encode_probe_50.jsonl", OUT / "canonical_benign_agentdojo.jsonl",
    OUT / "ft_spot2v2_test_adaptive_zerowidth.jsonl", OUT / "ft_spot2v2_test_adaptive_homoglyph.jsonl",
    OUT / "ft_spot2v2_test_adaptive_quote.jsonl", OUT / "ft_spot2v2_test_adaptive_case.jsonl",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    inventory = {}
    for path in FILES:
        rel = str(path.relative_to(ROOT))
        if path.is_file():
            inventory[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        else:
            inventory[rel] = {"missing": True}
    dest = DOCS / "artifact-hashes.json"
    dest.write_text(json.dumps(inventory, indent=2) + "\n")
    missing = [k for k, v in inventory.items() if v.get("missing")]
    print(json.dumps({"hashed": len(inventory) - len(missing), "missing": missing,
                      "file": str(dest.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
