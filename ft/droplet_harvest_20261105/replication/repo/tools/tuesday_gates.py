"""Fail-closed publication gates and canonical secret resolution for Tuesday-v2.

A published number must map to the full chain:
  dataset manifest hash -> training log -> checkpoint hash -> frozen score file
  -> analysis output -> replication report.
Any missing or hash-mismatched link blocks publication.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
CHAIN_MANIFEST = ROOT / "docs/redo-v2/frozen_chain_manifest.json"
BYPASS_ENV = "TUESDAY_PUBLICATION_APPROVED"

REQUIRED_LINKS = (
    "dataset_manifest", "training_log", "checkpoint_manifest", "frozen_score_file",
    "analysis_output", "replication_report", "load_check_record",
)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_frozen_chain(strict: bool = True) -> dict:
    """Raise unless every link exists and hash-matches its manifest entry."""
    if not CHAIN_MANIFEST.exists():
        raise RuntimeError(
            "PUBLICATION_GATE: frozen chain manifest missing "
            f"({CHAIN_MANIFEST}). Complete G3-G7 before publishing.")
    chain = json.loads(CHAIN_MANIFEST.read_text())
    problems = []
    checked = {}
    for link in REQUIRED_LINKS:
        entry = chain.get("artifacts", {}).get(link)
        if not entry or not entry.get("path") or not entry.get("sha256"):
            problems.append(f"{link}: MISSING MANIFEST ENTRY")
            continue
        p = ROOT / entry["path"] if not os.path.isabs(entry["path"]) else pathlib.Path(entry["path"])
        if not p.exists():
            problems.append(f"{link}: FILE MISSING ({p})")
            continue
        actual = sha256(p)
        checked[link] = {"path": str(p), "sha256": actual[:16]}
        if actual != entry["sha256"]:
            problems.append(f"{link}: HASH MISMATCH")
    if strict and problems:
        raise RuntimeError("PUBLICATION_GATE_FAILED: " + " | ".join(problems))
    return {"passed": not problems, "problems": problems, "checked": checked}


def _read_env(path: pathlib.Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# Known misspellings in existing .env files. Values are never printed.
SECRET_ALIASES = {
    "TYPESAFE_API_KEY": ("TYPESAFE_API_KEY", "TPESAFE_API_KEY", "JEV_API_KEY"),
    "TYPESAFE_MODEL": ("TYPESAFE_MODEL",),
    "TEXT_MODEL_API_KEY": ("TEXT_MODEL_API_KEY",),
    "DIGITAL_OCEAN_API_KEY": ("DIGITAL_OCEAN_API_KEY",),
    "HF_TOKEN": ("HF_TOKEN",),
}

ENV_FILES = (
    ROOT / ".env",
    pathlib.Path("/Users/mac/labrea/agent-relay-surface/.env"),
    pathlib.Path("/Users/mac/jevalin/.env"),
    pathlib.Path("/Users/mac/jev-ultrafast/.env"),
)


def canonical_secret(name: str) -> str | None:
    """Resolve a secret from env or alias spellings. Never prints the value."""
    merged = dict(os.environ)
    for p in ENV_FILES:
        for k, v in _read_env(p).items():
            merged.setdefault(k, v)
    for candidate in SECRET_ALIASES.get(name, (name,)):
        v = merged.get(candidate)
        if v:
            return v
    return None


def secret_inventory() -> dict:
    """Names and presence only -- no values."""
    names = {}
    for p in ENV_FILES:
        names[str(p)] = sorted(_read_env(p).keys())
    resolved = {n: bool(canonical_secret(n)) for n in SECRET_ALIASES}
    return {"env_files": names, "resolved": resolved,
            "aliases": {k: list(v) for k, v in SECRET_ALIASES.items()}}


if __name__ == "__main__":
    print(json.dumps(secret_inventory(), indent=2))
    try:
        print("chain:", json.dumps(require_frozen_chain(strict=False), indent=2))
    except RuntimeError as e:
        print("chain:", str(e))
