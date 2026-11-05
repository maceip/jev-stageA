"""Programmatic quality audit over frozen redo-v2 canonical datasets.

Counts only: matched text is never written to the report. The audit covers
label consistency (same normalized text with conflicting labels), script
distribution (a language proxy, not language ID), text-length samples, and
PII/secret pattern counts (emails, Luhn-valid card numbers, API-key shapes,
password assignments, URLs).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from redo_v2 import normalized_text_key

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
KEY_RE = re.compile(r"\b(?:sk-[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]{12,}|hf_[A-Za-z0-9]{12,}|AKIA[A-Z0-9]{12,})\b")
PASSWORD_RE = re.compile(r"(?i)\b(?:password|passwd|api[_-]?key|secret)\s*[:=]\s*\S+")
URL_RE = re.compile(r"https?://[^\s\"']+")


def luhn_ok(digits: str) -> bool:
    digits = re.sub(r"\D", "", digits)
    if not 13 <= len(digits) <= 19:
        return False
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def scan_text(text: str) -> dict[str, int]:
    hits: Counter = Counter()
    if EMAIL_RE.search(text):
        hits["email"] += len(EMAIL_RE.findall(text))
    cards = [c for c in CARD_RE.findall(text) if luhn_ok(c)]
    if cards:
        hits["card_luhn"] += len(cards)
    if KEY_RE.search(text):
        hits["api_key_like"] += len(KEY_RE.findall(text))
    if PASSWORD_RE.search(text):
        hits["password_assignment"] += len(PASSWORD_RE.findall(text))
    if URL_RE.search(text):
        hits["url"] += len(URL_RE.findall(text))
    return dict(hits)


def script_bucket(text: str) -> str:
    counts: Counter = Counter()
    for ch in text:
        o = ord(ch)
        if ch.isalpha() and o < 0x250:
            counts["latin"] += 1
        elif 0x400 <= o <= 0x4FF:
            counts["cyrillic"] += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF:
            counts["cjk"] += 1
        elif 0x600 <= o <= 0x6FF:
            counts["arabic"] += 1
        elif ch.isalpha():
            counts["other"] += 1
    if not counts:
        return "no_alpha"
    return counts.most_common(1)[0][0]


def audit_file(path: Path, sample_every: int = 37) -> dict:
    counts: Counter = Counter()
    scripts: Counter = Counter()
    patterns: Counter = Counter()
    pattern_rows: Counter = Counter()
    lengths: list[int] = []
    label_by_text: dict[str, bool] = {}
    conflicts = 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            counts[f"{r['source']}:{r['kind']}:{bool(r['label'])}"] += 1
            key = normalized_text_key(str(r["text"]))
            prev = label_by_text.setdefault(key, bool(r["label"]))
            if prev != bool(r["label"]):
                conflicts += 1
            if n % sample_every == 0:
                text = str(r["text"])
                scripts[script_bucket(text)] += 1
                lengths.append(len(text))
                hits = scan_text(text)
                for k, v in hits.items():
                    patterns[k] += v
                    pattern_rows[k] += 1
    lengths.sort()
    def pct(p):
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))] if lengths else None
    return {
        "rows": n,
        "counts": {k: v for k, v in sorted(counts.items())},
        "label_conflicts_same_normalized_text": conflicts,
        "sampled_rows": len(lengths),
        "sample_every": sample_every,
        "script_distribution_sampled": dict(scripts.most_common()),
        "text_length_sampled": {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
                                 "max": lengths[-1] if lengths else None},
        "pattern_hits_sampled": dict(patterns.most_common()),
        "pattern_rows_sampled": dict(pattern_rows.most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/redo-v2/dataset-audit.json")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    report = {"files": {str(Path(p)): audit_file(Path(p)) for p in a.files},
              "notes": [
                  "Pattern/script/length stats are deterministic samples (every Nth row), not full-corpus scans.",
                  "Script bucket is a dominant-script proxy, not language identification.",
                  "No matched text is stored; pattern results are counts only.",
                  "Human label adjudication is not replaced by this audit.",
              ]}
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {"rows": v["rows"], "conflicts": v["label_conflicts_same_normalized_text"],
                           "patterns": v["pattern_rows_sampled"]} for k, v in report["files"].items()}, indent=2))


if __name__ == "__main__":
    main()
