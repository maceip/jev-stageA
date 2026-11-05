"""Deterministic adaptive-attack variants of Spot-2 test rows (Phase 6, eval-only).

Mutates the untrusted tool outputs inside each test row's state with
cheap attacker-moves-second transforms. Labels and proposed calls are
unchanged; only the attack surface text changes. Output files keep the
ft_action input format so eval_secjev_frozen.py scores them unchanged.
Variants are never mixed into training data.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external/redo_v2_out"

ZW = "​"
HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "i": "і"}


def t_zerowidth(text: str) -> str:
    return ZW.join(text.split(" "))


def t_homoglyph(text: str) -> str:
    return "".join(HOMO.get(ch, ch) for ch in text)


def t_quote(text: str) -> str:
    return "\n".join("> " + line for line in text.splitlines() or [text])


def t_case(text: str) -> str:
    return text.swapcase()


TRANSFORMS = {"zerowidth": t_zerowidth, "homoglyph": t_homoglyph, "quote": t_quote, "case": t_case}


def mutate_state(state: dict, fn) -> dict:
    state = copy.deepcopy(state)
    seen = state.get("untrusted_tool_outputs_seen") or []
    for item in seen:
        if isinstance(item, dict):
            for key in ("output", "text", "content"):
                if isinstance(item.get(key), str):
                    item[key] = fn(item[key])
        elif isinstance(item, str):
            pass  # handled by caller replacement below
    state["untrusted_tool_outputs_seen"] = [
        fn(x) if isinstance(x, str) else x for x in seen
    ]
    return state


def has_output_text(row: dict) -> bool:
    return any(
        isinstance(item, dict) and isinstance(item.get("output"), str) and item["output"]
        for item in row["state"].get("untrusted_tool_outputs_seen") or []
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=str(OUT / "ft_spot2v2_test.jsonl"))
    ap.add_argument("--positives-only", action="store_true", default=True)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.split, encoding="utf-8") if line.strip()]
    # Adaptive text mutation is only meaningful where the state actually
    # carries tool-output text; most Spot-2 states carry provenance only.
    eligible = [r for r in rows if has_output_text(r)]
    summary = {"_coverage": {"split_rows": len(rows), "rows_with_output_text": len(eligible)}}
    rows = eligible
    for name, fn in TRANSFORMS.items():
        dst = OUT / f"ft_spot2v2_test_adaptive_{name}.jsonl"
        n = 0
        with dst.open("w", encoding="utf-8") as f:
            for row in rows:
                if a.positives_only and int(row["questions"]["authorization"]["label"]) == 0:
                    continue
                out = dict(row)
                out["state"] = mutate_state(row["state"], fn)
                out["_meta"] = {**(row.get("_meta") or {}), "adaptive_variant": name}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                n += 1
        summary[name] = {"rows": n, "file": str(dst.relative_to(ROOT))}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
