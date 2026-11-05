"""Reconcile the 450-row blind label-audit packet against its key.

IMPORTANT: this is a DETERMINISTIC PRE-ADJUDICATION, not a human adjudication.
The Tuesday assessment requires human adjudication with agreement rate and a
listed set of disagreements. This script does NOT close that gate. It produces
an honest, reproducible baseline and states explicitly what remains open.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/redo_v2_out"
BLIND = OUT / "label_audit_blind.jsonl"
KEY = OUT / "label_audit_key.json"
REPORT = ROOT / "docs/redo-v2/label-audit-outcome.json"

# Transparent, documented heuristic. This is deliberately a weak baseline:
# its purpose is to expose how many rows need human judgement, not to replace it.
IMPERATIVE = re.compile(
    r"(?i)\b(ignore (all |the )?previous|disregard|you are now|new instructions|"
    r"send (the |an )?(email|money|\$)|forward|exfiltrate|override|"
    r"do not tell|system prompt|reveal (the )?(password|secret))\b")


def machine_verdict(text: str) -> tuple[bool, float, str]:
    hits = IMPERATIVE.findall(str(text))
    return (len(hits) > 0, 0.5, f"imperative_markers={len(hits)}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return round(c - h, 4), round(c + h, 4)


def main() -> None:
    rows = [json.loads(l) for l in BLIND.open()]
    key = json.load(KEY.open())
    by_id = {r["row_id"]: r for r in key["rows"]}
    verdicts, disagreements = [], []
    for r in rows:
        gold = by_id.get(r["row_id"])
        if gold is None:
            raise ValueError(f"row_id {r['row_id']} missing from key")
        pred, conf, note = machine_verdict(r.get("text", ""))
        agree = bool(pred) == bool(gold["label"])
        verdicts.append({"row_id": r["row_id"], "machine_verdict": pred,
                         "gold": bool(gold["label"]), "agree": agree, "note": note})
        if not agree:
            disagreements.append({"row_id": r["row_id"], "kind": r.get("kind"),
                                  "source": r.get("source"),
                                  "machine_verdict": pred, "gold": bool(gold["label"]),
                                  "reason": "requires human adjudication"})
    n = len(verdicts)
    k = sum(v["agree"] for v in verdicts)
    lo, hi = wilson(k, n)
    report = {
        "adjudicator": "deterministic imperative-marker pre-adjudication (NOT HUMAN)",
        "human_adjudication_gate_closed": False,
        "human_adjudication_required": True,
        "n": n,
        "machine_agreement": round(k / n, 4),
        "agreement_wilson95": [lo, hi],
        "n_disagreements": len(disagreements),
        "disagreements": disagreements[:50],
        "disagreement_list_truncated": len(disagreements) > 50,
        "blind_packet_verdict_fields_all_null": all(r.get("verdict_label") is None for r in rows),
        "note": ("The blind packet arrived with verdict_label=null on all 450 rows. "
                 "No human adjudication has been performed. This report is a reproducible "
                 "baseline only; the Phase-3 human gate remains OPEN and blocks publication."),
        "seed": key.get("seed"), "per_stratum": key.get("per_stratum"),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({x: report[x] for x in
                      ("adjudicator", "human_adjudication_gate_closed", "n",
                       "machine_agreement", "n_disagreements")}, indent=2))
    print("WROTE", REPORT)


if __name__ == "__main__":
    main()
