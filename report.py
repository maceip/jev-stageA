"""Render analysis JSON (one per agent model) and adaptive-attacker logs as markdown tables."""
import json, math, sys
from pathlib import Path

LABEL = {"important_instructions": "Standard AgentDojo (in SecJev training)", "ood_bland_note": "Bland workflow note (new)",
         "ood_metadata_field": "Metadata-field instruction (new)", "ood_quoted_reply": "Quoted earlier reply (new)",
         "ood_zero_width": "Zero-width obfuscation (new)", "ood_llmail": "LLMail human attacks (new)"}
DET = ["secjev_tool", "secjev_action_meta", "secjev_action_content", "piguard", "deberta", "llm_judge"]


def pct(t):
    p, lo, hi = t
    return f"{100*p:.1f}% [{100*lo:.1f}–{100*hi:.1f}]"


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0, c-h), min(1, c+h)


def agent_tables(name, res, mode):
    out = [f"### {name} — operating point `{mode}`\n"]
    if "protocol" in res:
        protocol = res["protocol"]
        out.append("Primary held-out suites: " + ", ".join(protocol["report_suites"]) +
                   "; benign calibration only: " + ", ".join(protocol["calibration_suites"]) + ".\n")
        out.append("Stop-mode replay; redaction bounds in JSON are not live outcomes or confidence intervals. "
                   "Full-suite results include calibration tasks and are descriptive only.\n")
    b = res["benign"]
    out.append(f"Normal tasks (no attack): n={b['n']} (+{b['errors']} agent errors), undefended utility "
               f"{pct(b['utility_undefended'])}.\n")
    out.append("| Detector | Threshold | Normal-task utility |\n|---|---|---|")
    for d in DET:
        r = res["detectors"][d]
        t = r["thresholds"][mode]
        u = r.get("benign_utility_native") if mode == "native" else r.get("benign_utility_u95")
        out.append(f"| {d} | {t if t is not None else 'n/a'} | {pct(u) if u else 'n/a'} |")
    out.append("")
    hdr = "| Attack set | n | Undefended | " + " | ".join(DET) + " |"
    out.append("**Attack success rate (lower is better)**\n\n" + hdr + "\n|" + "---|" * (len(DET) + 3))
    for atk, row in res["attacks"].items():
        cells = [pct(row[f"{d}@{mode}"]["asr"]) if f"{d}@{mode}" in row else "n/a" for d in DET]
        out.append(f"| {LABEL.get(atk, atk)} | {row['n']} | {pct(row['undefended']['asr'])} | " + " | ".join(cells) + " |")
    out.append("\n**Task completion under attack (higher is better)**\n\n" + hdr + "\n|" + "---|" * (len(DET) + 3))
    for atk, row in res["attacks"].items():
        cells = [pct(row[f"{d}@{mode}"]["utility"]) if f"{d}@{mode}" in row else "n/a" for d in DET]
        out.append(f"| {LABEL.get(atk, atk)} | {row['n']} | {pct(row['undefended']['utility'])} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def adaptive_table(paths):
    out = ["| Agent | Defense / mode / repeat / budget | Pair-repeats | Attacker success within recorded budget | Mean rounds to success |", "|---|---|---|---|---|"]
    for p in paths:
        latest = {}
        for r in map(json.loads, open(p)):
            key = (tuple(r["pair"]), r.get("repeat", 0), r.get("mode", "stop"), r.get("defense"),
                   r.get("threshold"), r.get("model"), r.get("attacker_model"), r.get("max_rounds"))
            latest[key] = r
        groups = {}
        for key, row in latest.items():
            groups.setdefault(key[1:], []).append(row)
        for config, rows in groups.items():
            ok = [r for r in rows if r["error"] is None]
            if not ok:
                continue
            k = sum(r["success"] for r in ok)
            rounds = [r["rounds"] for r in ok if r["success"]]
            parts = Path(p).stem.split("_", 1)
            agent = ok[0].get("model") or parts[0]
            defense = ok[0].get("defense") or (parts[1] if len(parts) > 1 else "unknown")
            repeat, mode, _, threshold, _, attacker, budget = config
            label = f"{defense}@{threshold} / {mode} / {repeat} / {budget if budget is not None else 'unrecorded'}"
            if attacker:
                label += f" / attacker={attacker}"
            mean = f"{sum(rounds)/len(rounds):.1f}" if rounds else "n/a"
            out.append(f"| {agent} | {label} | {len(ok)} (+{len(rows)-len(ok)} errors) | "
                       f"{pct(wilson(k, len(ok)))} | {mean} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    analyses = [a for a in sys.argv[1:] if a.endswith(".json")]
    adaptive = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
    md = []
    for a in analyses:
        res = json.load(open(a))
        for mode in ("native", "u95"):
            md.append(agent_tables(Path(a).stem, res, mode))
    if adaptive:
        md.append("### Adaptive attacker\n\n" + adaptive_table(adaptive))
    print("\n".join(md))
