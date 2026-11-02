"""Render analysis JSON (one per agent model) and adaptive-attacker logs as markdown tables."""
import json, math, sys
from pathlib import Path

LABEL = {"important_instructions": "Standard AgentDojo (in SecJev training)", "ood_bland_note": "Bland workflow note (new)",
         "ood_metadata_field": "Metadata-field instruction (new)", "ood_quoted_reply": "Quoted earlier reply (new)",
         "ood_zero_width": "Zero-width obfuscation (new)", "ood_llmail": "LLMail human attacks (new)"}
DET = ["secjev_tool", "secjev_action_meta", "secjev_action_content", "secjev_cascade", "secjev_cascade_judge", "piguard", "deberta", "llm_judge"]


def _pooled_blocked_and_fp(res, det, mode="u95"):
    b_u0 = res["benign"]["utility_undefended"][0]
    d_info = res["detectors"].get(det, {})
    u_key = "benign_utility_u95" if (mode == "u95" and d_info.get("benign_utility_u95")) else "benign_utility_native"
    u_def = (d_info.get(u_key) or [b_u0])[0]
    fp_pct = max(0.0, (b_u0 - u_def) / b_u0 * 100.0) if b_u0 > 0 else 0.0
    total_n = 0
    total_s = 0.0
    for row in res.get("attacks", {}).values():
        cell = row.get(f"{det}@{mode}") or row.get(f"{det}@native")
        n = row.get("n", 0)
        if cell and n > 0:
            total_n += n
            total_s += cell["asr"][0] * n
    blocked_pct = (1.0 - (total_s / total_n)) * 100.0 if total_n > 0 else 0.0
    return blocked_pct, fp_pct


def _adaptive_blocked_map(paths):
    by_def = {}
    for p in paths or []:
        latest = {}
        for r in map(json.loads, open(p)):
            latest[tuple(r["pair"])] = r
        ok = [r for r in latest.values() if r.get("error") is None]
        if ok:
            _, defense = Path(p).stem.split("_", 1)
            by_def[defense] = (1.0 - sum(bool(r["success"]) for r in ok) / len(ok)) * 100.0
    return by_def


def executive_summary(name, res, adaptive_paths=None):
    ad = _adaptive_blocked_map(adaptive_paths)
    p_blk, p_fp = _pooled_blocked_and_fp(res, "secjev_tool", "u95")
    a_blk, a_fp = _pooled_blocked_and_fp(res, "secjev_action_meta", "native")
    c_blk, c_fp = _pooled_blocked_and_fp(res, "secjev_cascade", "u95")
    j_blk, j_fp = _pooled_blocked_and_fp(res, "secjev_cascade_judge", "u95")

    def _ad_str(key, fallback):
        return f"blocked {ad[key]:.1f}% of 10-try adaptive red-team attacks" if key in ad else fallback

    lines = [
        f"### Executive Summary ({name}) — JEV-4B in 2 places: Prompt Filter & Tool-Call Filter\n",
        f"- **Prompt filter (JEV-4B alone):** blocked **{p_blk:.1f}%** of 1-try attacks "
        f"({_ad_str('secjev_tool', '10-try adaptive test pending')}) with **{p_fp:.1f}% FP**",
        f"- **Tool-call filter (JEV-4B alone):** blocked **{a_blk:.1f}%** of 1-try attacks "
        f"({_ad_str('secjev_action_meta', '10-try adaptive test pending')}) with **{a_fp:.1f}% FP**",
        f"- **TOGETHER (Prompt Filter + Tool-Call Filter — Double JEV):** blocked **{c_blk:.1f}%** of 1-try attacks "
        f"({_ad_str('secjev_cascade', 'waiting on GPU to run 10-try adaptive test')}) with **{c_fp:.1f}% FP**",
        f"- **Double JEV + LLM Judge (Prompt Filter + Tool-Call Filter + LLM Judge second opinion on flags):** "
        f"blocked **{j_blk:.1f}%** of 1-try attacks "
        f"({_ad_str('secjev_cascade_judge', 'waiting on GPU to run 10-try adaptive test')}) with **{j_fp:.1f}% FP**\n",
    ]
    return "\n".join(lines)


def pct(t):
    p, lo, hi = t
    return f"{100*p:.1f}% [{100*lo:.1f}–{100*hi:.1f}]"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0, c-h), min(1, c+h)


def agent_tables(name, res, mode):
    out = [f"### {name} — operating point `{mode}`\n"]
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
    out = ["| Agent | Defense | Pairs | Attacker success within 10 rounds | Mean rounds to success |", "|---|---|---|---|---|"]
    for p in paths:
        latest = {}
        for r in map(json.loads, open(p)):
            latest[tuple(r["pair"])] = r
        R = list(latest.values())
        ok = [r for r in R if r["error"] is None]
        if not ok:
            continue
        k = sum(r["success"] for r in ok)
        rounds = [r["rounds"] for r in ok if r["success"]]
        agent, defense = Path(p).stem.split("_", 1)
        out.append(f"| {agent} | {defense} | {len(ok)} (+{len(R)-len(ok)} errors) | {pct(wilson(k, len(ok)))} | "
                   f"{sum(rounds)/len(rounds):.1f} |" if rounds else
                   f"| {agent} | {defense} | {len(ok)} (+{len(R)-len(ok)} errors) | {pct(wilson(k, len(ok)))} | – |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    analyses = [a for a in sys.argv[1:] if a.endswith(".json")]
    adaptive = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
    md = []
    for a in analyses:
        res = json.load(open(a))
        md.append(executive_summary(Path(a).stem, res, adaptive))
        for mode in ("native", "u95"):
            md.append(agent_tables(Path(a).stem, res, mode))
    if adaptive:
        md.append("### Adaptive attacker\n\n" + adaptive_table(adaptive))
    print("\n".join(md))
