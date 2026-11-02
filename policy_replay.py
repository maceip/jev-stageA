"""Evaluate the provenance policy (stop mode) on recorded runs by exact replay.

For each recorded episode: rebuild the task-start environment, snapshot known entities, walk the recorded tool calls in
order, cut at the first call the policy refuses, and recompute utility / attack success with AgentDojo's own checks
(replay.outcome). Episodes the policy never refuses keep their recorded outcome (replay parity is tested separately).

Usage: python policy_replay.py RUN.jsonl [RUN.jsonl ...] [--suites banking,slack] [--out results.json]
"""
import argparse, collections, json, math
from pathlib import Path

import policy
import replay
from score import decision_points


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def first_refusal(ep):
    s = replay.suites()[ep["suite"]]
    ut = s.get_user_task_by_id(ep["user_task"])
    env = ut.init_environment(s.load_and_inject_default_environment(ep["injections"]))
    pe = policy.Episode(policy.SUITE_OF_ENV[type(env).__name__], policy.known_entities(env))
    for kind, i, k, p in decision_points(ep["messages"]):
        if kind == "action":
            v = policy.check(pe, p["user"], p["call"]["function"], p["call"]["args"])
            if not v.allowed:
                return ("action", i, k), p["call"]
    return None, None


def evaluate(path, suites=None):
    rows = []
    for line in open(path):
        ep = json.loads(line)
        if ep["error"] is not None or (suites and ep["suite"] not in suites):
            continue
        cut, call = first_refusal(ep)
        util, atk = (ep["utility"], ep["attack_success"]) if cut is None else replay.outcome(ep, cut)
        rows.append({"suite": ep["suite"], "user_task": ep["user_task"], "injection_task": ep["injection_task"],
                     "variant": ep.get("variant", 0), "refused": call, "utility": bool(util),
                     "attack_success": bool(atk) if ep["injection_task"] else None,
                     "undefended_utility": bool(ep["utility"]),
                     "undefended_attack_success": bool(ep["attack_success"]) if ep["injection_task"] else None})
    return rows


def summary(rows):
    out = {}
    for suite in sorted({r["suite"] for r in rows}) + ["ALL"]:
        rs = [r for r in rows if suite in ("ALL", r["suite"])]
        n = len(rs)
        att = [r for r in rs if r["injection_task"]]
        out[suite] = {"n": n, "refused": sum(r["refused"] is not None for r in rs),
                      "utility": wilson(sum(r["utility"] for r in rs), n),
                      "undefended_utility": wilson(sum(r["undefended_utility"] for r in rs), n)}
        if att:
            out[suite]["asr"] = wilson(sum(r["attack_success"] for r in att), len(att))
            out[suite]["undefended_asr"] = wilson(sum(r["undefended_attack_success"] for r in att), len(att))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--suites", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    suites = set(a.suites.split(",")) if a.suites else None
    report = {}
    f = lambda t: f"{100 * t[0]:.1f}% [{100 * t[1]:.1f}-{100 * t[2]:.1f}]"
    for run in a.runs:
        rows = evaluate(run, suites)
        s = summary(rows)
        report[run] = {"summary": s, "refusals_on_completed_normal_tasks": collections.Counter(
            f"{r['suite']}/{r['user_task']}: {r['refused']['function']} {json.dumps(r['refused']['args'])[:120]}"
            for r in rows if r["refused"] and r["undefended_utility"] and not r["injection_task"]).most_common()}
        print(f"== {run}")
        for suite, x in s.items():
            line = f"  {suite:9s} n={x['n']:4d} refused={x['refused']:4d}  utility {f(x['undefended_utility'])} -> {f(x['utility'])}"
            if "asr" in x:
                line += f"  | hijacked {f(x['undefended_asr'])} -> {f(x['asr'])}"
            print(line)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
