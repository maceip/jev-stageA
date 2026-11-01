"""Rescore only the metadata action-monitor decision points of recorded runs with a (fine-tuned) SecJev checkpoint.

Writes a score file with just {kind, msg, call, secjev_action_meta[, _probs, _dropped_history]} per action point, named so
analyze.load_scores picks it up before the original secjev files (first value wins per point).
Usage: python rescore_action.py CKPT RUN.jsonl OUT.jsonl [--suites banking,slack] [--shard i/n]
"""
import argparse, json
from score import decision_points, secjev_action_score
from secjev_scorer import SecJev

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("run"); ap.add_argument("out")
ap.add_argument("--suites", default="")
ap.add_argument("--shard", default="0/1")
a = ap.parse_args()
si, sn = map(int, a.shard.split("/"))
suites = set(a.suites.split(",")) if a.suites else None
sj = SecJev(a.ckpt)
eps = [json.loads(l) for l in open(a.run)]
eps = [e for j, e in enumerate(eps) if j % sn == si and e["error"] is None and (suites is None or e["suite"] in suites)]
with open(a.out, "w") as f:
    for e in eps:
        pts = []
        for kind, i, k, p in decision_points(e["messages"]):
            if kind != "action":
                continue
            pr, dropped = secjev_action_score(sj, p, False)
            pts.append({"kind": kind, "msg": i, "call": k, "function": p["call"]["function"], "errors": {},
                        "secjev_action_meta": pr["2"] + 0.5 * pr["1"], "secjev_action_meta_probs": pr,
                        "secjev_action_meta_dropped_history": dropped})
        f.write(json.dumps({"suite": e["suite"], "user_task": e["user_task"], "injection_task": e["injection_task"],
                            "variant": e.get("variant", 0), "points": pts}) + "\n")
print("RESCORED", a.run, len(eps), flush=True)
