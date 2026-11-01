"""Reproduce SecJev-4B test accuracy on the agent-security tasks of SecJev-Corpus."""
import gzip, json, sys, time, collections
from secjev_scorer import SecJev

SOURCES = {"agentdojo_tool_outputs", "injecagent"}
m = SecJev(sys.argv[1])
hit, tot, lat = collections.Counter(), collections.Counter(), []
with gzip.open(sys.argv[2], "rt") as s:
    for line in s:
        r = json.loads(line)
        if r["source"] not in SOURCES:
            continue
        qs = json.loads(r["questions"])
        req = {"state": json.loads(r["state"]),
               "questions": {k: {kk: vv for kk, vv in q.items() if kk not in ("label", "src")} for k, q in qs.items()}}
        t = time.perf_counter(); out = m.score(req); lat.append(time.perf_counter() - t)
        for k, q in qs.items():
            p = out[k]
            if q["type"] == "noul":
                pred = p.get("true", p.get(True, 0)) >= 0.5 if "true" in p or True in p else max(p, key=p.get) in ("true", "yes")
                ok = pred == q["label"]
            else:
                ok = max(p, key=p.get) == q["label"]
            hit[q["src"]] += ok; tot[q["src"]] += 1
for k in tot:
    print(f"{k}: {hit[k]}/{tot[k]} = {100*hit[k]/tot[k]:.2f}%")
lat.sort(); print(f"latency p50 {1000*lat[len(lat)//2]:.0f} ms, p95 {1000*lat[int(len(lat)*.95)]:.0f} ms, n={len(lat)}")
