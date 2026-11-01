"""Reproduce SecJev-4B test accuracy on the agent-security tasks of SecJev-Corpus."""
import gzip, json, sys, time, collections

SOURCES = {"agentdojo_tool_outputs", "injecagent"}


def evaluate_corpus(model, corpus_path: str):
    hit, tot, lat = collections.Counter(), collections.Counter(), []
    with gzip.open(corpus_path, "rt") as s:
        for line in s:
            r = json.loads(line)
            if r["source"] not in SOURCES:
                continue
            qs = json.loads(r["questions"])
            req = {"state": json.loads(r["state"]),
                   "questions": {k: {kk: vv for kk, vv in q.items() if kk not in ("label", "src")} for k, q in qs.items()}}
            t = time.perf_counter(); out = model.score(req); lat.append(time.perf_counter() - t)
            for k, q in qs.items():
                p = out[k]
                if q["type"] == "noul":
                    pred = p.get("true", p.get(True, 0)) >= 0.5 if "true" in p or True in p else max(p, key=p.get) in ("true", "yes")
                    ok = pred == q["label"]
                else:
                    ok = max(p, key=p.get) == q["label"]
                hit[q["src"]] += ok; tot[q["src"]] += 1
    lat.sort()
    return hit, tot, lat


def main(argv=None):
    argv = argv or sys.argv[1:]
    from secjev_scorer import SecJev
    m = SecJev(argv[0])
    hit, tot, lat = evaluate_corpus(m, argv[1])
    for k in tot:
        print(f"{k}: {hit[k]}/{tot[k]} = {100*hit[k]/tot[k]:.2f}%")
    if lat:
        print(f"latency p50 {1000*lat[len(lat)//2]:.0f} ms, p95 {1000*lat[int(len(lat)*.95)]:.0f} ms, n={len(lat)}")


if __name__ == "__main__":
    main()
