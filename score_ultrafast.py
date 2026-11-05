"""Score hosted TypeSafe jev-ultrafast (jev-latest) on bench test_joint rows.
Calibration first (5 rows), then stratified sample. Resumable JSONL out."""
import json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

env = {}
for line in Path("/Users/mac/jev-ultrafast/.env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); env[k.strip()] = v.strip().strip('"').strip("'")
KEY = env["TYPESAFE_API_KEY"]; MODEL = env.get("TYPESAFE_MODEL", "jev-latest")
URL = "https://api.typesafe.ai/v1/systemone"
SRC = Path("/Users/mac/jev-stageA/data/redo/bench/test_joint.jsonl")
OUT = Path("/tmp/ultrafast_scores.jsonl")
N_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 200

def label_of(row):
    q = row["questions"].get("injection") or next(iter(row["questions"].values()))
    v = q.get("label", q.get("answer", q.get("true")))
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str): return v.lower() in ("true", "yes", "1")
    return None

def call(row):
    body = {"model": MODEL, "state": row["state"], "questions": {
        k: {kk: vv for kk, vv in q.items() if kk not in ("label", "answer", "true")}
        for k, q in row["questions"].items()}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    ans = out.get("answers", {})
    qid = "injection" if "injection" in ans else next(iter(ans))
    a = ans[qid]
    if "noul" in a: return float(a["noul"]), dt
    probs = a.get("probabilities") or a.get("probability") or {}
    if isinstance(probs, (int, float)): score = float(probs)
    else: score = float(probs.get("true", probs.get("True", probs.get("yes", 0.0))))
    return score, dt

rows = [json.loads(l) for l in SRC.open()]
lab = [label_of(r) for r in rows]
print("rows", len(rows), "pos", sum(1 for x in lab if x), "neg", sum(1 for x in lab if x is False), "unlabeled", sum(1 for x in lab if x is None), flush=True)
idx = [i for i, x in enumerate(lab) if x is not None]
cal = idx[:5]
t0 = time.perf_counter()
for i in cal:
    s, dt = call(rows[i]); print(f"cal row {i} label={lab[i]} score={s:.4f} dt={dt:.2f}s", flush=True)
print(f"calibration total {time.perf_counter()-t0:.1f}s for 5 rows sequential", flush=True)

done = set()
if OUT.exists():
    done = {json.loads(l)["i"] for l in OUT.open()}
rng = random.Random(20261105)
pos = [i for i in idx if lab[i]]; neg = [i for i in idx if not lab[i]]
rng.shuffle(pos); rng.shuffle(neg)
sample = [i for i in pos[:N_PER_CLASS] + neg[:N_PER_CLASS] if i not in done]
print("to score:", len(sample), flush=True)
fails = 0
with OUT.open("a") as f, ThreadPoolExecutor(max_workers=10) as ex:
    futs = {ex.submit(call, rows[i]): i for i in sample}
    for n, fut in enumerate(futs):
        i = futs[fut]
        try:
            s, dt = fut.result(); f.write(json.dumps({"i": i, "is_pos": lab[i], "score": s, "dt": round(dt, 3)}) + "\n"); f.flush()
        except Exception as e:
            fails += 1; print("FAIL", i, type(e).__name__, str(e)[:200], flush=True)
        if (n + 1) % 100 == 0: print("done", n + 1, flush=True)
print("fails:", fails, "out:", OUT)
