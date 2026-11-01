"""Sample SecJev-Corpus train records as replay; check each label maps to a Kev option key."""
import gzip, json, os, pathlib, random, sys, collections
WORK = pathlib.Path(os.environ["SECJEV_WORK"]); sys.path.insert(0, str(WORK / "source/kev"))
from kev.api import SystemOneRequest, to_record
src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
recs = [json.loads(l) for l in gzip.open(src, "rt")]
random.Random(7).shuffle(recs)
by = collections.Counter(); kept = []; bad = 0
def key(lab):
    return json.dumps(lab) if isinstance(lab, bool) else str(lab)
for r in recs:
    st = json.loads(r["state"]) if isinstance(r["state"], str) else r["state"]
    qs = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
    labels = {k: q.get("label") for k, q in qs.items()}
    clean = {"state": st, "questions": {k: {kk: vv for kk, vv in q.items() if kk not in ("label", "src")} for k, q in qs.items()}}
    try:
        _, meta = to_record(SystemOneRequest.model_validate(clean))
    except Exception as e:
        bad += 1; continue
    ok = True; newq = {}
    for m in meta:
        lab = labels[m["id"]]; ks = [str(x) for x in m["keys"]]
        cand = [key(lab), str(lab), str(lab).lower()]
        hit = next((c for c in cand if c in ks), None)
        if hit is None: ok = False; break
        newq[m["id"]] = dict(clean["questions"][m["id"]], label=hit)
    if not ok:
        if bad < 3: print("nomatch", labels, [m["keys"] for m in meta])
        bad += 1; continue
    kept.append({"state": st, "questions": newq, "_meta": {"source": r["source"]}}); by[r["source"]] += 1
    if len(kept) >= n: break
json.dump(None, open(os.devnull, "w"))
with open(out, "w") as f:
    for k in kept: f.write(json.dumps(k) + "\n")
print("kept", len(kept), "bad", bad, dict(by))
