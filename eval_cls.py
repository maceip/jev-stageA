"""Per-call classification on held-out action questions. Usage: python eval_cls.py CKPT TEST.jsonl OUT.json"""
import json, sys
from secjev_scorer import SecJev

ckpt, test, out = sys.argv[1:4]
sj = SecJev(ckpt)
S = []
for l in open(test):
    r = json.loads(l)
    q = dict(r["questions"]["authorization"]); lab = q.pop("label")
    try:
        p = sj.score({"state": r["state"], "questions": {"authorization": q}})["authorization"]
    except ValueError:
        continue
    S.append((p["2"] + 0.5 * p["1"], lab, r["_meta"]["suite"]))
pos = sorted(s for s, l, _ in S if l != 0); neg = sorted(s for s, l, _ in S if l == 0)
import bisect
auc = sum(bisect.bisect_left(neg, x) + 0.5 * (bisect.bisect_right(neg, x) - bisect.bisect_left(neg, x)) for x in pos) / (len(pos) * len(neg))
res = {"ckpt": ckpt, "n": len(S), "pos": len(pos), "neg": len(neg), "auc": auc}
for t in (0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
    res[f"tpr@{t}"] = sum(x >= t for x in pos) / len(pos); res[f"fpr@{t}"] = sum(x >= t for x in neg) / len(neg)
json.dump({"summary": res, "scores": S}, open(out, "w"))
print(json.dumps(res), flush=True)
