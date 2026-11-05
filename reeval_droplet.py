"""Correct per-question dev re-evaluation of droplet FT checkpoints.

ft_action.evaluate assumes 3-class authorization keys; injection rows
(noul, 'true'/'false') need P('true'). Reports AUC/TPR/FPR per question.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, "/root/work")
os.environ.setdefault("SECJEV_WORK", "/root/secjev-work")
sys.path.insert(0, "/root/secjev-work/source/kev")
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
import ft_action  # noqa: E402


def auc(pairs):
    pos = [s for s, p in pairs if p]; neg = [s for s, p in pairs if not p]
    if not pos or not neg: return None
    return sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dev", required=True)
    a = ap.parse_args()
    ck = Checkpoint(a.ckpt)
    ck.meta.base = "/root/secjev-work/base"; ck.meta.base_revision = None
    tok, model = ck.load("cuda", LoadOptions(dtype=torch.float32, merge=False, attn="sdpa", temperature=1.0))
    model.eval()
    rows = ft_action.load_jsonl(a.dev)
    items = []
    for r in rows:
        got = ft_action.encode(model, tok, r)
        if got is None: continue
        enc, tgt, keys, L = got
        qid = next(iter(r["questions"]))
        items.append((enc, tgt, keys, L, qid))
    by_q: dict[str, list] = {}
    ordered = sorted(items, key=lambda x: x[3])
    micros, cur = [], []
    for x in ordered:
        if cur and (len(cur) + 1) * x[3] > 16384: micros.append(cur); cur = []
        cur.append(x)
    if cur: micros.append(cur)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for micro in micros:
            zss = model.forward_batch([x[0] for x in micro])
            for zs, (_e, tgt, keys, _L, qid) in zip(zss, micro):
                p = dict(zip(keys, torch.softmax(zs[0].float(), -1).tolist()))
                truth = keys[tgt[0]]
                if qid == "injection":
                    by_q.setdefault(qid, []).append((p.get("true", 0.0), truth == "true"))
                else:
                    by_q.setdefault(qid, []).append((p.get("2", 0.0) + 0.5 * p.get("1", 0.0), truth != "0"))
    out = {}
    for qid, pairs in sorted(by_q.items()):
        pos = [s for s, p in pairs if p]; neg = [s for s, p in pairs if not p]
        out[qid] = {"n": len(pairs), "pos": len(pos), "auc": auc(pairs),
                    "tpr@0.5": sum(s >= 0.5 for s in pos) / max(1, len(pos)),
                    "fpr@0.5": sum(s >= 0.5 for s in neg) / max(1, len(neg))}
    print(json.dumps({"ckpt": a.ckpt, "dev": a.dev, "by_question": out}, indent=2))


if __name__ == "__main__":
    main()
