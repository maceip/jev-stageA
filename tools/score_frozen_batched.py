"""Batched per-row scoring of frozen splits with a Kev/SecJev checkpoint.

Uses exactly the same statistic as ft_action.evaluate() and
eval_secjev_frozen.py (P("2") + 0.5*P("1"), or P("true") for noul
questions via verdict_score), but length-sorted token-budget microbatches
instead of one forward per row. This removes a ~90-minute single-row scoring
bottleneck without changing any metric.

Writes per-row scores so every reported number is recomputable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

import torch

WORK = pathlib.Path(os.environ["SECJEV_WORK"]).resolve()
sys.path.insert(0, str(WORK / "source/kev"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kev.api import SystemOneRequest, to_record  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.model import rows_of  # noqa: E402
from ft_action import encode, verdict_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tok-budget", type=int, default=4096)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    a = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    ck = Checkpoint(a.weights)
    ck.meta.base = str(WORK / "base")
    ck.meta.base_revision = None
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    tok, model = ck.load(a.device, LoadOptions(dtype=dtype, merge=False, attn="sdpa", temperature=1.0))
    model.eval()
    model.lm.config.use_cache = False

    rows = [json.loads(l) for l in open(a.split) if l.strip()]
    prepared, skipped = [], 0
    for i, r in enumerate(rows):
        e = encode(model, tok, r)
        if e is None:
            skipped += 1
            continue
        meta = r.get("_meta") or {}
        prepared.append((i, e, meta))
    print(f"encoded {len(prepared)} of {len(rows)} (skipped_over_window {skipped})", flush=True)

    ordered = sorted(prepared, key=lambda x: x[1][3])
    micros, cur = [], []
    for x in ordered:
        L = x[1][3]
        if cur and (len(cur) + 1) * L > a.tok_budget:
            micros.append(cur)
            cur = []
        cur.append(x)
    if cur:
        micros.append(cur)

    out = []
    t0 = time.time()
    with torch.no_grad():
        for mi, micro in enumerate(micros, 1):
            zss = model.forward_batch([x[1][0] for x in micro])
            for zs, (idx, (_enc, tgt, keys, _L), meta) in zip(zss, micro):
                p = dict(zip(keys, torch.softmax(zs[0].float(), -1).tolist()))
                label_str = keys[tgt[0]]
                out.append({
                    "row_index": idx,
                    "row_id": meta.get("row_id"),
                    "score": verdict_score(p, keys),
                    "label": label_str not in ("0", "false"),
                    "group": meta.get("suite") or meta.get("source") or "?",
                    "suite": meta.get("suite"),
                    "question_keys": list(keys),
                })
            if mi % 20 == 0:
                print(f"microbatch {mi}/{len(micros)} rows={len(out)} elapsed={time.time()-t0:.1f}s", flush=True)

    out.sort(key=lambda r: r["row_index"])
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"wrote": len(out), "skipped": skipped,
                      "microbatches": len(micros), "seconds": round(time.time() - t0, 1),
                      "out": a.out}), flush=True)


if __name__ == "__main__":
    main()
