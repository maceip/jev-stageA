"""Measure the portable CLEF encode path on a CUDA host.

Pipeline: clef_litert.ClefHost.encode (pure Python/numpy tokenisation) ->
pinned Cloudflare/clef-flash PyTorch backbone on CUDA -> [T,4096] hidden states
+ mean lexical vectors from lm_head. No .tflite graph, no Metal, no Mac.

This is the G3 encode-rate measurement required by tuesday-protocol.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from transformers import AutoModel


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/root/work/checkpoints/clef-flash-LiteRT")
    ap.add_argument("--pytorch", default="/root/.cache/huggingface/hub/models--Cloudflare--clef-flash/snapshots/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f")
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    bundle = pathlib.Path(a.bundle)
    sys.path.insert(0, str(bundle / "host"))
    from clef_litert.errors import RequestTooLong, SchemaTooLong  # noqa: E402
    from clef_litert.host import ClefHost  # noqa: E402
    from clef_litert.positions import BUCKETS  # noqa: E402

    spec = importlib.util.spec_from_file_location("joint_schema_model", str(bundle / "joint_schema_model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    model = AutoModel.from_pretrained(a.pytorch, trust_remote_code=True,
                                      dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to(a.device).eval()
    print("BACKBONE_LOADED", type(model).__name__,
          sum(p.numel() for p in model.parameters()), flush=True)

    # The 3.8 GB BF16 tables are only needed by the LiteRT row_inputs/scores path.
    # For the CUDA path we serve lm_head rows from the pinned PyTorch weights instead.
    from safetensors import safe_open
    lm = None
    for shard in sorted(pathlib.Path(a.pytorch).glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            if "lm_head.weight" in f.keys():
                lm = f.get_tensor("lm_head.weight")
                break
    if lm is None:
        lm = model.get_input_embeddings().weight.detach().cpu()
    lm_np = lm.float().numpy()
    print("LM_HEAD", tuple(lm_np.shape), flush=True)

    class TableStub:
        shape = (int(lm_np.shape[0]), 4096)

        def rows(self, ids):
            return lm_np[np.asarray(ids, dtype=np.int64)]

    table = TableStub()
    host = ClefHost(
        Tokenizer.from_file(str(bundle / "tokenizer/tokenizer.json")),
        table, table, object(), buckets=BUCKETS)
    host.max_length = 4096  # strict: refuse anything longer than L4096

    rows = [json.loads(l) for l in open(a.split)][:a.limit]
    t0 = time.time()
    ok = overlong = 0
    lengths = []
    for i, row in enumerate(rows):
        qid, q = next(iter(row["questions"].items()))
        label = str(q["label"])
        clean_q = {k: v for k, v in q.items() if k != "label"}
        req = {"model": "clef-flash", "state": row["state"],
               "questions": {qid: clean_q}}
        try:
            enc = host.encode(req)
        except (RequestTooLong, SchemaTooLong):
            overlong += 1
            continue
        rec = enc.record
        ids = torch.tensor([int(x) for x in rec.ids], device=a.device).unsqueeze(0)
        with torch.no_grad():
            h = model(input_ids=ids, output_hidden_states=True).hidden_states[-1][0]
        seq = h[:rec.tokens].float().cpu().numpy()
        q_meta = rec.questions[0]
        ids_arr = np.asarray(rec.ids, dtype=np.int64)
        lex = np.stack([host.lm_head.rows(ids_arr[s:e]).mean(axis=0)
                        for s, e in q_meta.option_spans]).astype(np.float32)
        assert seq.shape == (rec.tokens, 4096), seq.shape
        assert lex.shape[0] == len(q_meta.option_ids)
        lengths.append(int(rec.tokens))
        ok += 1
        if (i + 1) % 10 == 0:
            print(f"encoded {i+1}/{len(rows)} ok={ok} overlong={overlong} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    elapsed = time.time() - t0
    result = {
        "device": a.device, "rows_attempted": len(rows), "rows_encoded": ok,
        "overlong_refused": overlong, "elapsed_seconds": round(elapsed, 2),
        "seconds_per_row": round(elapsed / max(1, ok), 4),
        "token_lengths": {"min": min(lengths) if lengths else None,
                          "median": sorted(lengths)[len(lengths)//2] if lengths else None,
                          "max": max(lengths) if lengths else None},
        "projected_hours_full_146984_rows": round(146984 * (elapsed / max(1, ok)) / 3600, 2),
        "cache_key_compatible_with": "ft_clef.py encode_and_extract (sha1 of sorted request JSON)",
        "note": "strict length policy: overlong rows refused and recorded, never silently truncated",
    }
    print(json.dumps(result, indent=2), flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
