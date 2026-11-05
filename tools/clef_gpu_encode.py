"""Full-corpus CLEF encode on CUDA, writing ft_clef.py-compatible cache files.

Host tokenisation (clef_litert, pure Python) + pinned Cloudflare/clef-flash
PyTorch backbone on CUDA. No .tflite graph, no Metal, no Mac. Cache keys and
npz layout are byte-compatible with ft_clef.encode_and_extract, so the existing
trainer can consume them with --cache-only.

Optimisations (measured on the MI355X):
  * last_hidden_state instead of output_hidden_states (10x faster: 0.0133 vs
    0.138 s/row single-stream).
  * length-bucketed batching to minimise padding.
  * background threads for zlib compression / disk writes.
Resumes by skipping existing cache files. Strict: overlong rows are refused
and recorded, never truncated.
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--batch-tokens", type=int, default=32768)
    ap.add_argument("--max-rows-per-batch", type=int, default=32)
    ap.add_argument("--save-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--manifest", default="")
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

    class TableStub:
        shape = (int(lm_np.shape[0]), 4096)

        def rows(self, ids):
            return lm_np[np.asarray(ids, dtype=np.int64)]

    table = TableStub()
    host = ClefHost(Tokenizer.from_file(str(bundle / "tokenizer/tokenizer.json")),
                    table, table, object(), buckets=BUCKETS)
    host.max_length = 4096
    pad_id = int(getattr(model.config, "eos_token_id", 0) or 0)

    cache = pathlib.Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {"device": a.device, "pytorch_revision": "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
                "bundle": str(bundle), "batch_tokens": a.batch_tokens,
                "strict_length": True, "capped": False, "splits": {}}
    t_all = time.time()
    split_list = a.splits.split(",")

    def save_one(out: pathlib.Path, seq: np.ndarray, lex: np.ndarray) -> None:
        tmp = out.with_name(out.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, hidden=seq, lexical=lex)
        tmp.rename(out)

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.save_workers) as pool:
        for split in split_list:
            src = pathlib.Path(a.data_dir) / f"clef_{split}.jsonl"
            rows = [json.loads(l) for l in src.open() if l.strip()]
            t0 = time.time()
            encoded = cached = overlong = 0
            items = []
            for idx, row in enumerate(rows, 1):
                qid, q = next(iter(row["questions"].items()))
                clean_q = {k: v for k, v in q.items() if k != "label"}
                req = {"model": "clef-flash", "state": row["state"],
                       "questions": {qid: clean_q}}
                key = sha1(json.dumps(req, sort_keys=True, ensure_ascii=False))
                if (cache / f"{key}.npz").exists():
                    cached += 1
                    continue
                try:
                    enc = host.encode(req)
                except (RequestTooLong, SchemaTooLong):
                    overlong += 1
                    with (cache / "skipped_overlong.jsonl").open("a") as f:
                        f.write(json.dumps({"key": key, "split": split,
                                            "reason": "RequestTooLong"}) + "\n")
                    continue
                rec = enc.record
                q_meta = rec.questions[0]
                items.append({"key": key, "ids": [int(x) for x in rec.ids],
                              "tokens": int(rec.tokens),
                              "spans": [tuple(map(int, s)) for s in q_meta.option_spans]})
                if idx % 25000 == 0:
                    print(f"{split} tokenised {idx}/{len(rows)} sec={time.time()-t0:.1f}", flush=True)

            # length bucketing: sort so each batch pads to a similar length
            items.sort(key=lambda x: x["tokens"])
            tokens_seen = [x["tokens"] for x in items]
            futures = []
            batch, batch_max = [], 0
            nb = 0

            def flush(b, bmax):
                nonlocal encoded, nb, futures
                if not b:
                    return
                ids = torch.full((len(b), bmax), pad_id, dtype=torch.long, device=a.device)
                attn = torch.zeros_like(ids)
                for i, x in enumerate(b):
                    ids[i, :len(x["ids"])] = torch.tensor(x["ids"], device=a.device)
                    attn[i, :len(x["ids"])] = 1
                with torch.no_grad():
                    h = model(input_ids=ids, attention_mask=attn).last_hidden_state
                for i, x in enumerate(b):
                    seq = h[i, :x["tokens"]].float().cpu().numpy().astype(np.float16)
                    lex = np.stack([lm_np[np.asarray(x["ids"][s:e], dtype=np.int64)].mean(axis=0)
                                    for s, e in x["spans"]]).astype(np.float16)
                    futures.append(pool.submit(save_one, cache / f"{x['key']}.npz", seq, lex))
                    encoded += 1
                nb += 1
                if nb % 500 == 0:
                    print(f"{split} batch {nb} rows={encoded} sec={time.time()-t0:.1f}", flush=True)
                # bound outstanding futures so RAM does not grow without limit
                if len(futures) > a.save_workers * 64:
                    for f in futures:
                        f.result()
                    futures = []

            for x in items:
                trial_max = max(batch_max, x["tokens"])
                if (batch and (len(batch) + 1 > a.max_rows_per_batch
                               or (len(batch) + 1) * trial_max > a.batch_tokens)):
                    flush(batch, batch_max)
                    batch, batch_max = [], 0
                    trial_max = x["tokens"]
                batch.append(x)
                batch_max = trial_max
            flush(batch, batch_max)
            for f in futures:
                f.result()

            elapsed = time.time() - t0
            manifest["splits"][split] = {
                "rows": len(rows), "encoded": encoded, "already_cached": cached,
                "overlong_refused": overlong, "seconds": round(elapsed, 1),
                "seconds_per_row": round(elapsed / max(1, encoded), 4),
                "batches": nb,
                "token_lengths": {"min": min(tokens_seen) if tokens_seen else None,
                                  "median": sorted(tokens_seen)[len(tokens_seen)//2] if tokens_seen else None,
                                  "max": max(tokens_seen) if tokens_seen else None},
                "source": str(src)}
            print(f"SPLIT_DONE {split} " + json.dumps(manifest["splits"][split]), flush=True)

    manifest["total_seconds"] = round(time.time() - t_all, 1)
    manifest["cache_files"] = len(list(cache.glob("*.npz")))
    out = pathlib.Path(a.manifest) if a.manifest else cache / "encode_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print("ENCODE_COMPLETE", json.dumps({k: manifest["splits"][k] for k in split_list}), flush=True)
    print("MANIFEST", out, flush=True)


if __name__ == "__main__":
    main()
