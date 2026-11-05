"""Publish a droplet CLEF redo FT checkpoint to Hugging Face.

Input: the assembled checkpoint dir produced by ft_clef_bench.py
(--ckpt_out, default checkpoints/clef-flash-LiteRT-ft): a full LiteRT
checkpoint (tflite backbone + tables + tokenizer + host) whose
head/joint_head.safetensors is the fine-tuned JointSchemaHead.

Gates, in order (any failure aborts before upload):
  1. joint_head.safetensors loads; true parameter count is computed
     from the state dict (the v1 "60.8M" claim was wrong: 121,762,820).
  2. Dtype contract: the deployment loader expects BF16 (v1 F32 head
     failed to load). An F32 head is converted to BF16 in staging and
     the conversion is recorded in redo-provenance.json.
  3. Required sibling files present (tflite, contract.json, tokenizer,
     host/, tables/, head/joint_head_config.json).
The card states dev-only status until frozen evaluation exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil

from tuesday_gates import require_frozen_chain  # Tuesday-v2 fail-closed gate

REQUIRED = ["head/joint_head.safetensors", "head/joint_head_config.json",
            "tokenizer", "host", "tables", "joint_schema_model.py"]

CARD = """---
language:
- en
license: other
tags:
- security
- prompt-injection
- litert
---

# {repo_name}

Redo fine-tune of the CLEF-Flash `JointSchemaHead` (121,762,820 parameters,
122 tensors) on frozen Cloudflare CLEF-Flash LiteRT backbone hidden states.
Only the head is trained; the backbone is unmodified. This repository root
is a complete CLEF LiteRT checkpoint directory with the fine-tuned head.

## Training

- Compute: {compute}, {wall}.
- Method: `{method}`, {epochs} epoch(s), lr {lr}, batch {batch},
  pos_weight {pos_weight}. Training details and hashes: `redo-provenance.json`;
  full log: `redo-training-log.jsonl` when present.
- Head dtype in this repo: BF16 (deployment contract). {dtype_note}

## Evaluation status

{eval_note}

No frozen-test, adaptive-attack, or independent-replication result exists yet
for this checkpoint unless stated above. Treat all numbers as development
measurements, not deployment guarantees.
"""


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_stage(src: pathlib.Path, stage: pathlib.Path) -> dict:
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    for rel in REQUIRED:
        assert (src / rel).exists(), f"missing required checkpoint member: {rel}"
    tflites = list(src.glob("*.tflite"))
    assert tflites, "missing *.tflite backbone in checkpoint dir"
    head_path = src / "head/joint_head.safetensors"
    with safe_open(str(head_path), framework="pt") as f:
        keys = list(f.keys())
        params = int(sum(int(np.prod(f.get_slice(k).get_shape())) for k in keys))
        dtypes = sorted({str(f.get_slice(k).get_dtype()) for k in keys})
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(src, stage, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "cache"))
    for tf in stage.glob("*.tflite*"):  # ckpt_out symlinks the backbone; materialize for upload
        if tf.is_symlink():
            target = tf.resolve()
            tf.unlink()
            shutil.copy2(target, tf)
    converted = False
    if dtypes != ["BF16"]:
        sd = load_file(str(head_path))
        save_file({k: v.to(torch.bfloat16) for k, v in sd.items()},
                  str(stage / "head/joint_head.safetensors"), metadata={"format": "pt"})
        converted = True
    return {"head_tensors": len(keys), "head_params": params, "source_head_dtypes": dtypes,
            "bf16_converted_in_staging": converted,
            "source_head_sha256": sha256(head_path),
            "staged_head_sha256": sha256(stage / "head/joint_head.safetensors"),
            "tflite": [{"name": t.name, "bytes": t.stat().st_size} for t in tflites]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="assembled clef-flash-LiteRT-ft dir (downloaded from droplet)")
    ap.add_argument("--staging", default="/tmp/hf-clef-redo")
    ap.add_argument("--repo", default="macmacmacmac/CLEF-Flash-Redo")
    ap.add_argument("--compute", default="DigitalOcean GPU droplet `jev-train-1` (NVIDIA H100 80GB)")
    ap.add_argument("--method", default="ft_clef_bench.py")
    ap.add_argument("--host-label", default="jev-train-1 (DigitalOcean gpu-h100x1-80gb, 159.65.192.220)")
    ap.add_argument("--wall", default="see redo-provenance.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--eval-note", default="Dev-only metrics pending corrected re-evaluation at publish time.")
    ap.add_argument("--training-log", default="", help="optional training log.jsonl to include")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    stage = pathlib.Path(a.staging)
    checks = verify_and_stage(pathlib.Path(a.ckpt), stage)
    print("verify:", json.dumps(checks, indent=2))
    if a.training_log:
        shutil.copy2(a.training_log, stage / "redo-training-log.jsonl")
    (stage / "redo-provenance.json").write_text(json.dumps(
        {"host": a.host_label, "method": a.method,
         "wall": a.wall, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
         "pos_weight": a.pos_weight, "checks": checks}, indent=2) + "\n")
    (stage / "README.md").write_text(CARD.format(
        repo_name=a.repo.split("/")[-1], compute=a.compute, method=a.method,
        wall=a.wall, epochs=a.epochs, lr=a.lr,
        batch=a.batch, pos_weight=a.pos_weight, eval_note=a.eval_note,
        dtype_note=("Converted from the F32 training artifact in staging; both hashes are in redo-provenance.json."
                    if checks["bf16_converted_in_staging"] else "Trained head was already BF16; no conversion.")))
    if a.verify_only:
        print("staged at", stage)
        return
    require_frozen_chain()  # refuses upload without the full frozen-test evidence chain
    token = os.environ.get("HF_TOKEN")
    if not token and pathlib.Path(".env").is_file():
        for line in pathlib.Path(".env").read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(stage), repo_id=a.repo, repo_type="model",
                      commit_message="CLEF redo FT checkpoint (JointSchemaHead)")
    print("uploaded https://huggingface.co/" + a.repo)


if __name__ == "__main__":
    main()
