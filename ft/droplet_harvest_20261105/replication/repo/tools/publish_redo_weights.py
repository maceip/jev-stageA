"""Publish droplet redo FT checkpoints (spot2/spot1/joint) to Hugging Face.

Staging dirs (downloaded from jev-train-1) already contain complete Kev
checkpoint directories plus redo-training-log.jsonl / redo-training.stdout.txt.
This script verifies weights load, writes an honest model card +
redo-provenance.json into each staging dir, creates public repos under
the token owner, and uploads. Metrics in the cards are dev-only, from
the corrected per-question re-evaluation; no frozen-test claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib

from tuesday_gates import require_frozen_chain  # Tuesday-v2 fail-closed gate

RESULTS = {
    "spot2": {
        "repo_name": "SecJev-4B-Spot2-Redo", "task": "Spot 2 — action authorization guard (outgoing tool calls)",
        "train_file": "data/bench/train_spot2_t.jsonl", "dev_file": "data/bench/dev_spot2_t.jsonl",
        "dev": {"authorization": {"n": 500, "pos": 85, "auc": 0.9911, "tpr@0.5": 0.9412, "fpr@0.5": 0.0048}},
        "base_dev_auc": 0.8983, "wall": "2026-11-03 22:29:44–22:53:41 UTC",
    },
    "spot1": {
        "repo_name": "SecJev-4B-Spot1-Redo", "task": "Spot 1 — tool-output injection filter (incoming tool responses)",
        "train_file": "data/bench/train_spot1_t.jsonl", "dev_file": "data/bench/dev_spot1_t.jsonl",
        "dev": {"injection": {"n": 500, "pos": 250, "auc": 0.9327, "tpr@0.5": 0.78, "fpr@0.5": 0.04}},
        "base_dev_auc": None, "wall": "2026-11-03 22:53:41–23:03:28 UTC",
    },
    "joint": {
        "repo_name": "SecJev-4B-Joint-Redo", "task": "Joint Spot 1 + Spot 2 (single checkpoint, two questions)",
        "train_file": "data/bench/train_joint_t.jsonl", "dev_file": "data/bench/dev_joint_t.jsonl",
        "dev": {"authorization": {"n": 500, "pos": 85, "auc": 0.9898, "tpr@0.5": 0.9412, "fpr@0.5": 0.0048},
                "injection": {"n": 500, "pos": 250, "auc": 0.9314, "tpr@0.5": 0.776, "fpr@0.5": 0.02}},
        "base_dev_auc": None, "wall": "2026-11-03 23:03:28–23:37:49 UTC",
    },
}

CARD = """---
language:
- en
license: other
license_name: secjev-use-and-distribution-agreement-v1.0
license_link: LICENSE
base_model: Qwen/Qwen3.5-4B-Base
tags:
- security
- decision-model
- lora
- prompt-injection
---

# {repo_name}

Redo fine-tune of SecJev-4B for {task}. This repository root is a complete
Kev checkpoint directory (LoRA adapter + trained head + tokenizer), loadable
with the Kev loader. The Qwen3.5-4B-Base backbone is not re-hosted;
`bootstrap.py` fetches the pinned backbone and Kev source.

## Training

- Compute: DigitalOcean GPU droplet `jev-train-1` (NVIDIA H100 80GB), {wall}.
- Method: `ft_action.py`, 1 epoch, lr 3e-5, batch 32 questions, token budget
  4096, gradient checkpointing, fp32 weights with bf16 autocast.
  Trainable parameters: 33,776,128 (LoRA + head); backbone frozen.
- Data: `{train_file}` (train) / `{dev_file}` (dev), jev-stageA redo bench sets.
- Full training log: `redo-training-log.jsonl` and `redo-training.stdout.txt`
  in this repository. Machine-readable details: `redo-provenance.json`.

## Dev results (only evaluation so far)

Corrected per-question re-evaluation of this checkpoint on the dev file
(score: authorization = P("2") + 0.5·P("1"); injection = P("true")):

{dev_table}

No frozen-test, adaptive-attack, or independent-replication result exists yet
for this checkpoint. Dev rows were used for training-time monitoring. Treat
the numbers above as development measurements, not deployment guarantees.

Note: the training-time dev metrics printed in `redo-training-log.jsonl` for
injection-question runs used an authorization-only scoring formula and are
invalid for those runs (they show AUC 0.0 / 0.14); the table above supersedes
them. Training itself used correct targets.
"""


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dev_table(dev: dict, base_auc) -> str:
    lines = ["| question | dev n | dev pos | AUC | TPR@0.5 | FPR@0.5 |", "|---|---|---|---|---|---|"]
    for q, m in dev.items():
        lines.append(f"| {q} | {m['n']} | {m['pos']} | {m['auc']:.4f} | {m['tpr@0.5']:.4f} | {m['fpr@0.5']:.4f} |")
    if base_auc:
        lines.append(f"\nUntrained SecJev-4B base on the same dev file: AUC {base_auc:.4f}.")
    return "\n".join(lines)


def verify(stage: pathlib.Path) -> dict:
    from safetensors import safe_open
    import torch
    adapter = stage / "adapter_model.safetensors"
    with safe_open(str(adapter), framework="pt") as f:
        keys = list(f.keys())
        dtypes = {str(f.get_slice(k).get_dtype()) for k in keys[:5]}
    head = torch.load(stage / "head.pt", map_location="cpu", weights_only=True)
    return {"adapter_tensors": len(keys), "adapter_sample_dtypes": sorted(dtypes),
            "head_entries": len(head) if hasattr(head, "__len__") else "n/a",
            "adapter_sha256": sha256(adapter), "head_sha256": sha256(stage / "head.pt")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staging", default="/tmp/hf-redo")
    ap.add_argument("--owner", default="macmacmacmac")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if not a.verify_only:
        require_frozen_chain()  # refuses upload without the full frozen-test evidence chain
    token = os.environ.get("HF_TOKEN")
    if not token and pathlib.Path(".env").is_file():
        for line in pathlib.Path(".env").read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
    api = None
    if not a.verify_only:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    for cfg, info in RESULTS.items():
        stage = pathlib.Path(a.staging) / cfg
        checks = verify(stage)
        print(cfg, "verify:", json.dumps(checks))
        (stage / "README.md").write_text(CARD.format(
            repo_name=info["repo_name"], task=info["task"], wall=info["wall"],
            train_file=info["train_file"], dev_file=info["dev_file"],
            dev_table=dev_table(info["dev"], info["base_dev_auc"])))
        (stage / "redo-provenance.json").write_text(json.dumps({
            "cfg": cfg, "host": "jev-train-1 (DigitalOcean gpu-h100x1-80gb, 159.65.192.220)",
            "wall_utc": info["wall"], "train_file": info["train_file"], "dev_file": info["dev_file"],
            "epochs": 1, "lr": 3e-5, "batch_questions": 32, "tok_budget": 4096,
            "trainable_params": 33776128, "dev_reeval": info["dev"], "checks": checks,
        }, indent=2) + "\n")
        if api is not None:
            repo = f"{a.owner}/{info['repo_name']}"
            api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
            api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="model",
                              commit_message=f"Redo FT checkpoint ({cfg}, epoch 1) from jev-train-1 H100 run 2026-11-03")
            print("uploaded https://huggingface.co/" + repo)


if __name__ == "__main__":
    main()
