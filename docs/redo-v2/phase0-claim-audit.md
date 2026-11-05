# Phase 0 — claim and artifact audit for the v2 redo

Status: execution record, 2026-11-03 UTC. This file freezes what the v2 redo may
and may not inherit from the previous CLEF and SecJev work. It is not a results
announcement.

## Recovery snapshot

- Repository: `/Users/mac/jev-stageA`
- Branch at audit: `main`
- HEAD at audit: `84e7c87bef635b6acca6f3f984c12de0018c94f9`
- Tracked-worktree recovery ref: `refs/tbh/recovery/redo-v2-before-20261103T075756Z`
- Recovery object: `604eccdf3484bde8c82c614ec36d9ea7903e075b`
- The snapshot was created with `git stash create` plus `git update-ref`; the
  working tree and index were not changed by snapshot creation.
- At snapshot time: 25 porcelain status entries, 13 tracked modified files, and
  1,203 untracked files when ignored files are excluded. Untracked training
  outputs are therefore not protected by the recovery ref and must be treated
  as fragile local evidence.

## Claim matrix

| Claim or implied claim | Status | Evidence checked in Phase 0 | Required disposition |
|---|---|---|---|
| CLEF work fine-tuned the clef-flash backbone | Contradicted as a description of `ft_clef.py` | Backbone runs through the frozen LiteRT host; only the PyTorch `JointSchemaHead` trains on cached hidden states. | Describe v2 CLEF work as head training on a frozen backbone unless a future run actually trains backbone weights. |
| CLEF head has 60.8M trainable parameters | Contradicted | `ft_clef.py` docstring says 60.8M; run log records 121,762,820 trainable parameters. | Correct the docstring and assert the parameter count from the saved state dict in v2. |
| Original CLEF test metrics came from epoch 6 | Contradicted | `ft/clef_ft_run/final_results.json` stamps epoch 6/step 216; `checkpoints/clef-flash-LiteRT-ft/provenance_ft.json` records epoch 5/step 180. | Retire the epoch-6 label. V2 evaluation must stamp the actual best checkpoint and fail if no best checkpoint is recorded. |
| Original CLEF head is strong at deployed thresholds | Contradicted | Original aggregate test record: Spot-1 AUC 0.982 but TPR@0.99 = 0.02; Spot-2 AUC 0.6322 and FPR@0.35 = 0.8167. No per-row score file exists for independent recomputation. | Do not reuse as a deployable guard or headline. V2 must report deployed operating points and trivial baselines from per-row scores. |
| CLEF artifacts are versioned with the repository | Contradicted | `git ls-files` returns zero paths containing `clef`; `ft_clef.py` and CLEF run outputs are untracked. | V2 training code, configs, manifests, and small provenance files must be tracked before training. Large weights need an explicit artifact policy. |
| Local CLEF bundle supports complete multimodal deployment | Unsupported locally | Local base and FT directories contain the head, tokenizer, tables, and host code, but the FT directory has no updated `contract.json`; prior review found only the L4096 text graph locally and an F32 FT head against a BF16 contract. | Either obtain and hash-verify the complete bundle or publish an explicit reduced text-only contract. No multimodal claim before load tests pass. |
| C1 was trained on 9,974 train / 915 dev / 7,588 test questions | Not reproducible from current artifacts | `ft/ckpt/provenance.json` states those counts. Current files count 5,147/355/5,427 (`ft_*.jsonl`) and 2,626/255/2,134 (`ft_c_*.jsonl`). | Treat the provenance counts as unreconciled. V2 must regenerate counts from frozen manifests and publish a correction if the old counts remain unreconcilable. |
| C1 live result 75.5% completion / 0.8% hijacked and adaptive 53–58% | Previously recomputed; not regenerated in Phase 0 | The earlier CLEF review recomputed these values from `run3/live` and `run3/adaptive_mini`. Phase 0 did not rerun the full recomputation. | Reuse only as historical context. Regenerate from raw artifacts in the replication phase before any v2 comparison or republication. |
| SecJev-4B reproduces published split accuracy through `verify_corpus.py` | Not rerun in Phase 0 | README presents the result as reproducible, but the external SecJev corpus/work directory was not restored or executed during Phase 0. | Restore the exact corpus and environment, rerun, and record hashes before repeating the claim. Absence of the corpus is a blocker, not a pass. |
| Joint CLEF/SecJev fine-tuning reduces false alarms on indirect-reference tasks | Unsupported | Current README research-direction text states the benefit, while CLEF deployed-threshold artifacts do not support a general false-alarm claim and no dedicated indirect-reference per-row artifact was audited. | Retire the wording until v2 evaluates a dedicated hard-negative indirect-reference set. |
| FASTJEV 2-epoch smoke head is deployable | Contradicted | `ft/fastjev_run/final_results.json`: Spot-1 AUC 0.9612, TPR@0.99 = 0; Spot-2 AUC 0.5978, FPR@0.35 = 1.0. Per-row files exist (149 dev, 220 test), so the weakness is reproducible rather than hidden. | Keep only as pipeline evidence. Do not deploy or use as the v2 starting checkpoint without a new dev-selected run. |

## Current split counts

| File | Rows |
|---|---:|
| `ft/ft_train.jsonl` | 5,147 |
| `ft/ft_dev.jsonl` | 355 |
| `ft/ft_test.jsonl` | 5,427 |
| `ft/ft_c_train.jsonl` | 2,626 |
| `ft/ft_c_dev.jsonl` | 255 |
| `ft/ft_c_test.jsonl` | 2,134 |
| `ft/clef_ft_run/clef_train.jsonl` | 569 |
| `ft/clef_ft_run/clef_dev.jsonl` | 149 |
| `ft/clef_ft_run/clef_test.jsonl` | 220 |

The 569-row CLEF training file is a capped derivative, not a canonical training
corpus. V2 training must not silently inherit that cap.

## Secret-name inventory

Values were not printed. Relevant nonempty variable names observed:

- `/Users/mac/labrea/agent-relay-surface/.env`: `TPESAFE_API_KEY` (spelled
  without `Y`)
- `/Users/mac/jevalin/.env`: `JEV_API_KEY`
- `/Users/mac/jev-stageA/.env`: `HF_TOKEN` only
- `/Users/mac/jev-ultrafast/.env`: missing

`jev_ultrafast/model.py` requires canonical `TYPESAFE_API_KEY` and optionally
uses `TYPESAFE_MODEL`; `TYPE_TEXT` requires `TEXT_MODEL_API_KEY`. Before any
live ultrafast arm, execution must confirm which existing variable is intended
and map canonical names in the runtime environment without printing or copying
secret values into tracked files.

## Phase 0 exit decision

Phase 0 is complete for planning the rebuild: the recovery ref exists, the
claim matrix separates reproduced history from contradicted and unsupported
claims, and the fragile untracked state is explicit. Training remains blocked
until Phase 1 produces a licensed source registry and Phase 2 freezes a
canonical dataset manifest.
