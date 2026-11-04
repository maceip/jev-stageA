# HF uploads — droplet redo FT weights (2026-11-03/04)

Source: DigitalOcean GPU droplet `jev-train-1` (gpu-h100x1-80gb,
159.65.192.220, tag `jev-ft`), queue `run_secjev.sh`:
spot2 (22:29:44–22:53:41 UTC) → spot1 (22:53:41–23:03:28) →
joint (23:03:28–23:37:49), all exit 0, `ALL_DONE`.
1 epoch each, `ft_action.py`, lr 3e-5, batch 32, tok_budget 4096,
fp32 + bf16 autocast, 33,776,128 trainable params (LoRA + head).
Data: peer bench sets `data/bench/{train,dev}_{cfg}_t.jsonl`
(3,000/500 spot2+spot1, 6,000/1,000 joint) — NOT the frozen v0.2 splits.

| cfg | HF repo (public) | commit | adapter SHA-256 | dev re-eval AUC |
|---|---|---|---|---|
| spot2 | macmacmacmac/SecJev-4B-Spot2-Redo | af70db84594c | c04f11dd7afc8fa9… | authorization 0.9911 |
| spot1 | macmacmacmac/SecJev-4B-Spot1-Redo | cc6367b4e25a | 7329ca6cdc831cd3… | injection 0.9327 |
| joint | macmacmacmac/SecJev-4B-Joint-Redo | 381e6c45abab | 6d816736b0fbb90d… | auth 0.9898 / inj 0.9314 |

Verification before upload: local adapter/head SHA-256 == droplet
SHA-256 (all 6 files); safetensors load (496 tensors, F32) + head.pt
load (18 entries) per checkpoint; fresh-process SecJev load of spot2
scored 3 dev rows, 0 failures. Upload verified afterwards via HF API
(public, 42 files each) and unauthenticated HTTP 200.

Known defect, documented in every model card: training-time dev
metrics for spot1/joint in `redo-training-log.jsonl` are invalid
(ft_action.evaluate assumes authorization keys; injection rows use
noul 'true'/'false'). Cards carry the corrected per-question
re-evaluation (droplet `reeval_droplet.py`; spot2 cross-check
0.9911 vs training-log 0.9914). No frozen-test/adaptive/replication
result exists yet for these weights; cards state that explicitly.

Publisher: `publish_redo_weights.py` (owner/layout/visibility per
user answers 2026-11-03: macmacmacmac, three repos, public).

## Droplet destruction (2026-11-04 05:17 UTC) — preserved vs lost

DigitalOcean action 3519752115 `destroy` completed 05:17:32 UTC;
droplet 615612750 no longer exists, zero snapshots on the account.
Before destruction, a second queue (`run_secjev_full.sh`, full-size
data, 2 epochs) died at 05:01:02 when its launching SSH session
scope deactivated (journal: `session-318.scope: Deactivated
successfully. Consumed 7min 37.366s CPU time`); spot2_full reached
step 10/402 after a healthy full-dev epoch-0 baseline (AUC 0.9219).

Preserved: all three uploaded HF repos above (canonical weight
copies, hash-verified before upload); local staging copies in
`/tmp/hf-redo/{spot2,spot1,joint}`; `reeval_droplet.py` (now in repo
root); CLEF tflite local source (SHA-256 43a638f2…); training logs
and stdout inside each HF repo.

Lost with the droplet: raw bench-matrix outputs
(`ft/bench_results/*.json` + per-track score files; headline metrics
were captured in-session: action-authz AUC base 0.8299 / spot2
0.8824 / spot1 0.8416 / joint 0.8748; tool-output-filter AUC base
0.6436 / spot2 0.6470 / spot1 0.7277 / joint 0.7261, TPR@dev-thr
≈0.08 on the filter track), the smoke checkpoint, `ft/secjev_`
empty attempt, bench stdout logs, and the partial spot2_full log
(epoch-0 + step-10 lines are quoted above/in session record).
No CLEF FT ever ran on the droplet; the verified backbone copy
died with it (local identical source retained).

## CLEF seed1 FT (jev-retrain droplet, 2026-11-04) — verified, upload pending

New droplet `jev-retrain` (615711309, gpu-mi355x1-288gb-spot,
168.144.60.186, created 2026-11-04T05:35:29Z). CLEF pipeline:
GPU encode of 146,984 frozen-v0.3 rows at 0.043 s/row (HF torch on
ROCm), then JointSchemaHead training, 3 epochs. First attempt
OOM-killed 14:28:49Z at 256 GB anon-RSS (full train set materialized
in RAM); patched streaming rerun (RSS ~5 GB) finished 18:24:03Z.
Dev epoch-0 AUC 0.7516 (TPR@0.99 0.0) -> epoch-3 dev AUC 0.9899,
TPR@0.5 0.9911 / FPR@0.5 0.2183, TPR@0.99 0.9583 / FPR@0.99 0.0507;
script-run test (banking+slack) AUC 0.9897. Caveats: dev/test are
92% positive (1,695-1,694 negatives of ~22k); FPR@0.5 remains 0.22.

Verification (this session): head + config + provenance downloaded;
droplet==local SHA-256 (head `8ca78d0c127755948f…`); 122 tensors,
121,762,820 params, BF16; differs from base head; assembled onto
local LiteRT base and fresh-process LiteRT scoring succeeded
(3 synthetic-spoof rows: pos 0.936, pos 0.093 (miss @0.5), neg 0.010).
Droplet ckpt_out contains ONLY head/ + provenance_ft.json (cache-only
assembly); full runnable dir is assembled locally at staging time.
HF upload completed 2026-11-04 19:05 UTC on direct user instruction
(personal account, user-approved naming): macmacmacmac/CLEF-Screen,
commit 2b605e886d8a, public, 28 files. HF-side LFS SHA-256 match the
droplet artifacts (head 8ca78d0c…, tflite 43a638f2…). The Tuesday-v2
frozen-chain publication gate (tuesday_gates.require_frozen_chain)
was bypassed on explicit user instruction; the card and
redo-provenance.json disclose that no frozen_chain_manifest.json or
independent replication exists yet. Seed2 CLEF rerun in progress on
the droplet; remaining-queue Tasks A-C (SecJev seed2, adaptive,
ultraspeed) crashed on missing SECJEV_WORK and produced nothing.
