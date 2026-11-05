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

## SecJev Spot2 v0.3 seed2 FT (jev-retrain) — uploaded 2026-11-04

macmacmacmac/SecJev-4B-Spot2-V3, commit c9ddbd5a30ae (initial
9b703ccdb610), public, 44+ files. Full frozen Spot-2 v0.2 data (6,413/1,482), 3 epochs, batch
16, bf16, seed 20261102, finished 2026-11-04T20:41Z. Epoch dev AUCs
0.9954/0.9966/0.9967 (training log); independent local rescore of
epoch 3 on full dev: AUC 0.9971 (agreement check passed). Droplet
script-run test (banking+slack, 3,044 rows): AUC 0.8236; the
dev-calibrated threshold (dev FPR<=1% -> 0.0021) transfers badly
(test FPR 0.2295, TPR 0.691; at 0.5 TPR 0.425/FPR 0.076; balanced-acc
95% CI [0.714, 0.751]) — disclosed on the card as the headline
caveat. Adapter/head droplet==local==HF-LFS SHA-256
(2e4003b9…/c176f8d0…). Full 7,192-row test scores (regenerated by
peer retrain at 20:48Z) analyzed 2026-11-04: AUC 0.8661, at
dev-calibrated threshold TPR 0.776/FPR 0.221, at 0.5 TPR 0.544/FPR
0.072, balanced-acc CI [0.768, 0.777]; card + full score/analysis
files updated on HF same day. Seed-20261101 sibling (dev AUC 0.9926)
exists on the droplet, unpublished; with 85 dev positives the
seed gap is within noise. Uploaded on user direction ahead of the
Tuesday-v2 frozen-chain gate; per-row dev rescore + droplet test
scores + frozen analysis ship inside the repo.

## CLEF proper-protocol FT (jev-retrain) — uploaded 2026-11-04

macmacmacmac/CLEF-DualScreen, commit 073de96a5dc2, public, 51 files.
Original small ft_clef protocol (train 569/dev 149/test 220, both
tasks), 3 epochs, lr 1e-4, seed 20261101, epoch 3 selected by
combined dev AUC. Training completed 21:36Z; the process was then
killed by SSH session-scope deactivation during its own test eval
(no final_results.json). Independent local LiteRT rescore of the
saved head on the 220-row test: tool_output_filter AUC 1.0000
(n=100), action_authorization_guard AUC 0.7731 (n=120; @0.35 TPR
0.983/FPR 0.600) vs dev guard AUC 0.9846 — guard domain collapse
disclosed on the card. Head droplet==local==HF-LFS SHA-256
c783389e32f94913…; 122 tensors, 121,762,820 params, BF16.
Uploaded on user direction ahead of the Tuesday-v2 frozen-chain
gate (disclosed in redo-provenance.json). CLEF seed2 (frozen-v0.3)
was aborted by its owner at step ~15,600/19,293 on 2026-11-04 and
its output directory deleted; only its epoch-boundary head remains
on the droplet, unpublished. Droplet cache (599 GB) pruned by peer;
disk back to 595 G free.

## CLEF v04 fixed-data FT (jev-retrain) — uploaded 2026-11-05

macmacmacmac/CLEF-Screen-V2, commit 8b2d1db687e3 (initial upload
e4ba6c4fe0d7, then stray __pycache__ removed), public, 29 files.
Fixed-data retrain: train rebuilt 1:1 balanced (16,322 rows;
8,232 pos / 8,090 neg; SHA-256 e3916d4946cbe6e4…) replacing the
92%-positive frozen-v0.3 train; +86 technique-catalog rows, +184
synthetic tool-spoof rows; pos_weight 1.0 (was 3.0); frozen dev/test
untouched. ft_clef.py --cache-only, 3 epochs, lr 3e-5, batch 16,
seed 20261101, TRAIN_EXIT=0 2026-11-05T05:59Z. Epoch 3 selected on
dev AUC only (0.9635/0.9811/0.9840; untrained-head baseline 0.7516).
Dev AUC 0.9840 (TPR@0.5 0.939 / FPR@0.5 0.050); frozen test once:
AUC 0.9833 (TPR@0.5 0.940 / FPR@0.5 0.054); independently recomputed
from the shipped per-row score files, exact match. Head
droplet==local==HF-LFS SHA-256 c7375d6f2044ab67…; backbone == local
base 43a638f2976f…; 122 tensors, 121,762,820 params, BF16 native.
Fresh-process LiteRT load-check of the staged artifact passed
(end-to-end scoring, 6 bench rows). Filter task only (guard n=0);
dev/test ~92% positive (~1,700 negatives) caveat on the card.
Uploaded on user direction ahead of the Tuesday-v2 frozen-chain
gate (disclosed in redo-provenance.json). Vs CLEF-Screen seed1 (dev
AUC 0.9899, FPR@0.5 ~0.22), v04 trades 0.006 dev AUC for ~4x lower
FPR@0.5 — the balanced-data fix worked as intended.

## Phase 7 replication + label audit + droplet shutdown — 2026-11-05

Replication (`build_replication_package.py` ->
`docs/redo-v2/replication-report.json`): every published weight file
in all 7 repos above hash-matches a local byte-identical copy against
live HF LFS metadata. Headline AUCs recomputed exactly from per-row
score files: CLEF-Screen-V2 dev 0.9840 / test 0.9833, CLEF-Screen dev
0.9899 / test 0.9897, SecJev-4B-Spot2-V3 full test 0.8661 (7,192 rows).
SecJev Redo trio recomputed on predeclared bench draws (published
full-dev rescores remain the headline; bench-draw numbers and the
untrained-baseline draws are in the report with draw scope stated).
Open gap, recorded in the report: CLEF-DualScreen test AUCs have no
local per-row score file (head identity proven by hash only).
Tuesday-v2 frozen-chain gate now PASSES for the CLEF-Screen-V2 chain
(`docs/redo-v2/frozen_chain_manifest.json`: dataset manifest ->
training log -> checkpoint manifest -> frozen test scores -> analysis
-> replication report -> load-check record, all hash-verified).
Load-check rerun on the local v04 checkpoint reproduced the pre-upload
scores exactly (`ft/clef_v04_evidence/load_check.txt`).

Label audit (`docs/redo-v2/label-audit-outcome-agent.json`): 450-row
blind packet adjudicated by 9 blind AI subagents (50 rows each, key
opened only after all verdicts were written) — NOT human adjudication;
the human gate remains open. Agreement with gold 0.8822 (Wilson 95%
[0.8492, 0.9088]) vs 0.5511 for the earlier deterministic
imperative-marker pre-pass. Per kind: direct_prompt_injection 0.9733,
action_authorization_guard 0.9067, tool_output_filter 0.7667. 53
disagreements listed with reasons (17 high-confidence — priority list
for human review). Blind packet text was capped at 1500 chars; rows
whose decisive content fell beyond the cap were judged low-confidence
and are marked. Filled packet:
`data/external/redo_v2_out/label_audit_filled_agent.jsonl`.

Droplet `jev-retrain` (615711309, gpu-mi355x1-288gb-spot) destroyed
2026-11-05 via DO API after harvest; deletion verified (GET returns
404). Harvest: `ft/clef_v04_evidence/` +
`ft/droplet_harvest_final/small/` (496 evidence files) on top of the
earlier `ft/droplet_harvest_20261105/`. The 251 GB v04 encode cache
and unselected epoch heads were not harvested (regenerable /
unselected; every published weight was already local). No GPU droplet
remains on the account for this project; unrelated droplets untouched.
