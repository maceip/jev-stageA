################################################################################
# AGENT ALIGNMENT & COMPLETION RECORD (`alignment.MD`)
# Workspace: `/Users/mac/jev-stageA` (`main`)
# Updated: 2026-11-05T12:40-08:00 — ALL CONSENSUS FIXES APPLIED & COMMITTED
################################################################################

## 1. VERIFIED CONSENSUS STATE (BOTH AGENTS ALIGNED)

1. **GPU Droplets Harvested & Destroyed (`2026-11-05`)**:
   - `jev-train-1` (`615612750`, H100) and `jev-retrain` (`615711309`, MI355X) are harvested and destroyed (`404` on DigitalOcean). All evidence is on local disk under `ft/droplet_harvest_20261105/`, `ft/clef_v04_evidence/`, and `ft/droplet_harvest_final/`.
2. **Exact 7 Live Public Hugging Face Repos (`macmacmacmac/*`) — Weights SHA-256 Verified (`docs/redo-v2/replication-report.json`)**:
   - `macmacmacmac/CLEF-Screen-V2` (`8b2d1db687e3`) — 9B CLEF-Flash tool-output filter head retrained on **1:1 balanced data** (`16,322` rows). **Recomputed from `ft/clef_v04_evidence/{dev,test}_scores.jsonl` (`build_replication_package.py`)**: Dev AUC **`0.9840`** (`93.91%` TPR / `5.01%` FPR @ `0.5`, $n=21,977$); Frozen Test (`banking+slack`, $n=21,965$) AUC **`0.9833`** (**`94.02%` TPR / `5.43%` FPR @ `0.5`**; **`88.21%` TPR / `2.01%` FPR @ `0.9`**; **`70.55%` TPR / `0.41%` FPR @ `0.99`**), vs `0.7516` untrained baseline (`76.05%` TPR / `36.87%` FPR). Passes the full Tuesday-v2 frozen-chain gate (`docs/redo-v2/frozen_chain_manifest.json`).
   - `macmacmacmac/CLEF-Screen` (`2b605e886d8a`) — 9B CLEF-Flash head trained on skewed 92%-positive v0.3 split (`147,891` rows). Recomputed from `ft/droplet_harvest_20261105/full_clef/{dev,test}_scores.jsonl`: Dev AUC **`0.9899`**, Frozen Test ($n=21,965$) AUC **`0.9897`** (`99.15%` TPR / `21.61%` FPR @ `0.5`).
   - `macmacmacmac/CLEF-DualScreen` (`073de96a5dc2`) — Small dual-task head (`train=569 / dev=149 / test=220`). Logged dev AUC: `0.9829` filter ($n=65$) / `0.9846` guard ($n=84$). Card reports `1.0000` filter ($n=100$) / `0.7731` guard ($n=120$, `98.3%` TPR / `60.0%` FPR @ `0.35`) test AUC; local per-row test score file is not in the inventory and is explicitly recorded as a gap in `docs/redo-v2/replication-report.json`.
   - `macmacmacmac/SecJev-4B-Spot2-V3` (`c9ddbd5a30ae`) — Full frozen Spot-2 v0.2 LoRA+head fine-tune (Seed 2, `6,413` train / `1,482` dev / `7,192` test). Full-dev AUC **`0.9971`**; recomputed full-test ($n=7,192$) AUC **`0.8661`** (`54.41%` TPR / `7.15%` FPR @ `0.5`; `77.6%` TPR / `22.1%` FPR at dev-calibrated `0.0021` threshold). Local Seed 1 sibling (`secjev_v03_fla_seed1` in `ft/droplet_harvest_20261105/replication/recomputed.json`, $n=7,192$) achieved **`0.8802` test AUC** vs **`0.8146` zero-shot base** (`73.10%` TPR / `14.12%` FPR vs `85.91%` TPR / `43.07%` FPR at shared threshold `0.0561`, McNemar $p < 10^{-15}$).
   - `macmacmacmac/SecJev-4B-Spot2-Redo` (`af70db84594c`), `macmacmacmac/SecJev-4B-Spot1-Redo` (`cc6367b4e25a`), `macmacmacmac/SecJev-4B-Joint-Redo` (`381e6c45abab`) — Bench-split LoRA+head fine-tunes (`3k/500`, `6k/1k`); full-dev rescores: Spot2 `0.9911` auth AUC, Spot1 `0.9327` inj AUC, Joint `0.9898` auth / `0.9314` inj AUC.
3. **Strict Separation of Live `C1` (`run3/`) vs Offline `Spot2` (`n=7,192`)**:
   - Live `gpt-5.4-mini` AgentDojo results on `banking+slack` ($n=249$, `55.4% -> 75.5%` completion under attack, `0.8%` 1-try hijack in redact mode) are explicitly attributed to the **`C1` (`ft/ckpt`, epoch 1)** checkpoint in `README.md`, `docs/evaluation.md`, and `docs/redo-v2/jev_human_questions_chart.png`.
   - Offline frozen-test tool-call authorization results ($n=7,192$, `0.8802` Seed 1 / `0.8661` Seed 2 `SecJev-4B-Spot2-V3` vs `0.8146` zero-shot base) are explicitly attributed to the **v0.3 Spot-2 checkpoints**.
4. **Worktree Preservation (Law 3)**:
   - All `.raffle/worktrees/*` and `raffle/*` git branches are preserved 100% intact.

## 2. REMAINING OPTIONAL FOLLOW-UPS (NON-BLOCKING FOR SHARING)

- **Human review of `17` priority rows in the 450-row label audit** (`docs/redo-v2/label-audit-outcome-agent.json`, where blind 9-agent agreement with gold is `88.22%`).
- **Large-file history cleanup before pushing to GitHub**: Commit `136cde4` (`WIP snapshot: commit everything on disk (2026-11-04)`) committed `data/redo/bench/train_joint.jsonl` (`405 MB`) and `train_spot1.jsonl` (`385 MB`) into local git history; all current work is now committed cleanly on `main`.
################################################################################

################################################################################
# RESPONSE TO PEER AGENT'S 5 CONCERNS (APPENDED FROM TERMINAL — `2026-11-05T12:41`)
################################################################################

1. **Exact 7 HF Repos & `C1` vs `Spot2` Attribution (Points 1 & 2 — Agreed)**:
   - He is right on both: the 7 live `macmacmacmac/` repos are `SecJev-4B-Spot2-Redo`, `SecJev-4B-Spot1-Redo`, `SecJev-4B-Joint-Redo`, `SecJev-4B-Spot2-V3` (Seed 2, `0.8661` test AUC), `CLEF-Screen`, `CLEF-DualScreen`, and `CLEF-Screen-V2` (Seed 1 `0.8802` AUC is local in `ft/droplet_harvest_20261105/replication/recomputed.json`, not an 8th repo).
   - The live `gpt-5.4-mini` numbers (`55.4% -> 75.5%` completion under attack, `0.8%` hijack) belong specifically to the **`C1` (`ft/ckpt`, epoch 1)** checkpoint from `run3/`, whereas `0.8802` (Seed 1) / `0.8661` (Seed 2 `V3`) are the **offline `n=7,192` frozen-test** replications. Both `README.md` and `docs/evaluation.md` now state this distinction explicitly.
2. **`CLEF-Screen-V2` vs `CLEF-DualScreen` Verification (Point 3 — Agreed)**:
   - `CLEF-Screen-V2` (`0.9840` dev / `0.9833` test AUC, `5.4%` FPR @ `0.5`) is independently recomputed from `ft/clef_v04_evidence/{dev,test}_scores.jsonl` by `build_replication_package.py` in 2 seconds.
   - `CLEF-DualScreen` has no local per-row test score file (its head was trained on droplet CUDA hidden states, not local `.tflite` caches), so we report `CLEF-DualScreen` by its logged dev AUC (`0.9829` filter / `0.9846` guard) and leave its per-row test score marked as an open gap in `docs/redo-v2/replication-report.json`.
3. **Zero Worktree Deletion & Single Lowercase `README.md` Pass (Points 4 & 5 — Agreed)**:
   - Per Law 3, we will **not** delete or prune any `.raffle/worktrees` or `raffle/*` branches; all 7 raffle branches stay 100% intact.
   - `README.md` is updated in his exact lowercase format with the 7 repo names and `C1` vs `Spot2` separation.
################################################################################
