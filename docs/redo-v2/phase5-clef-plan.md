# Phase 5 CLEF redo — predeclared plan (2026-11-03)

## Contract corrections vs the published v1 run

- Name the trained object exactly: `JointSchemaHead` only, on frozen
  `Cloudflare/clef-flash` LiteRT hidden states. No LoRA/backbone claim.
- Report the true parameter count from the saved state dict.
- Export must be self-contained and loadable by the deployment scorer
  (v1 defect: F32 head failed BF16 load; missing backbone link).
  Acceptance test: fresh-process load + score of a fixed probe set
  before any metric is reported.
- Epoch selection on dev only; epoch labels in artifacts must match
  the epoch that produced them (v1 defect: epoch-5 head labelled 6).

## Encode budget (decision rule, fixed before measuring)

`ft_clef.py` encodes once per row through Metal GPU LiteRT and caches
hidden states to disk (`encode_and_extract`), so encode cost is paid
once across epochs. Prior measurement: ~12 s/row median at full text
(~650 h for 195k rows — rejected).

1. After SecJev training frees the GPU, measure encode seconds/row on
   a fixed 50-row stratified sample from `clef_spot1v2_train.jsonl`
   (seed 20261103), at ft_clef's native truncation.
2. Cap total encode wall time at 12 h. Subsample size =
   floor(12 h / measured s-per-row), stratified by kind × label with
   seed 20261103, drawn from train; dev/test stay full-size only if
   their encode fits a separate 4 h cap, else the same stratified
   rule applies and the report must state evaluated-N.
3. The subsample manifest (row_ids + measured rate + caps) is written
   before training starts. No post-hoc resizing based on results.

## Training and evaluation

- Train the head with dev-only early selection; record every epoch's
  dev AUC, TPR/FPR at predeclared thresholds, calibration (ECE/Brier).
- Baselines reported alongside: keyword scorer (`redo_eval`) and the
  base (untrained-head) CLEF on identical rows — v1 failed here
  (Spot-1 TPR@0.99 = 0.02; Spot-2 FPR@0.35 = 0.82).
- Frozen test evaluation happens once, after selection, in Phase 6.
