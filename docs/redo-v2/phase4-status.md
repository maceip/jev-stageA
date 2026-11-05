# Phase 4 status — 2026-11-03

## MPS compute probe (completed, exit 0)

`ft_action.py` device-ported copy trained the SecJev-4B adapter on Apple MPS
(fp32 weights, bf16 autocast, gradient checkpointing 1): 64-row train,
32-row dev, 1 epoch, epoch-1 checkpoint saved (150 MB output dir,
33,776,128 trainable params). Dev rows were authorized-only in the probe
sample, so probe AUC is not meaningful; the probe proves the training
path (load → encode → backward → save → evaluate) runs on this Mac
(128 GiB RAM, MPS). The maintained `ft_action.py` now has the same
`--device` port (cuda default, mps fallback) and device-bound autocast.

## Real SecJev redo run (restarted with batched eval)

First launch was terminated after 30 min: the epoch-0 dev eval
(1,482 single-row MPS forwards, >1.1 s/row) produced no record.
`ft_action.evaluate()` now uses length-sorted token-budget
microbatches via `forward_batch` (same score statistic) and logs
`eval_seconds`. The change is probe-validated before relaunch.

Profiling then showed batching/dtype are not the wall: the HF
GatedDeltaNet torch fallback (no flash-linear-attention on MPS)
runs ~614 padded tok/s bf16 (fp32/bf16/fp16 within 15%), so
full-data training projected to 10–16 h. Predeclared subsample
protocol (`build_redo_subsamples.py`, seed 20261103, fixed before
any result): train on all 430 positives + 1,290 negatives (1:3,
suite-proportional), training-time dev signal on 512 rows (all 85
positives + 427 negatives); final epoch selection re-scores the
FULL dev split via `eval_secjev_frozen.py`. Relaunched run:
bf16 weights (deviation from fp32 reference, recorded),
2 epochs, 108 steps. First relaunch (gradient checkpointing off)
recorded the epoch-0 untrained baseline — dev-sub AUC 0.9175,
TPR@0.5 0.788, FPR@0.5 0.063, TPR@0.9 0.282 — then died with MPS
OOM (115.7 GiB) in early training. Final relaunch uses gradient
checkpointing on (probe-proven memory profile); dev-sub eval cost
674 s per epoch boundary.

Second death (same day): the checkpointing-on run reached step 10
(2.24 min/step) then OOM'd with only 22.2 GiB own MPS allocation —
160.5 GiB of the pool was held by a peer session's parallel SecJev
fine-tune (`ft_action_redo.py`, PID 10182, `ft/redo_secjev_spot2`,
94 steps, ETA ~3.5 h from its step-10 record). Peer session
(apricot-chromosphere) notified; this run relaunches only after
PID 10182 exits (monitor armed). System swap was 28.1/29.7 GiB —
never run two MPS trainings concurrently on this machine.

Peer PID exited later the same day (its log stops at step 40/94,
no checkpoint — peer notified). Third relaunch attempt collided
again: the peer relaunched (PID 46966, MPS watermark 1.7) seconds
before this run; this run was stopped at ~10 min (epoch-0 baseline
reproduced bit-identically a third time, AUC 0.9175092988014878)
to protect system stability. Sequencing rule proposed to the peer
session: exclusive MPS lock via /tmp/mps_training.lock, peer holds
the current slot, this run takes the next ~4 h slot on peer exit
(monitor armed on PID 46966).

- Inputs: `data/external/redo_v2_out/ft_spot2v2_{train,dev,test}.jsonl`
  from canonical Spot-2 v0.2 (train 6,413/430 unauthorized,
  dev 1,482/85, test 7,192 untouched until Phase 6).
- Command (current run): `SECJEV_WORK=/Users/mac/secjev-4b-work
  .venv/bin/python ft_action.py --train
  data/external/redo_v2_out/ft_spot2v2_train_sub.jsonl --dev
  data/external/redo_v2_out/ft_spot2v2_dev_sub.jsonl --epochs 2
  --batch 32 --checkpointing 1 --device mps --dtype bf16
  --out ft/redo_secjev_run1`
- Checkpoint/epoch selection is dev-only by construction (`evaluate()`
  logs dev AUC per epoch to `ft/redo_secjev_run1/log.jsonl`).
- macOS has no `setsid`; the run is a runtime-managed process. If it
  dies, rerun the command above after removing the output directory
  (`ft_action.py` refuses an existing `--out`).

## Benign expansion + label audit artifacts

- `canonical_benign_agentdojo.jsonl`: 113 new benign tool outputs from
  no-attack episodes (1,143 duplicates dropped, 57 already frozen).
  Verdict: run3 benign episodes cannot close the ~11.5k-vs-184k
  imbalance; more benign episodes or BIPIA clean contexts are needed.
- Blind human label-audit packet: 450 rows (75 per kind × label),
  `label_audit_blind.jsonl` + `label_audit_key.json` in the gitignored
  output directory. Human adjudication is the outstanding Phase-3 gate.

## Phase 6 tooling prepared

- `eval_secjev_frozen.py`: two-stage frozen eval. `score` writes
  per-row P("2") + 0.5·P("1") (same statistic as ft_action dev eval);
  `analyze` calibrates the threshold on dev only (FPR budget 0.01),
  reports test metrics + suite-clustered bootstrap CI via `redo_eval`.
  Analyze stage smoke-verified on synthetic fixtures (not model scores).
- `build_adaptive_variants.py`: 4 deterministic text transforms
  (zerowidth, homoglyph, quote, case) over test positives. Coverage
  finding: only 275 of 1,654 test positives (2,023 of 7,192 test rows)
  carry tool-output text in SecJev state; the rest carry provenance
  only ({function, provenance}), so adaptive text results apply to the
  275-row output-bearing subset and must be reported as such.
- Scoring cost forecast: single-row MPS forwards measure ≥0.3 s/row in
  the training-time dev eval, so a full 7,192-row test score run is
  roughly 1–3 h per checkpoint.

## Phase 5 prepared (not launched)

`clef_spot1v2_{train,dev,test}.jsonl` inputs for `ft_clef.py` are built.
Encode remains the cost gate (LiteRT ~12 s/row median measured); launch
only with a predeclared subsample/truncation budget and after the
SecJev run frees the machine.
