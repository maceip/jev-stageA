# Phase 7 replication package — checklist (2026-11-03)

Every published number must map to: dataset manifest hash → training
log → checkpoint hash → frozen score file → analysis output. A claim
without the full chain does not ship.

## Required artifacts

- [x] Frozen dataset manifests with SHA-256 per split
      (`frozen-manifest-v2.json`, `frozen-manifest-spot2-v2.json`).
      Re-verified 2026-11-03: all 12 frozen files (v0.1 + v0.2)
      hash-match their manifests; 29-artifact inventory in
      `artifact-hashes.json` (`build_artifact_hashes.py`).
- [ ] Dataset audit (`dataset-audit.json`) + human label-audit outcome
      (450-row blind packet; agreement rate reported, disagreements listed).
      PARTIAL 2026-11-05: blind AI adjudication done (agreement 0.8822,
      53 disagreements listed, `label-audit-outcome-agent.json`); HUMAN
      adjudication still outstanding — 17 high-confidence disagreements
      are the priority review list.
- [ ] SecJev: `ft/redo_secjev_run1/log.jsonl` (dev AUC per epoch),
      selected epoch + selection rule, checkpoint SHA-256, load-check
      record (fresh process, deployment scorer).
- [ ] CLEF: encode-budget manifest (measured s/row, caps, row_ids),
      training log, selected epoch, self-contained BF16 export +
      load-check record, true parameter count from state dict.
- [ ] Phase 6 frozen reports (dev-calibrated threshold, test metrics,
      suite-clustered CIs) for SecJev and CLEF; keyword + untrained
      baselines on identical rows.
- [ ] Adaptive report limited to the 275-row output-bearing subset,
      labelled as such.
- [ ] Ultrafast: eval-only (no fine-tuning). Live browser arm blocked
      until browser-harness daemon/Chrome remote debugging is up
      (`uv run browser-harness --doctor` FAIL on 2026-11-03).
- [x] Independent replication: second run of analysis from frozen
      score files; byte-identical metrics or explained deltas.
      DONE 2026-11-05 (`build_replication_package.py`,
      `replication-report.json`): all 7 HF weight files hash-match local
      copies; CLEF-Screen-V2, CLEF-Screen, SecJev Spot2-V3 headline AUCs
      recomputed exactly from per-row scores. Explained deltas/gaps:
      SecJev Redo trio local scores are bench draws (not full-dev);
      CLEF-DualScreen has no local per-row test score file. Clean-checkout
      commit pinning was not used; rerun is script + frozen files.
- [x] Frozen-chain gate: `frozen_chain_manifest.json` written for the
      CLEF-Screen-V2 chain; `tuesday_gates.require_frozen_chain()` passes
      (2026-11-05). Droplet `jev-retrain` destroyed after harvest
      (verified 404); see `hf-uploads.md` Phase 7 section.
- [ ] Corrected publication text: v1 errors named explicitly
      (head-only training, 121.8M not 60.8M params, epoch mislabel,
      threshold failures, deployment-load failure). Publication
      posture (visible correction vs withdrawal) is the user's call.

## Environment record

- Machine: Apple M5 Max, 18 cores, 128 GiB RAM, MPS (no CUDA).
- SecJev training: fp32 weights, bf16 autocast, gradient checkpointing.
- Seeds: dataset 20261103; ft_action default 20261101.
