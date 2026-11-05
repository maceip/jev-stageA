# Phase 3 — harness hardening status and concrete blockers

Status date: 2026-11-03 UTC. Phase 3 is partially complete. The new
evaluation harness exists and is tested; training phases 4–5 remain blocked
by named prerequisites below, each verified rather than assumed.

## Completed

- `redo_eval.py`: per-row metrics (AUC, TPR/FPR, balanced accuracy, Brier,
  ECE), dev-only threshold calibration, participant-cluster bootstrap CIs,
  checkpoint+code+data+config cache keys, and a keyword baseline CLI.
  Empty or single-class eval arms raise `ValueError` instead of reporting a
  number. Coverage: `tests/test_redo_eval.py` (suite total across redo test
  files: 26 passed).
- Keyword baseline on the frozen canonical dataset
  (`baseline-keyword-report.json`): tool-output-filter AUC 0.4984 dev /
  0.5011 test — chance level. LLMail attacks rarely contain classic
  injection keywords, so the corpus is genuinely hard for lexical methods.
  Direct-injection dev/test contain no negatives (all SaTML non-attacks
  landed in train; Gandalf is one component in test), so balanced metrics
  are undefined there and the harness correctly emits none.
- Existing C1 checkpoint scored on an ordered 50-positive/50-negative
  LLMail dev sample through `secjev_scorer` (kev-torch-mps, ~1.5 s/row):
  AUC 0.7604, balanced accuracy 0.63 at threshold 0.5, FPR 0.70. Ordered
  (non-random) sample; per-row scores in
  `data/external/redo_v2_out/scores_dev_c1_sample.jsonl`. C1 calibration does
  not transfer to LLMail distribution at 0.5.
- CLEF deployment contract, verified by loading through `secjev_scorer`:
  base `checkpoints/clef-flash-LiteRT` loads (litert-gpu) and scores a
  blatant injection at P(true) = 0.968. The F32 fine-tuned head in
  `checkpoints/fastjev-clef-ft` does **not** load
  (`dtype F32, expected BF16`). A BF16 conversion
  (`checkpoints/fastjev-clef-ft-bf16`, 122 tensors cast, backbone graph
  linked from the base bundle) loads and scores the same request at
  P(true) = 0.881. Two save-path defects are now proven: fine-tuned
  checkpoint dirs lack both `contract.json` and the backbone `.tflite`
  graph, so a saved FT checkpoint is not self-contained.

## Spot-2 label-rule rebuild (completed after the section above)

- Corrected rule in `redo_v2.authorization_label`: unauthorized requires an
  attacker argument value in the proposed call. The legacy rule is kept as
  `legacy_authorization_label` only to quantify impact. Across 3,990 run3
  episodes: 474 points stay unauthorized, 11,317 stay authorized, and **99
  legacy-unauthorized points become ambiguous** under the corrected rule.
- Frozen Spot-2 artifact `redo-v2-canonical-spot2-v0.1`
  (`frozen-manifest-spot2.json`, builder `build_redo_spot2.py`): 4,969
  deduped rows, suite-disjoint (banking+slack test 2,099; workspace+travel
  train 2,333 / dev 537). Hashes and counts re-verified independently.
- Hard limit found in v0.1: after dedupe, workspace+travel contain only
  **22 unauthorized rows** (dev: 1). Superseded by v0.2 below.
- **Spot-2 v0.2 built** (`frozen-manifest-spot2-v2.json`, builder
  `build_redo_spot2_v2.py`): legacy derived rows were relabelled by
  reconstructing AgentDojo ground truth from each row's suite/task ids.
  The reconstruction method first validated at **100% agreement
  (4,969/4,969)** against episode-derived v0.1 labels, and the recomputed
  legacy rule reproduced every stored legacy label (zero mismatches).
  Result: 15,087 rows (train 6,413 with 430 unauthorized; dev 1,482 with
  85 unauthorized; test 7,192). 422 legacy-unauthorized points became
  ambiguous under the corrected rule. Hashes and counts independently
  re-verified. Remaining caveat: ground truth was reconstructed with an
  empty injections dict; the 100% episode-side agreement is the evidence
  that ground-truth calls do not depend on the injected environment here.
- The BF16-converted smoke head (`checkpoints/fastjev-clef-ft-bf16`, with
  the backbone graph linked in) loads through `secjev_scorer` and scores a
  blatant injection at P(true) = 0.881 (base head: 0.968). The dtype/graph
  contract defects are therefore fixable at save time; `ft_clef.py` must
  export BF16 and a self-contained bundle in the redo.

## Blockers, verified

1. **SecJev training compute.** `ft_action.py` hardcodes CUDA
   (`load("cuda", ...)`, `torch.autocast("cuda", ...)`). This Mac has MPS
   only (`torch.cuda.is_available() == False`); `secure.build` has no
   `nvidia-smi` and no torch. No reachable CUDA host is configured.
   SecJev *inference* works on MPS; LoRA *training* needs either a GPU host
   or an MPS/MLX port of `ft_action.py` with a numerics-equivalence check.
2. **No Spot-2 training data in canonical v0.1.** The frozen dataset covers
   tool-output filtering and direct injection only. SecJev's defining task
   (action authorization) still needs the label-rule rebuild over recorded
   trajectories before any SecJev fine-tune can consume v2 data.
3. **CLEF encode cost and contract.** Measured on this Mac: load+graph
   compile 203.5 s, then **12.0 s median per row** for LLMail-length
   requests (8,000-char truncation, 11 rows after the first). Encoding all
   195k canonical rows single-stream projects to roughly 650 hours; even a
   15k-row subset projects to roughly 50 hours. A CLEF redo therefore needs
   an explicit predeclared subsample with shorter truncation, a parallel
   encode farm, or a CUDA torch-backend host — decided before scheduling.
   The reduced text-only contract also still needs to be written and
   versioned.
4. **Label quality gates.** Programmatic audit has run
   (`audit_redo_dataset.py`, report `dataset-audit.json`): zero same-text
   label conflicts in every frozen file; dominant-script distribution is
   overwhelmingly Latin (5,605 sampled rows vs 66 CJK, 5 Cyrillic, 2
   Arabic); sampled pattern rows include expected corpus emails/URLs, 12
   Luhn-valid card-like rows, and zero API-key-shaped hits. Human
   label-audit sampling with adjudication has **not** run and is not
   replaced by the programmatic audit. Direct-injection negatives are now
   present in dev/test in dataset v0.2 (`frozen-manifest-v2.json`,
   stratified assignment: dev 131, test 130 SaTML non-attacks; keyword
   direct-kind test AUC 0.569 in `baseline-keyword-report-v2.json`).
   Stratified mode splits direct rows at chat level, so SaTML participants
   can appear in multiple splits for that kind; participant-clustered
   bootstrap CIs are mandatory for direct-kind metrics. Human adjudication
   and benign expansion remain open.
5. **Live/adaptive evals.** `run_pentest_evals.sh` still contains the known
   stale-cache and zero-row reporting hazards in `score.py`/`analyze.py`/
   `report.py` (all three files are dirty from unrelated work and were not
   edited here). Harden those paths before any live arm runs.

## Next executable order

1. Human label adjudication and benign expansion (dataset v0.3); stratified
   v0.2 and programmatic audits are done.
2. GPU host provisioning or `ft_action.py` MPS/MLX port with equivalence
   tests (unblocks Phase 4).
3. CLEF reduced contract + predeclared encode subsample/truncation budget
   using the measured 12.0 s/row rate (unblocks Phase 5 scheduling).
5. Only then: candidate training with dev-only selection, frozen test,
   adaptive attacks, replication, corrected publication.
