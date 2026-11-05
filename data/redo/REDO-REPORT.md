# CLEF / SecJev redo — verified report (2026-11-03)

Scope: redo of the CLEF (ft_clef.py) and SecJev (ft_action.py) fine-tunes on real-world
prompt-injection data, with held-out evaluation and an adaptive-attack gate. Every
number below traces to a file in this repo or a recorded command output from the run.

## What was wrong before

- Withdrawn CLEF run's training split (`ft/ft_c_train.jsonl`) contained **49 unauthorized
  (label 2) examples out of 2,626**. Best saved head: epoch 3, combined dev AUC 0.6981.
- The published-direction numbers rested on that corpus. This redo replaces the data
  foundation first, modeling second.

## Corpus (data/redo/)

Sources, licenses verified by API and LICENSE-file read (`raw/license-ledger.json`):

| Source | License | Role in v1 |
|---|---|---|
| microsoft/llmail-inject-challenge (HF) | MIT | Real challenge emails; capped 6,000 attack=True + 203 benign FP-test emails |
| Lakera/gandalf_ignore_instructions (HF) | MIT | Real direct-override submissions (777 train + other splits) |
| uiuc-kang-lab/InjecAgent (GitHub) | MIT | Native agent/tool injection cases |
| ethz-spylab/ctf-satml24 (HF) | MIT | Downloaded + ledgered; NOT yet adapted (v2) |
| microsoft/BIPIA (GitHub) | MIT (LICENSE file; GitHub API says NOASSERTION) | Downloaded + ledgered; NOT yet adapted (v2) |

Construction (`build_corpus.py`): attack/benign texts are **verbatim** from sources;
the surrounding Jev state scaffold (trusted request, proposed tool call) is constructed
and recorded as `scaffold: constructed` in `_meta`. Each text yields a paired
unauthorized (attacker call, label 2) and authorized (benign call, label 0) row.
Full canon: 13,222 rows / 6,611 unique texts. v1 training subset (`canon/v1/`):
stratified by source, texts truncated to 1,500 chars — 900/160/160 texts
(train/dev/test). Splits by SHA1(normalized text); a pair never crosses splits.

## Results (held-out, scored once per checkpoint/test pair via eval_cls.py)

Authorization AUC. Redo test = `data/redo/canon/v1/redo_test.jsonl` (n=320, balanced).
Original = seeded 300-row balanced sample of `ft/ft_c_test.jsonl` (of 2,134).

| Checkpoint | Redo test AUC | Original test AUC | FPR@0.5 (redo / orig) |
|---|---|---|---|
| CLEF redo `checkpoints/redo-clef-1` (best epoch 2, dev comb 0.9914) | **1.000** | **0.779** | 0.063 / 0.367 |
| Old best `checkpoints/fastjev-clef-ft6` (epoch 3) | 0.965 | 0.665 | 0.444 / 0.387 |
| SecJev redo `ft/redo_secjev1/epoch2` | **1.000** (FPR 0.0) | 0.690 | 0.000 / 0.473 |

- Redo beats old on both distributions (+0.114 AUC on original, +0.035 on redo test,
  with far lower FPR on redo test: 0.063 vs 0.444 at threshold 0.5).
- Absolute original-distribution performance remains weak for every checkpoint
  (AUC 0.67–0.78, FPR@0.5 0.37–0.47). No number here supports a deployment claim on
  AgentDojo-style traffic yet.
- Trainer-internal test (build_datasets banking/slack, n=220): AUC 1.0 both heads —
  recorded for provenance only; the samples every 3rd row and is not the held-out verdict.

Dev trajectories: CLEF auth AUC 0.983 (init) → 0.999 (epoch 2). SecJev dev AUC 0.91
(init) → 1.0 (epoch 1).

## Adaptive gate (data/redo/adaptive_gate.py)

40 held-out unauthorized examples, budget 6 text mutants each (case, spacing, benign
sandwich padding, system-note wrap, synonym swap, quote wrap), threshold 0.5:
**static catch 40/40, adaptive bypass 0/40.** Caveat: these are shallow transforms, not
an optimizing attacker in the full sense of arXiv:2510.09023 (Nasr & Carlini et al.,
"The Attacker Moves Second"); treat as a floor test, not a robustness proof.

## Live TypeSafe arm (~/jev-ultrafast)

Unblocked and run 4 times total across two sessions (0/4 strict pass). The agent
(jev-1.14.0) consistently opens the correct Lisbon property in 4 actions but never
clicks the search submit, so the destination filter never applies and the strict
verifier fails. Key wiring that was missing and is now in place: the working TypeSafe
key is `JEV_API_KEY` in `~/jevalin/.env` (verified live: 422-not-401 on auth probe,
then real decisions); text helper runs on OpenAI (`gpt-4o-mini`) after a code fix —
`model.py` sent an OpenRouter-style `reasoning` object that OpenAI rejects with 400
(regression test `test_openai_base_sends_no_reasoning_param`, suite 32 passed).
Browser runs use a dedicated headless Chrome via `BU_CDP_WS`; the user's Chrome is
untouched.

## Limitations (read before quoting any number)

1. **Paired-template triviality risk.** Attacker/benign proposed calls come from a
   small fixed template set per channel. A head can score by call identity alone.
   The 1.000 redo-test AUCs therefore overstate field performance; the original-sample
   rows (0.78/0.69) are the more honest generalization signal.
2. **LLMail dominance** in the full canon (~82% of rows pre-cap); v1 stratification
   mitigates but email remains the majority channel. BIPIA + SaTML24 adapters (v2)
   are the planned diversification.
3. **v1 caps**: 1,500-char truncation and 900-train-text cap were compute-driven
   (fresh-text backbone encoding ≈6 s/example). Full-canon training is a ~10 h encode.
4. **Pre-existing scorer bug**: every fine-tuned head in this repo (old and new) is
   saved F32 while the LiteRT scorer loader demands BF16, so `eval_cls.py` cannot
   load any fine-tuned checkpoint unmodified. Evaluations above used BF16-converted
   copies (`*-bf16` dirs); fix the saver dtype at the source before the next release.
5. **SecJev MPS port**: retrain ran a device-patched copy (`ft_action_redo.py`) with
   batch 8 / tok_budget 4096 / gradient checkpointing on MPS after CUDA-hardcoded
   defaults OOMed at 180 GB. Config differs from the original CUDA run; treat the
   comparison as indicative, not controlled.
6. Adaptive gate transforms are shallow (see above). Budget and optimizer strength
   must grow before any robustness claim.

## Artifacts

- Corpus: `data/redo/canon/` (+ `canon/v1/`), builder `data/redo/build_corpus.py`,
  downloads + ledger `data/redo/download_sources.py`, `data/redo/raw/license-ledger.json`
- CLEF: `ft_clef_redo.py`, run `ft/redo_run1/` (log, final_results, scores, evals,
  adaptive_gate.json), checkpoint `checkpoints/redo-clef-1` (+ `-bf16` scorer copy)
- SecJev: `ft_action_redo.py`, run `ft/redo_secjev1/` (log, epoch1/epoch2, evals)
- Live arm: `~/jev-ultrafast/.env` (created), traces under `~/jev-ultrafast/artifacts/`

Nothing pre-existing was modified except: `~/jev-ultrafast/jev_ultrafast/model.py`
(text-helper reasoning fix, with test) and additive new files listed above.
