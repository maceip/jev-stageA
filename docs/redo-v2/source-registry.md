# Phase 1 — source registry and acquisition record

Status: acquisition audit in progress. A source listed here is not approved for
training until the status says the pinned files, license, schema, and label
mapping have all been checked. Kaggle and other aggregators are discovery
indexes only; no aggregator copy is a provenance substitute.

Downloaded primary data lives under `data/external/`, which is gitignored.
The machine-readable registry used by dataset code is `SOURCE_REGISTRY` in
`redo_v2.py`.

## Downloaded at pinned revisions

### LLMail-Inject Challenge

- Primary source: `microsoft/llmail-inject-challenge`
- Revision: `1063bdf01ec8762b812d5e06ee768a06faa5a6f7`
- License from dataset card metadata: MIT
- Local path: `data/external/llmail-inject-challenge`
- Downloaded bytes: 2,446,829,207 across the snapshot directory
- Evidence tier: A — human/adaptive
- Raw files:
  - `data/raw_submissions_phase1.jsonl` (1,666,413,489 bytes)
  - `data/raw_submissions_phase2.jsonl` (263,506,099 bytes)
  - `data/labelled_unique_submissions_phase1.json` (448,194,111 bytes)
  - `data/labelled_unique_submissions_phase2.json` (68,586,935 bytes)
  - `data/emails_for_fp_tests.json`
- Raw JSONL fields observed: `RowKey`, `body`, `subject`, `objectives`,
  `output`, `scenario`, `team_id`, and timestamps.
- Labelled-unique files are objects keyed by full email text; first inspected
  value contains `attack_attempt` and `reason`.
- Count discipline: the paper reports 208,095 distinct human-written attack
  prompts. The HF viewer exposes 462k raw rows (371k Phase 1 and 90.9k
  Phase 2). Distinct prompts and raw submission rows must never be conflated.
- Supported conversion today: Spot-1/tool-output filtering only. A raw row
  does not by itself identify a concrete proposed tool call, so no
  action-authorization label is fabricated from the raw row alone.
- Required split groups: team, scenario/level, near-duplicate/template
  cluster, and phase/time where available.

### SaTML 2024 LLM Capture-the-Flag

- Primary source: `ethz-spylab/ctf-satml24`
- Revision: `11426babe61ee16b0efe7b55e0430359e0fc4235`
- License from dataset card metadata: MIT
- Local path: `data/external/ctf-satml24`
- Downloaded bytes: 682,067,530 across the snapshot directory
- Evidence tier: A — human/adaptive
- Main chat fields observed: `_id`, `history`, `is_attack`,
  `is_evaluation`, `model`, `defense`, `user`, `secret`, and
  `was_successful_secret_extraction`.
- Supported conversion today: direct prompt injection only. SaTML is a
  secret-extraction chat setting, not evidence for tool-call authorization.
- Required split groups: user/team where recoverable, defense, model, and
  template cluster.

### Lakera Gandalf ignore-instructions

- Primary source: `Lakera/gandalf_ignore_instructions`
- Revision: `04737b65e90a6794ec227012e4a255a7def6344b`
- License from dataset card metadata: MIT
- Local path: `data/external/gandalf_ignore_instructions`
- Rows observed locally: 777 train, 111 validation, 112 test; fields are
  `text` and `similarity`.
- Evidence tier: A — small human seed/eval set
- Supported conversion today: direct prompt injection only.

## Not yet acquired or not approved for training

### Tensor Trust

- Tier A candidate: large human attack/defense game corpus.
- Status: not acquired. The abstract-level source inspected during planning
  does not settle file-level license terms. Acquisition remains quarantined
  until primary files and terms are verified.

### BIPIA

- Tier B benchmark/context candidate.
- Shallow clone acquired at `a004b69ec0dd446e0afd461d98cb5e96e120a5d0` under
  `data/external/repos/BIPIA`.
- Repository code license file is MIT. Component data remains separate:
  local context files currently include EmailQA (50 test / 50 train), CodeQA
  (50 / 50), and TableQA (100 / 900). WebQA and Summarization contexts are not
  included and must be generated from their source datasets under source
  terms. No bulk training use is approved until component rights are recorded
  per task.

### InjecAgent

- Tier B synthetic tool-integrated benchmark.
- Shallow clone acquired at `f19c9f2c79a41046eb13c03c51a24c567a8ffa07` under
  `data/external/repos/InjecAgent`; `LICENCE` is MIT.
- Local base test files contain 510 direct-harm and 544 data-stealing cases
  (1,054 total).
- Status: external test only. Training on InjecAgent would invalidate any
  reported InjecAgent benchmark result.

### PINT

- Tier B mixed public/proprietary detection benchmark.
- Status: external test only if access terms allow. The MIT license observed
  for benchmark code does not automatically license every proprietary dataset
  input for training.

### Local directories

- `data/`, `run3/`, and `ft/` remain historical local evidence until each file
  is tied to a generation command, model, date, and label rule.
- `/Users/mac/jevalin-collect` and `/Users/mac/siftrank/benchmarks` are
  quarantined for this redo until source-by-source provenance, consent/PII,
  and synthetic-vs-captured status are audited. Generated enterprise traffic
  must remain labelled synthetic.

## Initial conversion audit

A streaming sample audit converted the first 2,000 records from each large
downloaded source through `redo_v2.py`:

- LLMail Phase 1: 370,724 raw lines; 2,000 converted, 0 errors, 1,246
  distinct normalized texts in the ordered sample.
- LLMail Phase 2: 90,916 raw lines; 2,000 converted, 0 errors, 352 distinct
  normalized texts in the ordered sample.
- SaTML: 137,063 raw lines; 2,000 converted, 0 errors, 1,933 distinct
  normalized texts in the ordered sample; sampled labels were 1,998 attack
  and 2 non-attack.

These are ordered-prefix samples, not corpus-wide duplicate rates. A separate
full raw-file audit counted 370,724 Phase 1 rows (138,352 distinct normalized
bodies; 224 teams) and 90,916 Phase 2 rows (34,417 distinct normalized
bodies; 99 teams), with zero JSON parse errors. Cross-phase distinct counts
were not summed because cross-phase overlap was not computed in that audit.
Observed successful objective-chain counts were `exfil.sent` 41,173 in
Phase 1 and 5,733 in Phase 2; those objective counts describe challenge
outcomes, not action-authorization labels.

## Phase 1 exit status

Closed by decision (see `local-provenance.md`): Tensor Trust excluded for v2
training (no license in the data repo at `747a75e`); BIPIA and InjecAgent are
external-eval only; PINT is unavailable without access. Local-directory
provenance classification is recorded in `local-provenance.md`.

Canonical conversion has run for the approved MIT sources. The frozen
artifact is `redo-v2-canonical-spot1-direct-v0.1` (195,313 rows: 143,243
train / 26,637 dev / 25,433 test), described by `frozen-manifest.json` in
this directory and independently re-verified (sha256, counts, zero
participant/template/text overlap across splits). The frozen dataset covers
tool-output filtering and direct prompt injection only.

Still open before any training: human label-audit sampling with adjudication,
language and PII/secret scans over canonical rows, benign/hard-negative
expansion beyond the 203 LLMail FP emails and ~11.5k labelled negatives, and
the entire Spot-2 action-authorization dataset rebuild.
