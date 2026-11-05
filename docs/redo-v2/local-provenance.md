# Phase 1 — local provenance classification

Classification date: 2026-11-03 UTC. "Catalogued" does not mean admitted for
training. Each local source keeps the status below until a row-level lineage
check promotes the source.

| Local source | Classification | Redo-v2 disposition |
|---|---|---|
| `data/` and `run3/` recorded AgentDojo runs | Tier C — local recorded benchmark trajectories with in-repo generation code (`run_agent.py`, `build_action_data.py`) | Admissible as historical lineage after per-file command/model/date reconstruction. Not real-world traffic. |
| `ft/ft_*.jsonl`, `ft/ft_c_*.jsonl` | Tier C — derived action-authorization rows from recorded AgentDojo runs | Quarantined as Spot-2 training input until the label rule is rebuilt and counts reconcile with provenance. |
| `ft/clef_ft_run/`, `ft/fastjev_run*/` | Training outputs, not datasets | Evidence only. Never training inputs for v2. |
| `/Users/mac/jevalin-collect` generated enterprise/behavioral data | Synthetic by construction (generator entry points in the tree; catalog states inclusion is not training admission) | Excluded from v2 training. May inform benign-traffic shape only if a specific generated set is audited and labelled synthetic. |
| `/Users/mac/jevalin-collect/captures` | Unknown consent/PII status | Quarantined. No training use without capture-level consent and PII review. |
| `/Users/mac/siftrank/benchmarks` | Ranking-adapter evaluation artifacts (small synthetic experiments per its RESULTS.md) | Excluded from prompt-injection training. Not an injection corpus. |
| Tensor Trust clone (`data/external/repos/tensor-trust-data` @ `747a75e`) | Tier A candidate, terms unresolved: no LICENSE file found in the data repository | Quarantined. No training or redistribution until data terms are verified with the authors or a license is located. |
| PINT | Tier B mixed public/proprietary benchmark; full dataset not public | Not acquired. External-test only if access terms are granted; excluded from v2 training regardless. |

## Rights-check closure decisions

- Tensor Trust: **closed as excluded for v2 training** on current evidence.
  Reopening requires a located license or written terms from the authors.
- BIPIA: **closed as external-eval only for v2.** Repository code is MIT;
  Email/Code/Table contexts are present locally, while WebQA and
  Summarization contexts must be generated from source datasets under their
  own terms. No BIPIA row enters v2 training.
- PINT: **closed as unavailable.** Benchmark code is MIT, but the full
  4,314-input dataset blends proprietary data and is not public. No score or
  training claim will reference PINT until access exists.
- InjecAgent: **closed as external test only** (MIT `LICENCE`, clone
  `f19c9f2`). Training on InjecAgent would invalidate reported InjecAgent
  results.
