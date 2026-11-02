# Go Plan: Layered IPI Defense Agent

Status: **GO**, decided by Ryan MacArthur on 2026-11-01. This file replaces the Stage A gate in section 0 of
`Layered IPI Defense Agent Project Plan.md` and turns sections 1-7 of that plan into the build. Everything here is
built as production code: real dependencies, no mocks, no stubs, no silent fallbacks, no skipped tests.

## 1. Decision record

What Stage A measured (RUNLOG.md, entries of 2026-11-01; live runs on gpt-5.4-mini, banking+slack held out):

| Defense | Normal tasks | Hijacked (static) | Work done under attack | Adaptive hijacked |
|---|---|---|---|---|
| Undefended | 97.3% | 34.1% | 78.3% | 64% |
| LLM judge, redact | 95.3% | 0.0% | 28.5% | 25% (stop) |
| SecJev tool check 0.99, redact | 98.6% | 0.0% | 27.3% | 53% (stop) |
| Fine-tuned SecJev action monitor (C1), redact | 90.5% | 0.8% | 75.5% | 53% |

Against the frozen Stage A rules, SecJev alone did not pass: rule 2 (normal tasks at least 95% of undefended) and
rule 3 (adaptive attacks) failed. The go decision does not rest on SecJev alone. It rests on two findings:

1. **A fine-tuned SecJev model is the only defense measured that keeps the user's work going under attack.**
   In redact mode it keeps 96% of undefended work, where the judge and the tool check keep about 35%.
2. **The attacks that beat every model-based defense are data substitution.** For example, a forged bill carrying
   the attacker's IBAN, planted where the user asked the agent to pay the bill. Paying it is consistent with the user's
   request, so no model that reads the conversation can stop it. Only provenance and policy can: the multi-CaMeL
   interpreter and EffectMatch layers of the plan, which hold out-of-band facts such as approved payees and signed
   approvals.

So the layered design is the response to the evidence: deterministic layers own authority, and model layers
(SecJev, CounterSteer, Sentinel) reduce how often those layers have to say no.

Stage A's rules carry forward as **regression gates** (section 6). The composed system must beat every single layer
on them.

## 2. Engineering standards (apply to every phase)

- **No mocks, stubs or fakes, anywhere.** CI lint fails the build on `unittest.mock`, `pytest-mock`, `responses`,
  `vcr`, `monkeypatch` of library internals, `MagicMock`, modules named `fake*`/`stub*`/`mock*`, `except: pass`,
  bare `except`, and any flag that disables a defense without failing startup.
- **Fail closed.** A defense that cannot produce a verdict raises, and the call is refused. Startup refuses when any
  model artifact, steering vector, probe, SecJev head or signing key is missing or fails its hash check.
- **Coverage.**
  - Deterministic packages (interpreter, policies, EffectMatch, adapters, protocol, harness logic) need 100% line and
    branch coverage, measured by `coverage.py` in tier-1 CI. A drop fails the PR.
  - Model-backed code (serving hooks, SecJev service, CounterSteer, Sentinel) is covered by tier-2 tests on real GPUs,
    with the covered-lines report checked in per run.
  - Every public function has at least one test that exercises it against its real dependency.
- **Reproducibility.**
  - Model weights, datasets and steering artifacts are pinned by SHA-256 in `deploy/manifests/`.
  - Pre-registrations (splits, thresholds, hypotheses) are committed with hashes before any held-out data is touched.
  - Run logs are append-only; corrections are new dated entries, never edits.
- **Statistics.**
  - Episode is the unit; Wilson 95% intervals throughout.
  - Paired comparisons on the same episodes use McNemar exact tests plus Newcombe paired-difference intervals.
  - Thresholds are tuned only on tuning splits.
- **Secrets.** API keys live in a secrets manager on the run hosts, never in files, commits, logs or chat. Keys pasted
  during Stage A are revoked and replaced before Phase 0 ends.
- **Agents under test.**
  - gpt-5.4-mini is the measurement agent: it is attackable, so defenses can be told apart.
  - gpt-6-astra is the robustness agent: it resisted all static attacks.
  - gpt-5.5 is the attacker.
  - Model names are validated against the provider's model list at startup.

## 3. Starting point (what exists and is kept)

- `defenses.py`: tool-output gate and action monitor, stop and redact modes.
- `run_agent.py`, `adaptive.py`, `replay.py`, `analyze.py`: AgentDojo v1.2.2 harness. Replay matches 346 of 346
  recorded outcomes.
- `ft_action.py`, `build_action_data.py`: action-monitor training data and trainer. Checkpoint C1 epoch 1 is stored in
  `ft/ckpt/` (adapter and head) and on the run hosts.
- `tests/`: 19 real end-to-end tests that pass on an H100 80 GB and an A100 40 GB (`test_results/`).

## 4. Milestones

Each milestone ends with its exit criteria met on the real stack, a tagged commit, and test results committed to
`test_results/`.

### M0. Foundations (week 1)

- Restructure into the packages of plan section 3 (`camel/`, `effectmatch/`, `secjev/`, `steer/`, `sentinel/`,
  `serving/`, `harness/`, `deploy/`, `tests/`). The existing harness moves to `harness/agentdojo/` unchanged in
  behavior; replay parity is the move's test.
- CI with three tiers:
  - T1: CPU, Postgres 17 and MySQL 8 containers, every PR.
  - T2: 2 GPUs, nightly.
  - T3: 8 GPUs plus API budget, weekly and at release.
  - Lint bans and coverage gates as in section 2.
- Dependency audit (plan Phase 0): every dataset, weight and benchmark is obtainable and licensed, or replaced by a
  recorded decision.
- Secrets manager on the run hosts; keys rotated.

**Exit:** CI green on all three tiers; the existing 19 e2e tests run in T2; lint bans enforced; dependency audit
recorded.

### M1. Payee and recipient provenance policy (weeks 1-3): closes the measured gap first

- Policy engine: every effectful call (send_money, update_scheduled_transaction, send_email, send_direct_message,
  share_file, invite_user_to_slack, post_webpage, update_password, and the rest of the effect list, derived from tool
  schemas rather than name prefixes) is checked against provenance of its arguments.
  - A recipient, IBAN, URL or address must come from the user request, an approved-entity registry, or a value the user
    approved.
  - A value that came from untrusted tool output is a proposal the user confirms. It is never executed directly.
- Approved-entity registry, scoped per user, with signed changes.
- Confirmation channel: refused calls return a typed confirmation request to the user, never a silent retry.
- Composition: policy, then fine-tuned SecJev content monitor (redact), then SecJev tool check (0.99, redact). Each
  layer can be switched off for ablation, and startup fails if a configured layer cannot load.

**Exit:** on the Stage A adaptive protocol (100 banking+slack targets, gpt-5.5 attacker, 10 rounds, same feedback):

- adaptive hijack rate at most 5% and lower than the LLM judge's 25%, with non-overlapping intervals;
- normal tasks at least 95% of undefended, counting confirmations as a cost;
- work done under attack at least 70%.

### M2. EffectMatch (weeks 2-6, plan Phase 1)

- Effect manifests, signed licenses, occurrence ledger with atomic claim, authority binding, deployment frame.
- Adapters:
  - PROMOTABLE: PostgreSQL first, then MySQL and file.
  - MEDIATED: HTTP and GitHub.
  - OPAQUE: explicit opt-in only.
- Terminal journal, continuation gate, reconciliation worker.

**Exit:** plan Phase 1 exit criteria: 39 task-fault combinations × 3 repeats with 0 incorrect commits; 206 public tasks;
6 ablations each reproduce the failure the removed mechanism prevents; cross-process recovery under `kill -9`. Plan
e2e scenarios 1-3 pass.

### M3. Multi-CaMeL runtime (weeks 2-7, plan Phase 2)

- Capability-tracking interpreter with taint graph and policy hooks. The M1 policies move into the interpreter as
  capability rules.
- Agent-as-tool registry, `query_agent_j(message, variables)`, `get_variable`, instruction-channel integrity and
  cross-agent provenance.
- Individual-CaMeL mode kept for the composition-failure reproduction.
- MultiAgentDojo partitions; effectful calls routed through EffectMatch.

**Exit:** plan Phase 2 exit criteria: boundary laundering succeeds under individual CaMeL and fails under multi-CaMeL;
OMNI-Leak blocked; MultiAgentDojo ASR 0.0%; utility gap inside the 95% CI on AssetOpsBench. Plan e2e scenarios 4 and
6 pass.

### M4. SecJev decision service (weeks 3-8, plan Phase 5)

- Production service: batched single-pass scoring, typed responses, temperature calibration on a calibration split,
  latency SLO, decisions emitted as capability labels.
- Action-monitor training moves from `ft_action.py` into `secjev/`, with:
  - group-preserving splits;
  - leave-one-suite-out evaluation, since transfer to unseen tool sets was the measured weakness;
  - adversarial examples from the M1 adaptive runs;
  - replay of SecJev-Corpus to keep the tool-output skill.
- SecJev-Corpus builders for the 8 sources, so the released model can be retrained from source.

**Exit:**

- plan Phase 5 accuracy targets: task-macro 4B ≥ 94.5%, all 14 tasks above Kev init;
- action monitor on held-out suites: normal tasks ≥ 95% of undefended in redact mode, static hijack ≤ 1%;
- no regression on the SecJev release test split: at least 419/426, 424/426 and 124/124.

### M5. White-box serving and CounterSteer (weeks 3-10, plan Phase 3)

- vLLM patch with tool-span masks, forward hooks, and cache keys that include the steering-config hash.
- Five-step direction recipe with causal and capability gates; dose search; SHA-pinned artifacts per model.

**Exit:** plan Phase 3 exit per model; serving byte-identity on 1k prompts without tool spans; plan e2e scenario 5
(steering half).

### M6. Sentinel RepE circuit breaker (weeks 6-11, plan Phase 4)

- Activation capture with CounterSteer active; layer sweep; probe and cosine detectors; FPR-5% thresholds.

**Exit:** plan Phase 4 exit; plan e2e scenario 5 (Sentinel half).

### M7. Evaluation plane and composed ablation (weeks 4-14, plan Phase 6)

- Paper-1 matrix, adaptive suite (framing search, surrogate refit, exact-vector GCG, LLMail-Inject replay, AutoDojo),
  effects benchmarks.
- Stage A harness and gates as a standing regression suite.
- Ablation grid over every subset of {multi-CaMeL, policy, CounterSteer, Sentinel, SecJev, EffectMatch}.

**Exit:** paper tables reproduced within stated intervals; the composed system meets section 6 on every agent under
test.

### M8. Hardening and release (weeks 14-16, plan Phase 7)

- Load tests, SLOs, per-decision journal and traces, runbooks (INDETERMINATE resolution, direction refits on model
  upgrade, key rotation).
- Threat-model document; release checklist; all seven plan e2e scenarios green in T3.

## 5. Order of work and staffing of compute

- Start M0 and M1 now. M1 is first because it targets the failure measured in Stage A and reuses the existing harness.
- M2 and M3 run in parallel on CPU after M0. M4 starts with M1's adaptive data. M5 and M6 need the serving patch and
  8 GPUs; they begin once M0's dependency audit is complete.
- Compute:
  - T2 nightly: 2× H100.
  - Training and T3: 8× H100 or A100 80 GB, booked per run with a self-terminate watchdog.
  - Idle GPU nodes are shut down within the hour.

## 6. Release gates (the Stage A rules, applied to the composed system)

Measured on held-out suites, live runs, both agents under test. Each defense configuration is compared with the
undefended agent and with the LLM judge on the same episodes.

| Gate | Requirement |
|---|---|
| G1: security, static | Hijack rate on every static attack set ≤ 1%, with the 95% upper bound below the LLM judge's or a paired McNemar test showing no worse |
| G2: normal tasks | ≥ 95% of undefended completion, counting user confirmations as a cost |
| G3: adaptive | Adaptive hijack rate ≤ 5% and lower than the LLM judge's, with non-overlapping 95% intervals |
| G4: useful work under attack | ≥ 70% task completion under attack (Stage A: judge 28.5%, monitor 75.5%) |
| G5: latency | Per-decision overhead p95 ≤ 250 ms on one GPU for the model layers |
| G6: reliability | Agent and harness errors < 1%; zero incorrect commits in the EffectMatch fault matrix |

## 7. Open decisions

- **Confirmation UX for M1:** confirm every new payee, or only above an amount threshold.
- **Primary QLLM for the steered serving stack:** gpt-oss-20b (deepest CounterSteer evidence) or Qwen3-30B.
- **Non-CaMeL agent mode:** ships, or stays evaluation-only.
- **Model docs:** `ANALYSIS_PLAN.md` and `RUNBOOK.md` still name `gpt-6.1-astra`, which does not exist on the API key.
  They need dated corrections, not rewrites.
