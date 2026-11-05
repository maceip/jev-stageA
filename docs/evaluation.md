# Evaluation guide

jev-stageA evaluates JEV agent defenses on AgentDojo v1.2.2 in four stages: record agent episodes, score every decision point with every detector, compute defended outcomes by exact replay (stop mode) or live runs (redact mode), and attack the defended agent with a defense-aware adaptive attacker. Thresholds are tuned on the workspace+travel suites and reported on banking+slack. The frozen decision rules are in [analysis-plan.md](analysis-plan.md).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "git+https://github.com/ethz-spylab/agentdojo@089ed468cf3ed0322acc66b0211f26d9d90dbf60"

hf download MetaPublic/SecJev-4B --local-dir checkpoints/SecJev-4B
hf download MetaPublic/SecJev-4B-Action-Monitor-C1 --local-dir checkpoints/SecJev-4B-Action-Monitor-C1
export SECJEV_WORK="$PWD/secjev-4b-work"
python checkpoints/SecJev-4B/bootstrap.py --inference-only
export SECJEV_TF32=1 OPENAI_API_KEY=...
```

Hardware: one NVIDIA GPU with at least 40 GB runs one SecJev checkpoint plus the classifier baselines. The cascade holds two checkpoints; the shared GPU lock in `secjev_scorer.py` keeps the cascade within one 40 GB card for scoring. Fine-tuning (`ft_action.py`) used A100 80 GB cards.

Cost warning: recording and adaptive stages call the agent, attacker, and judge APIs once per episode or round. A full `run_pentest_evals.sh` run is thousands of API episodes. Start with `--limit` on `run_agent.py` and a small `--pairs` file for `adaptive.py`.

## Stage 1 — record

```bash
# Undefended baselines
python run_agent.py --attack benign --model gpt-5.4-mini --out runs/benign.jsonl
python run_agent.py --attack important_instructions --model gpt-5.4-mini --out runs/important_instructions.jsonl
python run_agent.py --attack ood:bland_note --model gpt-5.4-mini --out runs/ood_bland_note.jsonl

# Or record with a live defense (required for redact-mode outcomes)
python run_agent.py --attack ood:bland_note --suites banking,slack --model gpt-5.4-mini \
  --defense cascade:content:0.35:redact \
  --secjev checkpoints/SecJev-4B-Action-Monitor-C1 \
  --secjev-tool checkpoints/SecJev-4B --tool-threshold 0.99 \
  --out runs/bland_cascade_redact.jsonl
```

Attack names: AgentDojo built-ins (for example `important_instructions`), `benign`, and `ood:<name>` for the out-of-distribution templates in `attacks_ood.py` (`bland_note`, `metadata_field`, `quoted_reply`, `zero_width`, `llmail`). The OOD templates are attack text absent from SecJev-Corpus training data; every built-in AgentDojo attack appears in SecJev-Corpus training data, so built-in attack sets measure memorized-text detection and OOD sets measure generalization.

## Stage 2 — score

```bash
python score.py runs/ood_bland_note.jsonl --group secjev --secjev checkpoints/SecJev-4B --outdir scores
python score.py runs/ood_bland_note.jsonl --group classifiers --outdir scores
python score.py runs/ood_bland_note.jsonl --group judge --port https://api.openai.com/v1 --model gpt-5.4-mini --outdir scores
```

Detector groups: `secjev` (tool-output + both action views + cascade), `classifiers` (PIGuard, DeBERTa protectai v2), `judge` (LLM judge). Raw scores are stored per decision point; no threshold is applied at scoring time.

## Stage 3 — analyze

```bash
python analyze.py --runs runs --scores scores --out results.json
python report.py results.json
```

Stop-mode defended outcomes come from exact replay (`replay.py`): recorded tool calls are re-executed deterministically and episodes are cut at the first flag. Replay parity was validated on every recorded run used for reported numbers. `analyze.py` reports attack success and task utility with Wilson 95% intervals per detector and threshold, including the `secjev_cascade` combination, with thresholds calibrated on `--tune-suites` and optionally filtered to `--report-suites`.

## Stage 4 — adaptive red-team

```bash
python adaptive.py --pairs pairs_bs100.json --defense secjev_cascade \
  --secjev checkpoints/SecJev-4B-Action-Monitor-C1 --secjev-tool checkpoints/SecJev-4B \
  --threshold 0.35 --tool-threshold 0.99 --mode redact \
  --model gpt-5.4-mini --attacker-model gpt-5.5 --rounds 10 \
  --out adaptive/cascade_redact.jsonl
```

The attacker LLM receives the defense verdict and outcome of each live round and revises the payload, up to `--rounds` rounds per target. Every defense arm receives the same feedback rule. An attacker turn refused by the provider safety filter counts as a used round for every arm.

## All-in-one

```bash
bash run_pentest_evals.sh
```

`run_pentest_evals.sh` runs the component tests, then evaluates four arms live on banking+slack with a `gpt-5.4-mini` agent: Tool Output Filter alone (0.99), Action Authorization Guard alone (C1 at 0.35), cascade, and cascade + LLM judge — each on normal tasks, the 1-try bland-note attack (249 episodes), and the 10-round adaptive attacker (100 targets), then prints a summary table. Checkpoint paths come from `SECJEV_MODEL` and `SECJEV_ACTION_CKPT`; the default local layout is `checkpoints/SecJev-4B` and `checkpoints/SecJev-4B-Action-Monitor-C1`.

## Tests

```bash
SECJEV_WORK="$PWD/secjev-4b-work" \
SECJEV_MODEL="$PWD/checkpoints/SecJev-4B" \
SECJEV_ACTION_CKPT="$PWD/checkpoints/SecJev-4B-Action-Monitor-C1" \
OPENAI_API_KEY=... python -m pytest -q tests/
```

The test suite runs real AgentDojo, real SecJev checkpoints, and real API calls — no mocks and no skips for missing requirements; a missing requirement fails the run with a named environment variable. `tests/test_e2e_components.py` covers replay parity, scoring, and defense construction; `tests/test_e2e_live.py` runs live defended episodes and one adaptive target.

## Reported runs and data layout

Three run generations produced the reported numbers:

| Generation | Agents | Contents |
|---|---|---|
| Run 1 (`data/`) | gpt-oss-20b, Qwen3-30B-A3B (local vLLM) | 97 normal tasks + 6 attack sets per agent; tables in `data/results/` |
| Run 3 (`run3/`) | `gpt-6-astra`, `gpt-5.4-mini` (OpenAI API) | Standard + bland-note + LLMail sets, live stop/redact arms, adaptive arms, score archives (`scores_*.tgz`) |
| Fine-tune (`ft/`) | Recorded runs from runs 1 and 3 | Action-authorization train/dev/test JSONL (`ft_*.jsonl`), per-variant replay scorings (`ft/evals/`), C1 checkpoint metadata (`ft/ckpt/CHECKPOINT.md`) |

Key Run 3 findings (full tables in the repository README research summary and `data/results/`):

- Static attacks never hijacked `gpt-6-astra` (0/~2,800 across standard, bland-note, and LLMail sets). `gpt-5.4-mini` × bland-note (9.3% hijacked overall; 34.1% on banking+slack) is the configuration that separates defenses.
- Stop mode, banking+slack, thresholds tuned on workspace+travel: SecJev tool at 0.99 → 2.8% hijacked, 94.6% normal; LLM judge → 0.0% hijacked, 83.8% normal; zero-shot action monitor → 6.4% hijacked, 73.0% normal, 55.4% completion under attack.
- Fine-tuned C1 content monitor at 0.35 (live): stop 1.6% hijacked / 89.9% normal; redact 0.8% hijacked / 90.5% normal / 75.5% completion under attack.
- Adaptive (10 rounds, 100 banking+slack targets): undefended 64%, LLM judge 25%, SecJev tool at 0.5 31%, zero-shot action monitor 34%, C1 monitor 53–58%. The adaptive result is the main open weakness and the primary target for RepE/steering follow-up work.

## Reproducing SecJev-4B published accuracy

```bash
python verify_corpus.py
```

Expected: 419/426, 424/426, 124/124 on the SecJev-Corpus agent-security test tasks, identical to the upstream release.
