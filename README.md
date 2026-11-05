# Weaponizing JEV for Defenders

**jev-stageA** tests whether a small JEV-style typed decision model — [SecJev-4B](https://github.com/UESTC1010/SecJev/releases/tag/v1.2.0) (arXiv 2610.03073), a 4B security-trained model in the Kev/JEV family — can defend a tool-using LLM agent against indirect prompt injection, and provides the defense code, evaluation harness, and fine-tuned checkpoint needed to rerun the evaluation and keep pushing the approach forward.

> **Status: research preview, not a production security product.** The pre-registered decision gate for a full build was **not** passed (see [Results](#results)). SecJev-4B works as an agent-independent injection detector and the action-monitor placement preserves work under attack better than any tool-output detector tested, but SecJev-4B does not beat a strong LLM judge on attack success, and the fine-tuned action monitor remains weak against a 10-round adaptive attacker. Run the defense in a sandbox and treat every number in the repository as reproducible research evidence, not as a deployment guarantee.

## The defense

The same SecJev-4B family checkpoint is placed at up to two points in the agent loop:

| Placement | Code | Checkpoint | Question SecJev-4B answers |
|---|---|---|---|
| Tool Output Filter (`secjev_tool`) | `ToolOutputGate` in [defenses.py](defenses.py) | SecJev-4B v1.2.0 (zero-shot) | Does the incoming tool result contain an injected instruction? |
| Action Authorization Guard (`secjev_action_content`) | `ActionGuardedExecutor` in [defenses.py](defenses.py) | SecJev-4B-Action-Monitor-C1 (fine-tuned) | Is the proposed outgoing tool call authorized by the user's request, given the tool outputs the agent has seen? |
| Both gates (`secjev_cascade`) | `build_secjev_cascade` in [defenses.py](defenses.py) | SecJev-4B at Tool Output Filter + SecJev-4B-Action-Monitor-C1 at Action Authorization Guard | Both questions; an optional LLM judge can give a second opinion on flagged calls (`secjev_cascade_judge`) |

Each placement runs in two modes. **Stop mode** aborts the episode on the first flag. **Redact mode** replaces a flagged tool output with a fixed removal notice, or refuses only the flagged tool call, and the agent continues. A detector that cannot produce a verdict raises an error; no detector failure is silently passed through.

The programmatic interface is documented in [Defense API](docs/defense-api.md). The forward research design — layering a JEV decision service with multi-CaMeL capability tracking, CounterSteer steering, a RepE probe circuit breaker (Sentinel), and an effect-authorization gate (EffectMatch) — is documented in [Weaponizing JEV for Defenders](docs/weaponizing-jev-for-defenders.md). Only the SecJev layers are implemented in jev-stageA; the remaining layers are specified designs with defined integration points in [defenses.py](defenses.py), not shipped code.

## Model weights

| Checkpoint | Hugging Face | Role |
|---|---|---|
| SecJev-4B v1.2.0 (unmodified upstream mirror) | [MetaPublic/SecJev-4B](https://huggingface.co/MetaPublic/SecJev-4B) | Tool Output Filter |
| SecJev-4B-Action-Monitor-C1 (fine-tuned in jev-stageA) | [MetaPublic/SecJev-4B-Action-Monitor-C1](https://huggingface.co/MetaPublic/SecJev-4B-Action-Monitor-C1) | Action Authorization Guard |

Each Hugging Face repository is a complete Kev/SecJev checkpoint directory: LoRA adapter, pointer decision head, tokenizer, calibration, inference code (`infer.py`), and a pinned bootstrap (`bootstrap.py`). The bootstrap downloads the two remaining upstream components from the upstream distributors: the Qwen3.5-4B-Base backbone (pinned revision `1001bb4d`, Apache-2.0, from the Qwen Hugging Face repository) and the Kev source code (pinned commit `90990a5f`, Apache-2.0, from GitHub). The Qwen3.5-4B-Base backbone weights are not re-hosted under MetaPublic because Qwen distributes the pinned backbone directly.

Weights are licensed under the SecJev Use and Distribution Agreement v1.0 plus applicable upstream terms, not under the Apache-2.0 license that covers the jev-stageA code. See [License](#license).

## Quickstart: run the defense

Requirements: Python 3.12+, one NVIDIA GPU with at least 40 GB (Apple Silicon MPS/MLX also works for scoring; every reported number used CUDA), and an OpenAI API key for the agent model. Local OpenAI-compatible endpoints (for example vLLM) work through `--base-url`.

```bash
git clone https://github.com/maceip/jev-stagea && cd jev-stagea
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "git+https://github.com/ethz-spylab/agentdojo@089ed468cf3ed0322acc66b0211f26d9d90dbf60"

# Download both checkpoints and prepare the pinned backbone + Kev source
hf download MetaPublic/SecJev-4B --local-dir checkpoints/SecJev-4B
hf download MetaPublic/SecJev-4B-Action-Monitor-C1 --local-dir checkpoints/SecJev-4B-Action-Monitor-C1
export SECJEV_WORK="$PWD/secjev-4b-work"
python checkpoints/SecJev-4B/bootstrap.py --inference-only

export OPENAI_API_KEY=...   # never commit an API key; see .env.example
export SECJEV_TF32=1        # validated fast mode; omit for reference numerics

# Run AgentDojo banking+slack under the bland-note attack, defended at both gates, redact mode
python run_agent.py --attack ood:bland_note --suites banking,slack --model gpt-5.4-mini \
  --defense cascade:content:0.35:redact \
  --secjev checkpoints/SecJev-4B-Action-Monitor-C1 \
  --secjev-tool checkpoints/SecJev-4B --tool-threshold 0.99 \
  --out runs/bland_cascade.jsonl
```

Sanity-check the checkpoint stack without an agent run:

```bash
python checkpoints/SecJev-4B/infer.py --model ./checkpoints/SecJev-4B \
  --base "$SECJEV_WORK/base" --request checkpoints/SecJev-4B/examples/certificate.json --device cuda
python verify_corpus.py   # reproduces SecJev-4B published test accuracy
```

## Rerun the evaluations

[Evaluation guide](docs/evaluation.md) documents the full record → score → replay → analyze → adaptive pipeline, the frozen decision rules in [docs/analysis-plan.md](docs/analysis-plan.md), and the data layout. The batteries-included entry point is:

```bash
bash run_pentest_evals.sh   # 4 JEV arms x {normal tasks, 1-try attack, 10-round adaptive}, banking+slack
```

`run_pentest_evals.sh` reads checkpoint locations from `SECJEV_MODEL` and `SECJEV_ACTION_CKPT`, runs the component test suite first, and writes JSONL results under `run3/clean_live/` and `run3/clean_adaptive/`. The evaluation calls paid agent/attacker/judge APIs; read the cost notes in [docs/evaluation.md](docs/evaluation.md) before launching a full run.

Recorded trajectories and score archives from the reported runs are kept in `data/` and `run3/` so the analysis stage can be rerun without recording new agent episodes.

## Results

Headline findings, with full tables in `data/results/` and the run-by-run account in [docs/evaluation.md](docs/evaluation.md):

- **SecJev-4B reproduces the SecJev paper numbers** (98.36% / 99.53% / 100% on the SecJev-4B test split; `verify_corpus.py`).
- **Tool Output Filter works and is agent-independent.** Against previously unseen attack text, attack success fell from 6–26% undefended to 0–0.7% with SecJev-4B on both open-weight agents tested, at the smallest normal-task cost of any classifier tested (PIGuard and DeBERTa could not reach 95% normal-task retention at any threshold).
- **Action Authorization Guard preserves work under attack.** In live redact-mode runs on held-out banking+slack suites (`gpt-5.4-mini` agent, bland-note attacks), the fine-tuned C1 monitor kept 75.5% of tasks completing under attack at 0.8% hijacked, versus 28.5% completing for the LLM judge at 0.0% hijacked. Undefended: 78.3% completing, 34.1% hijacked.
- **The pre-registered gate failed.** A strong LLM judge matches or beats SecJev-4B on attack success, and against a 10-round adaptive attacker the C1 monitor alone allows 53–58% hijacks (64% undefended, 25% LLM judge). Per the frozen rules in [docs/analysis-plan.md](docs/analysis-plan.md), the full layered build does not start on zero-shot SecJev-4B evidence.
- **Current frontier agents resist static AgentDojo injection.** `gpt-6-astra` was hijacked 0 times in ~2,800 statically attacked runs, so adaptive attacks and false-alarm cost are the measurable outcomes for strong agents.

## Research directions

jev-stageA is an open research platform, not a finished product. The highest-value next steps, specified in [Weaponizing JEV for Defenders](docs/weaponizing-jev-for-defenders.md):

1. **RepE Sentinel probe.** A representation-engineering probe (linear probe + cosine danger direction at the tool-input position) read during the same forward pass as generation, blocking before a tool call is emitted. The probe plugs into the scorer interfaces in [defenses.py](defenses.py) documented in [Defense API](docs/defense-api.md); no probe ships yet.
2. **CounterSteer.** Prefill-only, norm-preserving activation steering on tool-result tokens, fitted per agent model with causal and capability gates. Design specified; not implemented.
3. **Multi-view JEV head calibration and margin training.** Fine-tuning the JEV decision head (`ft_action.py`, `ft_clef.py`) jointly on Tool Output Filter and Action Authorization Guard (full-context) examples from `workspace + travel` reduces false alarms on indirect-reference tasks (such as paying a bill from a file) without brittle regex or string-matching heuristics.
4. **Stage B on multi-CaMeL.** Test the learned JEV monitor against the semantic attacks a capability-tracking interpreter lets through, separating the value of the typed JEV interface from check placement.

Contributions of new attack sets, RepE probe implementations, and additional agent-model evaluations are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository layout

| File | Purpose |
|---|---|
| [defenses.py](defenses.py) | Live defenses: `ToolOutputGate`, `ActionGuardedExecutor`, `build_secjev_cascade` (stop / redact) |
| [secjev_scorer.py](secjev_scorer.py) | Loads a SecJev checkpoint once; scores Kev System One requests |
| [run_agent.py](run_agent.py) | Records AgentDojo episodes (undefended or with a live defense) to JSONL |
| [score.py](score.py) | Scores every decision point of recorded runs with every detector |
| [replay.py](replay.py) | Exact counterfactual stop-mode outcomes by deterministic tool replay |
| [analyze.py](analyze.py), [report.py](report.py) | Attack success / utility with Wilson 95% intervals; Markdown tables |
| [adaptive.py](adaptive.py) | Defense-aware adaptive attacker (up to 10 rounds per target) |
| [attacks_ood.py](attacks_ood.py) | Attack templates absent from SecJev-Corpus training data + LLMail-Inject payloads |
| [build_action_data.py](build_action_data.py), [ft_action.py](ft_action.py) | Action-authorization training data; SecJev-4B action-monitor fine-tuning |
| [run_pentest_evals.sh](run_pentest_evals.sh) | Full 4-arm live evaluation suite |
| [publish_weights.py](publish_weights.py), [hf/](hf) | Assemble and upload both MetaPublic Hugging Face checkpoint repositories; model cards |
| [tests/](tests) | Real end-to-end tests (real AgentDojo, real SecJev, real APIs; no mocks) |
| [docs/](docs) | Research design, defense API, evaluation guide, frozen analysis plan |
| `data/`, `run3/`, `ft/` | Recorded runs, score archives, fine-tune data and per-variant evaluations |

## Responsible use

jev-stageA contains prompt-injection payloads and an adaptive attacker for controlled benchmark environments (AgentDojo). Use the attack code only against systems you own or are authorized to test. Report vulnerabilities in the defense code through [SECURITY.md](SECURITY.md).

## License

- **jev-stageA code and documentation:** Apache-2.0 ([LICENSE](LICENSE)).
- **Model weights** ([MetaPublic/SecJev-4B](https://huggingface.co/MetaPublic/SecJev-4B), [MetaPublic/SecJev-4B-Action-Monitor-C1](https://huggingface.co/MetaPublic/SecJev-4B-Action-Monitor-C1)): SecJev Use and Distribution Agreement v1.0 plus upstream terms. The SecJev agreement includes source-specific conditions (for example, ToN-IoT academic-use terms in SecJev-Corpus) that the Apache-2.0 code license does not remove.
- **Upstream components:** Kev (Apache-2.0), Qwen3.5-4B-Base (Apache-2.0), AgentDojo (MIT; `agent_llm.py` is vendored verbatim from AgentDojo at the pinned commit with attribution in the file header).

## Acknowledgements and citation

jev-stageA builds on SecJev (arXiv 2610.03073; UESTC1010/SecJev), Kev/JEV (jaredpalmer/kev), AgentDojo (ETH Zurich SPY Lab), and Qwen3.5 (Alibaba). Cite the SecJev paper and the upstream projects for the model and benchmark; cite jev-stageA for the defense placements, the C1 action-monitor fine-tune, and the evaluation results.
