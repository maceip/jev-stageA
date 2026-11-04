# jev-stagea

research code. jev-stagea tests small typed decision models as security checks inside a tool-using llm agent. the threat is indirect prompt injection: instructions hidden in tool outputs such as email, calendar entries, and web pages. the benchmark is agentdojo. jev-stagea is not a production security product.

## the two checks

- tool output check: after a tool returns text, the model answers one question: does the text contain an injected instruction. code: `ToolOutputGate` in `defenses.py`.
- tool call check: before a tool call runs, the model answers one question: did the user ask for the call, given the tool outputs the agent has seen. code: `ActionGuardedExecutor` in `defenses.py`.
- both checks together: `build_secjev_cascade` in `defenses.py`. an llm judge can review flagged calls.

each check has two modes. stop mode ends the run at the first flag. redact mode removes the flagged tool output, or blocks only the flagged call, and the run continues. a check that cannot produce a score raises an error. no check passes content through in silence.

two model families are supported: secjev-4b (kev format, loaded by `secjev_scorer.py`) and cloudflare clef-flash (litert and pytorch paths in `secjev_scorer.py`).

## results, short version

- the secjev-4b tool output check works. on attack text outside secjev training data, attack success fell from 6–26% with no defense to 0–0.7% with the check, on both open-weight agents tested.
- the fine-tuned c1 tool call check keeps more work done under attack: 75.5% of tasks completed at 0.8% hijacked in redact mode, against 28.5% completed at 0.0% hijacked for an llm judge. no defense: 78.3% completed, 34.1% hijacked. held-out suites: banking and slack.
- both checks together gained nothing in live runs: 2 of 249 hijacked, same as the tool call check alone, with task completion under attack down to 72 of 249 against 194 of 249.
- against an attacker that retries up to 10 rounds per target, the c1 tool call check is weak: 53–58% hijacked, against 64% with no defense and 25% for an llm judge.
- the pass rules fixed before the rerun were not met. full numbers and rules: `docs/evaluation.md` and `docs/analysis-plan.md`.
- redo and clef fine-tunes report dev auc up to 0.99 on small dev sets. no frozen test result exists for redo weights yet. read `docs/redo-v2/hf-uploads.md` before citing redo numbers.

## weights

public hugging face repos under the account `macmacmacmac`:

- `macmacmacmac/SecJev-4B-Spot2-Redo` — tool call check, redo fine-tune
- `macmacmacmac/SecJev-4B-Spot1-Redo` — tool output check, redo fine-tune
- `macmacmacmac/SecJev-4B-Joint-Redo` — both questions, one checkpoint
- `macmacmacmac/CLEF-Screen` — clef-flash fine-tuned head, runnable litert directory

the original secjev-4b release is at github `UESTC1010/SecJev` release v1.2.0. the c1 tool call checkpoint from the first evaluation is staged and hash-checked on the maintainer machine and is not on hugging face yet. `publish_weights.py` assembles and uploads the c1 pair. there is no metapublic hugging face organization; older docs that name metapublic are wrong.

each secjev checkpoint needs the pinned qwen3.5-4b-base backbone and the kev source. `bootstrap.py --inference-only` inside a checkpoint directory downloads both from the upstream sources.

## run the defense

needs python 3.12+, one gpu with 40 gb or more for the reported setup, and an openai api key for the agent model. apple silicon works for scoring at lower speed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "git+https://github.com/ethz-spylab/agentdojo@089ed468cf3ed0322acc66b0211f26d9d90dbf60"
hf download macmacmacmac/SecJev-4B-Spot2-Redo --local-dir checkpoints/SecJev-4B-Spot2-Redo
export SECJEV_WORK="$PWD/secjev-4b-work" OPENAI_API_KEY=...
python run_agent.py --attack ood:bland_note --suites banking,slack --model gpt-5.4-mini \
  --defense action:content:0.35:redact --secjev checkpoints/SecJev-4B-Spot2-Redo \
  --out runs/bland.jsonl
```

## rerun tests and evaluations

```bash
python -m pytest -q tests/test_e2e_components.py tests/test_fastjev_guard.py
bash run_pentest_evals.sh
```

`run_pentest_evals.sh` runs normal tasks, one-try attacks, and 10-round adaptive attacks for four defense settings. the script calls paid agent and attacker apis. the offline path is record, score, replay, analyze: `run_agent.py`, `score.py`, `replay.py`, `analyze.py`, `report.py`. details: `docs/evaluation.md`. programmatic use: `docs/defense-api.md`. design and open work, including the repe probe and steering layers that are specified but not built: `docs/weaponizing-jev-for-defenders.md`.

## limits and license

run attack code only on systems you own or may test. thresholds in the docs were tuned on agentdojo workspace and travel tasks; recalibrate before any other tool set with `fastjev_guard.py`. code and docs: apache-2.0, see `LICENSE`. weights: secjev use and distribution agreement v1.0 plus upstream terms, see each model card.
