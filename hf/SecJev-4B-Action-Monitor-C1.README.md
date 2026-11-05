---
language:
- en
license: other
license_name: secjev-use-and-distribution-agreement-v1.0
license_link: LICENSE
base_model: Qwen/Qwen3.5-4B-Base
tags:
- security
- prompt-injection
- agent-defense
- decision-model
- lora
- jev
---

# SecJev-4B-Action-Monitor-C1

A [SecJev-4B](https://github.com/UESTC1010/SecJev/releases/tag/v1.2.0) checkpoint fine-tuned to act as an
**action monitor** for tool-using LLM agents: before a proposed tool call executes, the checkpoint scores
whether the call is authorized by the user's request, given the tool outputs the agent has seen.

SecJev-4B-Action-Monitor-C1 is the Spot 2 (outgoing tool-call) half of the JEV defense evaluated in the
[jev-stageA](https://github.com/maceip/jev-stagea) research repository. The Spot 1 (incoming tool-output)
half uses the unmodified upstream SecJev-4B checkpoint, mirrored at
[macmacmacmac/SecJev-4B](https://huggingface.co/macmacmacmac/SecJev-4B). Used together, the two checkpoints form
the `secjev_cascade` defense.

## What changed relative to upstream SecJev-4B

Only `adapter_model.safetensors`, `head.pt`, and `adapter_config.json` differ. The LoRA adapter and pointer
decision head were fine-tuned for one epoch (variant C1: lr 3e-5, no corpus replay, content view) on
action-authorization questions built from recorded AgentDojo runs. Training used the workspace and travel
suites only; banking and slack were held out for reporting. Full details: `provenance.json` in this
repository. Upstream SecJev-4B v1.2.0 records are preserved under `upstream-secjev-4b/` and
`upstream-provenance.json`. This is a modified release; SecJev-4B-Action-Monitor-C1 is not the original
SecJev-4B and must not be cited as the original SecJev-4B.

## Run

The checkpoint is a Kev/SecJev checkpoint directory: LoRA adapter + decision head + tokenizer. The pinned
Qwen3.5-4B-Base backbone and the Kev source code are fetched by the included bootstrap script (the backbone
is distributed from the Qwen Hugging Face repository under Apache-2.0 and is not re-hosted here).

```bash
hf download macmacmacmac/SecJev-4B-Action-Monitor-C1 --local-dir SecJev-4B-Action-Monitor-C1
python -m pip install -r SecJev-4B-Action-Monitor-C1/requirements.txt
export SECJEV_WORK="$PWD/secjev-4b-work"
python SecJev-4B-Action-Monitor-C1/bootstrap.py --inference-only
python SecJev-4B-Action-Monitor-C1/infer.py --model ./SecJev-4B-Action-Monitor-C1 \
  --base "$SECJEV_WORK/base" --request SecJev-4B-Action-Monitor-C1/examples/certificate.json --device cuda
```

To run SecJev-4B-Action-Monitor-C1 as a live agent defense (stop or redact mode, alone or cascaded with the
tool-output filter), use `defenses.py` and `secjev_scorer.py` from the jev-stageA repository and pass this
directory as `--secjev`. The Kev checkpoint loader also accepts the Hub repo id directly
(`macmacmacmac/SecJev-4B-Action-Monitor-C1`).

## Reported results (read the caveats)

AgentDojo v1.2.2, banking+slack held-out suites, `gpt-5.4-mini` agent, bland-note attack set, live runs,
threshold 0.35 tuned on workspace+travel:

| Mode | Normal tasks | Hijacked | Tasks completed under attack |
|---|---:|---:|---:|
| Undefended | 97.3% | 34.1% | 78.3% |
| C1 monitor, stop | 89.9% | 1.6% | 55.4% |
| C1 monitor, redact | 90.5% | 0.8% | 75.5% |
| LLM judge (`gpt-5.4-mini`), redact | 95.3% | 0.0% | 28.5% |

Against a 10-round adaptive attacker (`gpt-5.5`, 100 banking+slack targets), C1 alone is weak: 58% hijacked
in stop mode and 53% in redact mode, versus 64% undefended and 25% for the LLM judge. SecJev-4B-Action-Monitor-C1
is a research checkpoint, not a complete defense. The jev-stageA README reports the full results, including
the decision rules SecJev-4B-Action-Monitor-C1 did not pass.

## License and terms

The SecJev Use and Distribution Agreement v1.0 (`LICENSE`) applies, together with the upstream terms in
`UPSTREAM.md`, `SOURCES.md`, and `THIRD_PARTY_NOTICES.md` (Kev and Qwen components are Apache-2.0). The
upstream agreement's source-specific conditions, including ToN-IoT commercial-use limits in SecJev-Corpus,
are not removed by this release.
