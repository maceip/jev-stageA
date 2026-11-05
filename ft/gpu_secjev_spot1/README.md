---
language:
- en
license: other
license_name: secjev-use-and-distribution-agreement-v1.0
license_link: LICENSE
base_model: /root/secjev-work/base
tags:
- base_model:adapter:/root/secjev-work/base
- lora
- transformers
library_name: peft
---

# SecJev-4B

SecJev-4B scores Boolean judgments, choices and ordered ratings from security evidence and explicit policies. It returns probabilities over candidate answers instead of generating answer text.

This release contains the **trained LoRA adapter, decision head, tokenizer, calibration and inference code**. Download the pinned Qwen3.5-4B-Base backbone separately with the included bootstrap. We adapted official Kev-4B on SecJev-Corpus for three epochs; full development selected epoch 3. Temperature 1.3349681034 is fitted on calibration only.

## Run

```bash
python -m pip install -r SecJev-4B/requirements.txt
export SECJEV_WORK="$PWD/secjev-4b-work"
python SecJev-4B/bootstrap.py --inference-only
python SecJev-4B/infer.py --model ./SecJev-4B --base "$SECJEV_WORK/base" --request SecJev-4B/examples/certificate.json --device cuda
```

The included example predicts `reject`; the actual answer is in example-output.json. Six short development cases (two per type) confirm export parity. They are a packaging check, not an accuracy estimate.

## Data and results

SecJev-Corpus v1.0.0 has 170,185 questions from eight sources and 14 tasks, split into 118,818 training / 14,821 calibration / 14,912 development / 21,634 test questions. [Download the unchanged corpus](https://github.com/UESTC1010/SecJev/releases/download/v1.0.0/SecJev-Corpus-v1.0.0.tar.gz).

On all security test questions: **88.01% micro accuracy, 94.51% macro-task accuracy, 92.42% macro-task balanced accuracy**. Original Kev clean suites: 86.58% decision-v7 and 83.84% transfer-v4. [Full per-task results and all three checkpoints](evaluation/README.md). Vehicle tasks remain weaker; research splits were inspected during corpus construction and are not a new blind deployment benchmark.

## Training and terms

Nine RTX 3090 GPUs, three epochs, LoRA r16/alpha32 across attention/MLP/DeltaNet, 256-dimensional head, FP32 masters and BF16 autocast. [Reproduction code](https://github.com/UESTC1010/SecJev/tree/main/training/secjev4b). Exact initialization and selection records are included in provenance.json and security-training-config.json.

The SecJev Use and Distribution Agreement v1.0 and applicable upstream source/model terms apply. See LICENSE, SOURCES.md and THIRD_PARTY_NOTICES.md. This release does not replace the project agreement with Apache-2.0.
### Framework versions

- PEFT 0.21.2