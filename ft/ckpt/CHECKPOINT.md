# SecJev-4B-Action-Monitor-C1 (fine-tuned content action monitor, epoch 1)

LoRA adapter + pointer head fine-tuned from SecJev-4B (Qwen3.5-4B-Base @1001bb4d) by `ft_action.py` on
content-view action-authorization questions (train: workspace+travel; held out: banking+slack), lr 3e-5,
no SecJev-Corpus replay, 1 epoch.

**Download:** [MetaPublic/SecJev-4B-Action-Monitor-C1](https://huggingface.co/MetaPublic/SecJev-4B-Action-Monitor-C1)
— a complete checkpoint directory (adapter, head, tokenizer, inference code, pinned bootstrap). The
`adapter_model.safetensors` file (124 MB) is hosted on Hugging Face and is not stored in git.

| File | SHA-256 |
|---|---|
| adapter_model.safetensors | 184fbde48a39ca6a8c3f227518581a24e7af2b478db1dac58e028e455284494c |
| head.pt | 93ec77f5536d29dcd468b21ff519e5c2885ddbe0d685ff05f9ab50d1ce892a51 |
| adapter_config.json, provenance.json | in the Hugging Face repository (`provenance.json` records the fine-tune; `upstream-provenance.json` preserves the SecJev-4B v1.2.0 record) |

To load: download the Hugging Face repository and pass the directory as `--secjev`, or pass the Hub repo id
directly where the Kev checkpoint loader accepts a repo id. Small metadata files in this folder
(`adapter_config.json`, `head.pt`, `provenance.json`) are retained in git for provenance; the Hugging Face
repository is the authoritative complete checkpoint.
