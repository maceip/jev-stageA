# Fine-tuned SecJev content action monitor (C1, epoch 1)

LoRA adapter + pointer head on SecJev-4B (Qwen3.5-4B-Base @1001bb4d). Trained by ft_action.py on content-view
action-authorization questions (train: workspace+travel; held out: banking+slack), lr 3e-5, no replay, 1 epoch.

| File | Where | SHA-256 |
|---|---|---|
| adapter_model.safetensors (124 MB, over GitHub's 100 MB file limit) | GitHub release `ckpt-c1-e1` asset | 184fbde48a39ca6a8c3f227518581a24e7af2b478db1dac58e028e455284494c |
| adapter_config.json, head.pt, provenance.json | this folder | see git |

To load: copy the released SecJev-4B directory, replace these files, and pass the directory as --secjev.
