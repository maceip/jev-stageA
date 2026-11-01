#!/bin/bash
source ~/vllm-env/bin/activate
exec vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --port 8000 --gpu-memory-utilization 0.6 --max-num-seqs 64 --max-model-len 65536 --enable-auto-tool-choice --tool-call-parser hermes --override-generation-config '{"max_new_tokens": 4096}'
