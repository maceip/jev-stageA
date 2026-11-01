#!/bin/bash
# Adaptive attacker vs gpt-6.1-astra agent (only model used; no Qwen or gpt-oss-20b), 60 targets x up to 10 rounds, stop mode.
cd ~/jev-stageA; mkdir -p adaptive
export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPENAI_API_KEY="$1"
run() { ~/env/bin/python adaptive.py --pairs adaptive_pairs.json --defense $1 --threshold ${2:-0.5} --secjev ~/SecJev/SecJev-4B \
   --model gpt-6.1-astra --agent-url https://api.openai.com/v1 --attacker-url https://api.openai.com/v1 --attacker-model gpt-6.1-astra \
   --judge-url https://api.openai.com/v1 --judge-model gpt-6.1-astra --rounds 10 --workers 12 --out adaptive/astra_$1.jsonl > adapt_$1.log 2>&1; echo "done $1 $(date -u)"; }
run none & run llm_judge & run piguard &
until grep -q SCORED score_benign.log 2>/dev/null; do sleep 15; done   # GPU free of the benign SecJev scorer
run secjev_tool
run secjev_action_meta
run secjev_action_content
run secjev_cascade
run secjev_cascade_judge
wait
echo ADAPT3_DONE $(date -u)
