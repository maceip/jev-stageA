#!/bin/bash
AGENT=${1:-astra}; MODEL=${2:-gpt-6-astra}
cd ~/jev-stageA
source ~/secjev-env/bin/activate; export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1 PYTHONPATH=~/agentdojo/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { python adaptive.py --pairs adaptive_pairs.json --defense $1 --secjev ~/SecJev/SecJev-4B --model $MODEL \
          --attacker-model $MODEL --rounds 10 --workers 8 --out adaptive/${AGENT}_$1.jsonl > logs/adapt_${AGENT}_$1.log 2>&1; }
run llm_judge &
run secjev_action_meta; run secjev_action_content; run secjev_cascade
wait
for d in llm_judge secjev_action_meta secjev_action_content secjev_cascade; do run $d; done   # second pass reruns harness-error pairs only
echo MORE_DONE $(date -u)
