#!/bin/bash
# Rerun harness-error pairs for every adaptive arm of gpt-6-astra (successful pairs are kept as final).
AGENT=${1:-astra}; MODEL=${2:-gpt-6-astra}; WORKERS=${3:-8}
cd ~/jev-stageA
source ~/secjev-env/bin/activate; export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1 PYTHONPATH=~/agentdojo/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for d in none piguard deberta secjev_tool secjev_action_meta secjev_action_content secjev_cascade; do
  python adaptive.py --pairs adaptive_pairs.json --defense $d --secjev ~/SecJev/SecJev-4B --model $MODEL \
    --attacker-model $MODEL --rounds 10 --workers $WORKERS --out adaptive/${AGENT}_$d.jsonl > logs/adapt_rerun_${AGENT}_$d.log 2>&1
  echo "rerun $d $(tail -1 logs/adapt_rerun_${AGENT}_$d.log) $(date -u)"
done
echo RERUN_DONE $(date -u)
