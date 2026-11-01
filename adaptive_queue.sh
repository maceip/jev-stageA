#!/bin/bash
# Adaptive-attacker arms for gpt-6.1-astra (only model used; no Qwen or gpt-oss-20b).
AGENT=${1:-astra}; MODEL=${2:-gpt-6.1-astra}; WAITLOG=$3; WAITPAT=$4; SKIPNONE=${5:-0}
cd ~/jev-stageA; mkdir -p adaptive logs
source ~/secjev-env/bin/activate; export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1 PYTHONPATH=~/agentdojo/src
run() { python adaptive.py --pairs adaptive_pairs.json --defense $1 --secjev ~/SecJev/SecJev-4B --model $MODEL \
          --attacker-model $MODEL --rounds 10 --workers 16 --out adaptive/${AGENT}_$1.jsonl > logs/adapt_${AGENT}_$1.log 2>&1; }
[ "$SKIPNONE" = 1 ] || run none
until grep -q "$WAITPAT" $WAITLOG 2>/dev/null; do sleep 30; done
run secjev_tool & run secjev_cascade & run piguard & run deberta & wait
echo ADAPTIVE_DONE $(date -u)
