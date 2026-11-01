#!/bin/bash
# Redact mode, live: gpt-5.4-mini agent, banking+slack, normal tasks + bland-note attack, three defenses.
cd ~/jev-stageA; mkdir -p runs/mini_redact
until grep -q SETUP_DONE ~/setup3.log; do sleep 10; done
export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPENAI_API_KEY="$1"
run() { d=$1; tag=$(echo $d | tr ":." "__")
  for A in benign ood:bland_note; do N=$(echo $A | tr ":" "_")
    ~/env/bin/python run_agent.py --attack $A --suites banking,slack --base-url https://api.openai.com/v1 --model gpt-5.4-mini \
      --reasoning low --workers 24 --defense $d --secjev ~/SecJev/SecJev-4B --judge-url https://api.openai.com/v1 --judge-model gpt-5.4-mini \
      --out runs/mini_redact/${tag}__$N.jsonl > log_$tag_$N.log 2>&1
  done; echo "done $d $(date -u)"; }
run tool:secjev_tool:0.99:redact & run tool:llm_judge:0.5:redact & run action:meta:0.55:redact & wait
echo REDACT_DONE $(date -u)
