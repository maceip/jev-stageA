#!/bin/bash
cd ~/jev-stageA; export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPENAI_API_KEY="$1"
R(){ ~/env/bin/python run_agent.py --attack benign --suites banking,slack --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
   --defense "$1" --secjev "$2" --judge-url https://api.openai.com/v1 --judge-model gpt-5.4-mini --workers 12 --out runs/live/$3_r$4.jsonl > runs/live/$3_r$4.log 2>&1; }
for r in 1 2 3 4; do
  R action:content:0.35:redact ~/ckpt/C1_e1 c1content035redact $r &
  R tool:llm_judge:0.5:redact ~/SecJev/SecJev-4B judgeredact $r &
  R tool:secjev_tool:0.99:redact ~/SecJev/SecJev-4B sjtool099redact $r &
  wait
done; echo LIVE_NR_DONE
