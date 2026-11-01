#!/bin/bash
cd ~/jev-stageA; mkdir -p runs/live
export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPENAI_API_KEY="$1"
R(){ ~/env/bin/python run_agent.py --attack ood:bland_note --suites banking,slack --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
   --defense "$1" --secjev "$2" --judge-url https://api.openai.com/v1 --judge-model gpt-5.4-mini --workers 16 --out runs/live/bland_$3.jsonl > runs/live/bland_$3.log 2>&1; }
R action:content:0.35:stop ~/ckpt/C1_e1 c1content035_stop &
R action:content:0.35:redact ~/ckpt/C1_e1 c1content035_redact &
R tool:llm_judge:0.5:redact ~/SecJev/SecJev-4B judge_redact &
R none ~/SecJev/SecJev-4B none &
wait; echo LIVE_ATTACK_DONE
