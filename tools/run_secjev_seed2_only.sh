#!/bin/bash
# Launch ONLY SecJev seed 2 retrain (Task A) in parallel with everything else.
set -e
exec > /root/work/secjev_seed2_retrain.log 2>&1
echo "=== SECJEV SEED2 RETRAIN PARALLEL START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

export SECJEV_WORK=/root/work/secjev-4b-work

. /root/venv/bin/activate
cd /root/work/jev-stageA

DATA=/root/work/data/redo_v2_out
INIT=/root/work/checkpoints/SecJev-4B
SECJEV_SEED2=/root/work/ft/secjev_v03_fla_seed2
rm -rf "$SECJEV_SEED2"

python3 ft_action.py \
  --init "$INIT" \
  --train "$DATA/ft_spot2v2_train.jsonl" \
  --dev "$DATA/ft_spot2v2_dev.jsonl" \
  --out "$SECJEV_SEED2" \
  --device cuda \
  --dtype bf16 \
  --epochs 3 \
  --batch 16 \
  --lr 3e-5 \
  --seed 20261102

echo "SecJev seed2 frozen scoring..."
python3 eval_secjev_frozen.py score \
  --weights "$SECJEV_SEED2/epoch3" \
  --device cuda \
  --split "$DATA/ft_spot2v2_test.jsonl" \
  --out "$SECJEV_SEED2/test_scores.jsonl" \
  2>&1 | tail -10

echo "SECJEV SEED2 RETRAIN COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
