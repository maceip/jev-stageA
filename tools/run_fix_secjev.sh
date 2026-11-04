#!/bin/bash
# Re-run failed Tasks A, B, C with SECJEV_WORK set correctly.
# These failed because run_remaining.sh didn't export SECJEV_WORK.
# This script waits for CLEF seed 2 to finish, then runs the SecJev tasks.
set -e
exec > /root/work/fix_secjev_tasks.log 2>&1
echo "=== FIX SECJEV TASKS START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

export SECJEV_WORK=/root/work/secjev-4b-work
echo "SECJEV_WORK=$SECJEV_WORK"

. /root/venv/bin/activate
cd /root/work/jev-stageA

DATA=/root/work/data/redo_v2_out
INIT=/root/work/checkpoints/SecJev-4B
SEED1_CKPT=/root/work/ft/secjev_v03_fla_seed1/epoch2

# Wait for CLEF seed 2 to finish (it's using the GPU now)
echo "Waiting for CLEF seed 2 to finish..."
while pgrep -f "ft_clef.py.*seed 20261102" > /dev/null 2>&1; do
    sleep 60
done
echo "CLEF seed 2 finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- TASK A: SecJev seed 2 ----
echo "=== TASK A: SecJev seed 2 retrain $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
SECJEV_SEED2=/root/work/ft/secjev_v03_fla_seed2
mkdir -p "$SECJEV_SEED2"

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
  --seed 20261102 \
  2>&1 | tail -30

echo "SecJev seed2 frozen scoring..."
python3 eval_secjev_frozen.py score \
  --weights "$SECJEV_SEED2/epoch3" \
  --device cuda \
  --split "$DATA/ft_spot2v2_test.jsonl" \
  --out "$SECJEV_SEED2/test_scores.jsonl" \
  2>&1 | tail -10

echo "TASK A DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- TASK B: Adaptive variants (score seed1 at 4 thresholds on test) ----
echo "=== TASK B: Adaptive variants $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ADAPT=/root/work/adaptive_variants
mkdir -p "$ADAPT"
for THR in 0.01 0.05 0.1 0.5; do
    NAME=$(echo $THR | tr . _)
    echo "Scoring at threshold=$THR..."
    python3 eval_secjev_frozen.py score \
      --weights "$SEED1_CKPT" \
      --device cuda \
      --split "$DATA/ft_spot2v2_test.jsonl" \
      --out "$ADAPT/thr_${NAME}_scores.jsonl" \
      2>&1 | tail -5
done
echo "TASK B DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- TASK C: Ultraspeed arm (base model zero-shot) ----
echo "=== TASK C: Ultraspeed arm (base zero-shot) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 eval_secjev_frozen.py score \
  --weights "$INIT" \
  --device cuda \
  --split "$DATA/ft_spot2v2_test.jsonl" \
  --out /root/work/ultraspeed_arm_scores.jsonl \
  2>&1 | tail -10
echo "TASK C DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== ALL FIX TASKS COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
df -h / | tail -1
