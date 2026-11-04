#!/bin/bash
# Run SecJev tasks A, B, C IN PARALLEL with CLEF seed 2.
# GPU has 309 GB VRAM, CLEF uses only 4.7 GB. SecJev 4B model is ~8 GB in bf16.
# Plenty of room for both.
set -e
exec > /root/work/fix_secjev_tasks.log 2>&1
echo "=== SECJEV TASKS PARALLEL START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

export SECJEV_WORK=/root/work/secjev-4b-work
echo "SECJEV_WORK=$SECJEV_WORK"

. /root/venv/bin/activate
cd /root/work/jev-stageA

DATA=/root/work/data/redo_v2_out
INIT=/root/work/checkpoints/SecJev-4B
SEED1_CKPT=/root/work/ft/secjev_v03_fla_seed1/epoch2

# ---- TASK C first: Ultraspeed arm (base model zero-shot, fastest, just scoring) ----
echo "=== TASK C: Ultraspeed arm (base zero-shot) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 eval_secjev_frozen.py score \
  --weights "$INIT" \
  --device cuda \
  --split "$DATA/ft_spot2v2_test.jsonl" \
  --out /root/work/ultraspeed_arm_scores.jsonl \
  2>&1 | tail -10
echo "TASK C DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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

# ---- TASK A: handled by run_secjev_seed2_only.sh in parallel ----
echo "=== TASK A: skipped (running as separate process) ==="

echo "=== ALL SECJEV TASKS COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
df -h / | tail -1
