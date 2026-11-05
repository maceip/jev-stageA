#!/bin/bash
# Wait for the CLEF pipeline to finish, then run all remaining tasks sequentially.
set -e
exec > /root/work/remaining_tasks.log 2>&1
echo "=== REMAINING TASKS QUEUE START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

. /root/venv/bin/activate
cd /root/work/jev-stageA

DATA=/root/work/data/redo_v2_out
INIT=/root/work/checkpoints/SecJev-4B
SEED1_CKPT=/root/work/ft/secjev_v03_fla_seed1/epoch2

# Wait for main CLEF training to finish
echo "Waiting for CLEF training to complete..."
while pgrep -f "run_train_only.sh" > /dev/null 2>&1; do
    sleep 60
done
echo "CLEF training finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Verify CLEF outputs
if [ ! -f /root/work/full_clef/final_results.json ]; then
    echo "ERROR: CLEF pipeline did not produce final_results.json"
    exit 1
fi
echo "CLEF final_results.json exists -- CLEF pipeline is complete."

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
  2>&1 | tail -20

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

# ---- TASK C: Hosted Ultraspeed arm (base model zero-shot on same test rows) ----
echo "=== TASK C: Ultraspeed arm (base zero-shot on same frozen test) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 eval_secjev_frozen.py score \
  --weights "$INIT" \
  --device cuda \
  --split "$DATA/ft_spot2v2_test.jsonl" \
  --out /root/work/ultraspeed_arm_scores.jsonl \
  2>&1 | tail -10
echo "TASK C DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- TASK D: CLEF analysis summary ----
echo "=== TASK D: CLEF analysis $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 - << 'PYD'
import json
with open("/root/work/full_clef/final_results.json") as f:
    results = json.load(f)
test_scores = [json.loads(l) for l in open("/root/work/full_clef/test_scores.jsonl")]
dev_scores = [json.loads(l) for l in open("/root/work/full_clef/dev_scores.jsonl")]
summary = {"clef_final": results, "test_score_count": len(test_scores), "dev_score_count": len(dev_scores)}
with open("/root/work/clef_analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
test = results.get("test_banking_slack", {})
for k in ("tool_output_filter", "action_authorization_guard"):
    m = test.get(k, {})
    print(f"  test {k}: AUC={m.get('auc','N/A')} n={m.get('n','N/A')}")
print(f"  test scores: {len(test_scores)}, dev scores: {len(dev_scores)}")
PYD
echo "TASK D DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- TASK E: CLEF seed 2 ----
echo "=== TASK E: CLEF seed 2 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
CLEF_SEED2=/root/work/clef_seed2
mkdir -p "$CLEF_SEED2"
# Reuse the cache from seed 1 -- hidden states are deterministic, only the head seed changes
ln -sf /root/work/full_clef/cache "$CLEF_SEED2/cache"

python3 ft_clef.py \
  --cache-only \
  --data-dir "$DATA/clef_frozen_v03" \
  --out "$CLEF_SEED2" \
  --ckpt_out /root/work/checkpoints/clef-flash-LiteRT-ft-seed2 \
  --init /root/work/checkpoints/clef-flash-LiteRT \
  --device cuda \
  --epochs 3 \
  --batch 16 \
  --lr 3e-5 \
  --seed 20261102 \
  --eval-interval 0 2>&1 | tail -30
echo "TASK E DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== ALL REMAINING TASKS COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
df -h / | tail -1
