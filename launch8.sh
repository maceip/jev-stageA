#!/bin/bash
# 8 action-monitor fine-tunes, one per GPU. name lr replay_frac pos_weight epochs
cd ~; export SECJEV_WORK=~/secjev-4b-work PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
i=0
while read name lr rf pw ep; do
  CUDA_VISIBLE_DEVICES=$i nohup env/bin/python -I ft_action.py --train ft_train.jsonl --dev ft_dev.jsonl --replay replay.jsonl \
    --replay_frac $rf --lr $lr --pos_weight $pw --epochs $ep --batch 32 --checkpointing 1 --tok_budget 16384 --out runs/$name > logs/$name.log 2>&1 &
  i=$((i+1))
done <<'V'
a_lr1e5_r30_pw3 1e-5 0.3 3 2
b_lr3e5_r30_pw3 3e-5 0.3 3 2
c_lr1e4_r30_pw3 1e-4 0.3 3 2
d_lr3e5_r0_pw3 3e-5 0.0 3 2
e_lr3e5_r30_pw1 3e-5 0.3 1 2
f_lr3e5_r50_pw8 3e-5 0.5 8 2
g_lr1e4_r50_pw1 1e-4 0.5 1 2
h_lr3e5_r30_pw3_e3 3e-5 0.3 3 3
V
echo LAUNCHED
