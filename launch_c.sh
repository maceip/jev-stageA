cd ~; export SECJEV_WORK=~/secjev-4b-work PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L(){ CUDA_VISIBLE_DEVICES=$1 nohup env/bin/python -I ft_action.py --train fc_train.jsonl --dev fc_dev.jsonl --replay replay.jsonl --replay_frac $3 --lr $4 --pos_weight $5 --epochs 2 --batch 32 --checkpointing 1 --tok_budget 8192 --out runs/$2 > logs/$2.log 2>&1 & }
L 0 C1_lr3e5_r0_pw3 0.0 3e-5 3
L 5 C2_lr3e5_r30_pw3 0.3 3e-5 3
L 4 C3_lr1e4_r0_pw3 0.0 1e-4 3
