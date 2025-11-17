#!/bin/bash
# export HF_HOME='/home/xuxinchen/data/hfhub'
export HF_HOME='/data/hfhub/'
export HF_ENDPOINT='https://hf-mirror.com'

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export TIMEOUT_NCCL_MINUTES=120

# Ascend settings
# export ASCEND_LAUNCH_BLOCKING=1
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# model_name quick reference
LLAMA1B_PATH='/data/hfhub/hub/llama-1b-fresh'
LLAMA150M_PATH='PrimeIntellect/llama-150m-fresh'
# LLAMA1B_PATH='/home/xuxinchen/crosstrain/models/llama-1b-fresh'
# LLAMA150M_PATH='/home/xuxinchen/crosstrain/models/llama-150m-fresh'

# dataset quick reference
# C4EN_PATH='allenai/c4'
C4EN_PATH='/data/hfhub/datasets/c4'
SST2_PATH='glue/sst2'

run_experiment() {
    local args="$@"
    local timestamp=$(date +"%Y%m%d-%H%M%S")
    torchrun --nnodes=1 --nproc_per_node=4 --master_port=29501 train.py \
        --log_dir "logs/sdiloco/$timestamp" \
        $args
}


### LLAMA150M C4EN
# DiLoCo H=50
# model name直接传地址
run_experiment --dataset_name "$C4EN_PATH" --model_name "$LLAMA150M_PATH" \
    --sync_interval 5 --use_nesterov \
    --eval_interval 5 --eval_batch_size 4 --max_eval_batches 400\
    --use_amp --amp_type 'bf16' --total_steps 44000 \
    --checkpoint_interval 5 --checkpoint_dir 'ckpts/diloco_150m' \
    --max_checkpoints 3 \
    --delay_steps 3 \
    --algorithm "diloco" \
    --batch_size 32 \
    --effective_batch_size 256 --resume
    # \
    # --resume
    # --algorithm "streaming" \
    # --algorithm "diloco" \
    # --algorithm "dc" \
    # --simulated_comp_time 0.3 \

### BERT SST2
# DP
# run_experiment --sync_interval 1 --outer_lr 1.0 --use_nesterov

# DiLoCo H=50
# run_experiment --sync_interval 50 --outer_lr 0.7 --use_nesterov
