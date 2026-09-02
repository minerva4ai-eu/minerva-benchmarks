#!/bin/bash

#SBATCH --job-name=MEGATRON_NEMO

## ENV ACTIVATION
##################################################
###           Load HPC modules                 ###
##################################################
module purge
if [ ! -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi

## CODE EXECUTION
set -x
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,COLL,NET,GRAPH,TUNING,NVLS,PROXY
export NCCL_DEBUG_TIMESTAMP_LEVELS=WARN,INFO
export NCCL_DEBUG_FILE=$LAUNCH_FOLDER/slurm-logs/${SLURM_JOB_ID}/${SLURM_JOB_NAME}.nccl
ulimit -s 8192  # issue with thread stack when using a lot of GPUs

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export PYTHONUNBUFFERED=1
export HF_HOME=$LAUNCH_FOLDER
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

# Convert HF ckpt into Megatron ckpt
if [[ ! -e "$MODEL_PATH/nemo_ckpt" || ! -e "$DATASET_PATH/training.jsonl" || ! -e "$DATASET_PATH/validation.jsonl" ]]; then
    echo "######################################"
    echo "#       Running preparation stage    #"
    echo "######################################"
    
    srun --nodes=1 -n 1 singularity exec \
        --nv \
        --bind "$LAUNCH_FOLDER","$DATASET_PATH","$MODEL_PATH" \
        --env CUDA_VISIBLE_DEVICES=0 \
        --env HF_HUB_OFFLINE=1 \
        "$SINGULARITY_CONTAINER" \
        python "$TRAIN_SCRIPT" \
                --prepare \
                --dataset-path "$DATASET_PATH" \
                --dataset-root "$DATASET_PATH" \
                --model-path "$MODEL_PATH" \
                --nemo-ckpt-path "$MODEL_PATH/nemo_ckpt"
fi

echo "######################################"
echo "#     Running Megatron-Nemo train    #"
echo "######################################"
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM * NODES * GPUS_PER_NODE))
time srun --export=ALL singularity exec \
    --nv \
    --bind "$LAUNCH_FOLDER","$DATASET_PATH","$MODEL_PATH" \
    "$SINGULARITY_CONTAINER" \
    torchrun \
        --nnodes="$SLURM_NNODES" \
        --nproc-per-node="$SLURM_GPUS_ON_NODE" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        "$TRAIN_SCRIPT" \
            --global-batch-size "$GLOBAL_BATCH_SIZE" \
            --batch-size "$BATCH_SIZE" \
            --seq-length "$MAX_MODEL_LENGTH" \
            --epochs 1 \
            --test \
            --test-nsteps "$STEPS" \
            --enable-flop-counter \
            --pytorch-profiler \
            --dataset-root "$DATASET_PATH" \
            --model-path "$MODEL_PATH" \
            --nemo-ckpt-path "$MODEL_PATH"/nemo_ckpt \
            --no-activation-checkpointing \
            --no-fp8 \
            --devices-per-node "$SLURM_GPUS_ON_NODE" \
            --num-nodes "$SLURM_NNODES" \
            --dp-size "$DP" \
            --pp-size "$PP" \
            --tp-size "$TP" \
            --cp-size "$CP"

date
