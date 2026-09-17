#!/bin/bash

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

# Convert HF ckpt into Megatron ckpt
if [[ "$SLURM_NODEID" -eq 0 ]]; then
    if [[ ! -e "$MODEL_PATH/nemo_ckpt" || ! -e "$DATASET_PATH/training.jsonl" || ! -e "$DATASET_PATH/validation.jsonl" ]]; then
        echo "######################################"
        echo "#       Running preparation stage    #"
        echo "######################################"
        
        singularity exec \
            --nv \
            --env CUDA_VISIBLE_DEVICES="0" \
            --env HF_HUB_OFFLINE="1" \
            "$SINGULARITY_CONTAINER" \
            python $TRAIN_MODULE --yaml "$1" --prepare
    fi
fi

echo "######################################"
echo "#     Running Megatron-Nemo train    #"
echo "######################################"
singularity exec \
    --nv \
    "$SINGULARITY_CONTAINER" \
    torchrun \
        --nnodes="$SLURM_NNODES" \
        --nproc-per-node="$SLURM_GPUS_ON_NODE" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$HEAD_NODE:$HEAD_PORT" \
        $TRAIN_MODULE --yaml "$1"
        # --enable-flop-counter 

date
