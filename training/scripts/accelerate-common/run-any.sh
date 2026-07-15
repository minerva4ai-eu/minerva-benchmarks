#!/bin/bash

##################################################
###            Setup Environment               ###
##################################################

echo Starting accelerate launch...


# Print Arguments Received
echo "LAUNCH_FOLDER: {$RESULTSDIR}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}"
echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$SLURM_GPUS_ON_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"


##################################################
###             Torchrun Setup                 ###
##################################################

# NOTE: SLURM_PROCID is the same as SLURM_NODEID
export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT=29500
# Is this used in the python script?
export NODE_RANK=$SLURM_PROCID

# echo "NODE_RANK: {$NODE_RANK}"
echo "SLURM_STEP_NUM_NODES: {$SLURM_STEP_NUM_NODES}"
# echo "NUM_PROCS: {$NUM_PROCS}"
echo "MASTER_ADDR: {$MASTER_ADDR}"
echo "MASTER_PORT: {$MASTER_PORT}"
echo "PARALLELISM: {$PARALLELISM}"


dist_args="--multi-gpu \
    --machine_rank $SLURM_NODEID \
    --rdzv_backend c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_processes $(( $SLURM_STEP_NUM_NODES * $SLURM_GPUS_ON_NODE )) \
    --num_machines $SLURM_STEP_NUM_NODES"

if [[ $PARALLELISM == "fsdp" ]]; then
    RESULTSDIR=$RESULTSDIR/$SLURM_JOB_ID-min-overlap
fi

ft_args="--model $MODEL_PATH \
        --data $DATASET_PATH \
        --output_dir $RESULTSDIR \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_MODEL_LENGTH \
        ${STEPS:+--max_steps "$STEPS"} \
        --precision $PRECISION \
        --lr $LR \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --dataloader_num_workers 4 \
        --dataset $DATASET"
# ${EPOCHS:+--epochs "$EPOCHS"} \

if [[ $ENABLE_COMPILE == "True" || $ENABLE_COMPILE == "true" ]]; then
    train_command="$train_command --enable_compile"
fi

echo dist_args: $dist_args
# echo ft_script: $ft_script
echo ft_args: $ft_args

accelerate launch $dist_args scripts/accelerate-common/finetune-$PARALLELISM.py $ft_args

# 2 runs for fsdp (overlaps)
if [[ $PARALLELISM == "fsdp" ]]; then

    ft_args="${ft_args/$RESULTSDIR/"$(dirname "$RESULTSDIR")/$SLURM_JOB_ID-max-overlap"}"
    ft_args+=" --max_comm_comp_overlap"

    echo ft_args: $ft_args
    accelerate launch $dist_args scripts/accelerate-common/finetune-$PARALLELISM.py $ft_args
fi

echo "Completed."