#!/bin/bash

##################################################
###            Setup Environment               ###
##################################################


echo Starting torchrun...


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

dist_args="--nnodes $SLURM_STEP_NUM_NODES --nproc_per_node $SLURM_GPUS_ON_NODE \
    --rdzv_id $SLURM_JOB_ID.$SLURM_STEP_ID --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT}"

if [[ $PARALLELISM == "fsdp" ]]; then
    RESULTSDIR=$RESULTSDIR/$SLURM_JOB_ID.$SLURM_STEP_ID-min-overlap
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


if [[ $PARALLELISM == "none" ]]; then
    python scripts/torchrun-common/finetune-$PARALLELISM.py $ft_args
    # python scripts/torchrun-common/finetune-$PARALLELISM.py $ft_args > $RESULTSDIR/step-$SLURM_JOB_ID.$SLURM_STEP_ID-rank$SLURM_NODEID.out 2>&1
else
    torchrun $dist_args scripts/torchrun-common/finetune-$PARALLELISM.py $ft_args
    # torchrun $dist_args scripts/torchrun-common/finetune-$PARALLELISM.py $ft_args > $RESULTSDIR/step-$SLURM_JOB_ID.$SLURM_STEP_ID-rank$SLURM_NODEID.out 2>&1
fi



# 2 runs for fsdp (overlaps)
if [[ $PARALLELISM == "fsdp" ]]; then
    echo Running second FSDP config...

    ft_args="${ft_args/$RESULTSDIR/"$(dirname "$RESULTSDIR")/$SLURM_JOB_ID.$SLURM_STEP_ID-max-overlap"}"
    ft_args+=" --max_comm_comp_overlap"

    echo ft_args: $ft_args
    torchrun $dist_args scripts/torchrun-common/finetune-$PARALLELISM.py $ft_args
    # torchrun $dist_args scripts/torchrun-common/finetune-$PARALLELISM.py $ft_args >> $RESULTSDIR/step-$SLURM_JOB_ID.$SLURM_STEP_ID-rank$SLURM_NODEID.out 2>&1
fi

echo "Completed."

