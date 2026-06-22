#!/bin/bash

#SBATCH --job-name=PYTORCH_DYNAMIC


##################################################
###            Setup Environment               ###
##################################################
if [ ! -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi
source shared/runtime_environment.sh
training_activate_runtime_environment

# Get Arguments
OUTPUT_DIR="${LAUNCH_FOLDER}/output"
mkdir -p $OUTPUT_DIR

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}"
echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"


# Export environment variables
# export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=none

# Torchrun args
export JOB_ID=${SLURM_JOB_ID}
export NNODES=${SLURM_NNODES}
export NPROC_PER_NODE=$GPUS_PER_NODE
export NUM_PROCS=$((NNODES * NPROC_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
###################################################

##################################################
###           Training Execution              ###
##################################################
# Define GPU monitoring command.
runtime_prefix="$(training_build_runtime_prefix)"
echo "Will apply runtime prefix '$runtime_prefix'"


gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m gpu_plots"

train_command="${runtime_prefix:+$runtime_prefix} python $TRAIN_SCRIPT \
        --model $MODEL_PATH \
        --data $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        ${EPOCHS:+--epochs "$EPOCHS"} \
        ${STEPS:+--max_steps "$STEPS"} \
        --max_length $MAX_MODEL_LENGTH \
        --precision $PRECISION \
        --lr $LR \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --dataloader_num_workers 2 \
        --dataset '$DATASET'"

if [[ $DISABLE_COMPILE == "True" || $DISABLE_COMPILE == "true" ]]; then
    train_command="$train_command --enable_compile"
fi

# Launch Run
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --export=ALL bash -c "
    # Start monitoring in background
    $gpu_plots_monitor_command &
    monitor_pid=\$!

    # Optional: give the monitor time to initialize
    sleep 5

    export RANK=0
    export LOCAL_RANK=0
    export WORLD_SIZE=1
    # Run training in foreground (this blocks until done)
    $train_command

    kill -SIGTERM \"\$monitor_pid\"

    # Wait for the monitor to clean up and exit
    wait \"\$monitor_pid\"
"


echo "✅ Single Node job completed."

