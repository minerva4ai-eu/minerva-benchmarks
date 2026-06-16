#!/bin/bash

#SBATCH --job-name=ACCELERATE_DYNAMIC


##################################################
###           Activate Environment             ###
##################################################
# Activate virtual environment using conda
# source activate-env-per-supercomputer.sh $ENVIRONMENT_FINETUNING
eval "$LOAD_MODULES"
# source activate $ENVIRONMENT_FINETUNING
# export PATH=$ENVIRONMENT_FINETUNING/bin:$PATH
# which python

##################################################


##################################################
###        Environment Variables Setup         ###
##################################################

# Get Arguments
OUTPUT_DIR="${LAUNCH_FOLDER}/output"
# Deprecated script arguments - provided by launch environment
#LAUNCH_FOLDER=$1
#DATASET=$2
#DATASET_PATH=$3
#TRAIN_SCRIPT=$4
mkdir -p $OUTPUT_DIR

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}"
echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"


# Export environment variables
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}


# Torchrun args
export JOB_ID=${SLURM_JOB_ID}
export NNODES=${SLURM_NNODES}
export NPROC_PER_NODE=$GPUS_PER_NODE
export NUM_PROCS=$((NNODES * NPROC_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID

##################################################

##################################################
###           Training Execution              ###
##################################################
# Define GPU monitoring command.
singularity_prefix="singularity exec \
    $SINGULARITY_ARGS \
    $SINGULARITY_BINDS \
    --home $LAUNCH_FOLDER \
    $SINGULARITY_CONTAINER"
echo "singularity prefix $singularity_prefix"


gpu_plots_monitor_command="$singularity_prefix  python -m gpu_plots"

train_command="$singularity_prefix accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    $TRAIN_SCRIPT \
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
        --dataloader_num_workers 8 \
        --dataset '$DATASET' \
        --disable_compile $DISABLE_COMPILE"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --export=ALL bash -c "
    # Start monitoring in background
    $gpu_plots_monitor_command &
    monitor_pid=\$!

    # Optional: give the monitor time to initialize
    sleep 5
    
    # Run training in foreground (this blocks until done)
    $train_command

    kill -SIGTERM \"\$monitor_pid\"

    # Wait for the monitor to clean up and exit
    wait \"\$monitor_pid\"
"

echo "✅ Single Node job completed."
exit 0
