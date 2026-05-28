#!/bin/bash

#SBATCH --job-name=torch.none
#SBATCH --time=24:00:00


##################################################
###           Activate Environment             ###
##################################################
# Activate virtual environment
source activate-env-per-supercomputer.sh $ENVIRONMENT_FINETUNING

##################################################
###        Environment Variables Setup         ###
##################################################

# Get Arguments
LAUNCH_FOLDER=$1
DATASET=$2
DATASET_PATH=$3
OUTPUT_DIR="${LAUNCH_FOLDER}/output"
mkdir -p $OUTPUT_DIR

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}"
echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"

# Machine-specific env variables (RCCL, ROCR, etc.)
source activate-env-variables-per-supercomputer.sh

export SLURM_CPU_BIND=none

##################################################
###           Training Execution              ###
##################################################
# Define GPU monitoring command.
gpu_plots_monitor_command="python -m gpu_plots"

# Start GPU monitoring in background
$gpu_plots_monitor_command &
monitor_pid=$!

# Optional: give the monitor time to initialize
sleep 5

# Launch training on a single Node
python finetune-none.py \
    --minerva_dir "${CURRENT_DIR}" \
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
    --dataloader_num_workers 32 \
    --dataset $DATASET

# Kill the GPU monitoring running in background 
kill -SIGTERM "$monitor_pid"

# Wait for the monitor to clean up and exit
wait "$monitor_pid"


echo "✅ Single Node job completed."

