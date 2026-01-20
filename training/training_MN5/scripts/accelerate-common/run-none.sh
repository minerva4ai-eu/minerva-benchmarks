#!/bin/bash

#SBATCH --job-name=ACCELERATE_DYNAMIC
#SBATCH --time=24:00:00


##################################################
###           Activate Environment             ###
##################################################
# Activate virtual environment using conda
source activate-env-per-supercomputer.sh $ENVIRONMENT_FINETUNING
# module load $MODULES
# source activate $ENVIRONMENT_FINETUNING
# export PATH=$ENVIRONMENT_FINETUNING/bin:$PATH
# which python

##################################################


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


# Export environment variables
# export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=none
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True

##################################################

##################################################
###           Training Execution              ###
##################################################
# Define GPU monitoring command.
gpu_plots_monitor_command="python -m gpu_plots"

# Start GPU monitoring in background
$gpu_plots_monitor_command &
monitor_pid=\$!

# Optional: give the monitor time to initialize
sleep 5

# Decide number of processes based on GPU_NODE
if [[ "$GPU_NODE" -eq 1 ]]; then
    echo "Launching on 1 GPU"
    accelerate launch \
        --num_processes 1 \
        --num_machines 1 \
        --mixed_precision "$PRECISION" \
        finetune-none.py \
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
elif [[ "$GPU_NODE" -eq 4 ]]; then
    echo "Launching on 4 GPUs (1 node)"
    accelerate launch \
        --num_processes 4 \
        --num_machines 1 \
        --mixed_precision "$PRECISION" \
        finetune-none.py \
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
else
    echo "Error: GPU_NODE must be 1 or 4"
    exit 1
fi

# Kill the GPU monitoring running in background 
kill -SIGTERM \"\$monitor_pid\"

# Wait for the monitor to clean up and exit
wait \"\$monitor_pid\


echo "✅ Single Node job completed."

