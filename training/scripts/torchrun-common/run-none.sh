#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################

# # Get Arguments
# OUTPUT_DIR="${LAUNCH_FOLDER}/output"
# mkdir -p $OUTPUT_DIR

# Print Arguments Received
echo yaml_path=$1
# echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}"
# echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"


# # Export environment variables
# # export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
# export SLURM_CPU_BIND=none

# Torchrun args
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID

###################################################

##################################################
###           Training Execution              ###
##################################################
# Define GPU monitoring command.


gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.gpu_plots"

train_command="${runtime_prefix:+$runtime_prefix} python $TRAIN_SCRIPT --yaml $1"

# Launch Run

# Start monitoring in background
$gpu_plots_monitor_command &
monitor_pid=\$!

# Optional: give the monitor time to initialize
sleep 5

# # TODO: Check
# export RANK=0
# export LOCAL_RANK=0
# export WORLD_SIZE=1
# Run training in foreground (this blocks until done)
$train_command

kill -SIGTERM \"\$monitor_pid\"

# Wait for the monitor to clean up and exit
wait \"\$monitor_pid\"


echo "✅ Single Node job completed."

