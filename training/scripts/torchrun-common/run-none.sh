#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################

echo yaml_path=$1

# Torchrun args
export MASTER_PORT=29500
# export NODE_RANK=$SLURM_PROCID

###################################################

##################################################
###           Training Execution              ###
##################################################
# Define GPU monitoring command.


gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m shared.gpu_plots"

train_command="${runtime_prefix:+$runtime_prefix} python $TRAIN_SCRIPT --yaml $1"

echo "ENABLE_COMPILE: $ENABLE_COMPILE"
if [[ $DISABLE_COMPILE == "True" || $DISABLE_COMPILE == "true" ]]; then
    train_command="$train_command --enable_compile"
fi

prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m shared.prepare \
        --model $MODEL_PATH \
        --data $DATASET_PATH \
        --dataset $DATASET \
        --output_dir $OUTPUT_DIR/$SLURM_JOB_ID \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_MODEL_LENGTH "

srun --nodes=1 --ntasks=1 --export=ALL $prepare_train_command

# Launch Run

# Start monitoring in background
$gpu_plots_monitor_command &
monitor_pid=$!

# Optional: give the monitor time to initialize
sleep 5

# Run training in foreground (this blocks until done)
$train_command

kill -SIGTERM "$monitor_pid"

# Wait for the monitor to clean up and exit
wait "$monitor_pid"


echo "✅ Single Node single GPU job completed."

