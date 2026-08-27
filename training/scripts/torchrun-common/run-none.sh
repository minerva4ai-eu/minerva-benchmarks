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


gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.gpu_plots"

train_command="${runtime_prefix:+$runtime_prefix} python $TRAIN_SCRIPT --yaml $1"

# Launch Run

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


echo "✅ Single Node job completed."

