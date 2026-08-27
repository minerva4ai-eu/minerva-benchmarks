#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################

echo yaml_path=$1

####################################################

##################################################
###             Torchrun Setup                 ###
##################################################
gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.gpu_plots"

export MASTER_PORT=29500

train_command="${runtime_prefix:+$runtime_prefix} torchrun \
      --nnodes $SLURM_STEP_NUM_NODES --nproc_per_node $SLURM_GPUS_ON_NODE \
      --rdzv_id $SLURM_JOB_ID.$SLURM_STEP_ID --rdzv_backend c10d --rdzv_endpoint ${HEAD_NODE}:${MASTER_PORT} \
      $TRAIN_SCRIPT --yaml $1"

if [[ $ENABLE_COMPILE == "True" || $ENABLE_COMPILE == "true" ]]; then
    train_command="$train_command --enable_compile"
fi

# Launch Run
# Start monitoring in background
$gpu_plots_monitor_command &
monitor_pid=\$!

# Optional: give the monitor time to initialize
sleep 5
$train_command

kill -SIGTERM \"\$monitor_pid\"

# Wait for the monitor to clean up and exit
wait \"\$monitor_pid\"

echo "DDP Job Completed."

