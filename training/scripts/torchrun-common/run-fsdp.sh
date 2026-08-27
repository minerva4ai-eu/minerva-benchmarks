#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################

echo yaml_path=$1

###################################################

##################################################
###             Torchrun Setup                 ###
##################################################

export MASTER_PORT=29500

gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.gpu_plots"

train_command_max_overlap="${runtime_prefix:+$runtime_prefix} torchrun \
    --nnodes $SLURM_STEP_NUM_NODES --nproc_per_node $SLURM_GPUS_ON_NODE \
    --rdzv_id $SLURM_JOB_ID.$SLURM_STEP_ID --rdzv_backend c10d --rdzv_endpoint ${HEAD_NODE}:${MASTER_PORT} \
    $TRAIN_SCRIPT --yaml $1 --max_comm_comp_overlap"

# Launch Run
# Start monitoring in background
$gpu_plots_monitor_command &
monitor_pid=\$!

# Optional: give the monitor time to initialize
sleep 5

# Run training in foreground (this blocks until done)

$train_command_max_overlap

kill -SIGTERM \"\$monitor_pid\"

# Wait for the monitor to clean up and exit
wait \"\$monitor_pid\"

echo "FSDP Job Completed."

