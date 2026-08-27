#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################

echo yaml_path=$1

MASTER_PORT=29500
# export MASTER_ADDR=$HEAD_NODE
NUM_PROCS=$(($SLURM_STEP_NUM_NODES * SLURM_GPUS_ON_NODE))

gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.gpu_plots"

# accelerate_config_path="scripts/accelerate-common/accelerate_config.yaml"

# # Update Accelerate config placeholders
# sed -i "s/{{MASTER_IP}}/$HEAD_NODE/g" "$accelerate_config_path"
# sed -i "s/{{NUM_NODES}}/$SLURM_STEP_NUM_NODES/g" "$accelerate_config_path"
# sed -i "s/{{NUM_GPUS}}/$NUM_PROCS/g" "$accelerate_config_path"
# sed -i "s/machine_rank: 0/machine_rank: $SLURM_NODEID/g" "$accelerate_config_path"

train_command_max_overlap="${runtime_prefix:+$runtime_prefix} accelerate launch \
    --multi-gpu \
    --machine_rank $SLURM_NODEID \
    --rdzv_backend c10d \
    --main_process_ip $HEAD_NODE \
    --main_process_port $MASTER_PORT \
    --num_processes $NUM_PROCS \
    --num_machines $SLURM_STEP_NUM_NODES \
      $TRAIN_SCRIPT --yaml $1 --max_comm_comp_overlap"

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

