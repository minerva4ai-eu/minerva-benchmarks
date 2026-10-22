#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################
module purge
if [ ! -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi
source scripts/shared/runtime_environment.sh
training_activate_runtime_environment

echo "EXECUTION_MODE: $EXECUTION_MODE"
runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix: $runtime_prefix"

echo yaml_path=$1

NUM_PROCS=$(($SLURM_STEP_NUM_NODES * SLURM_GPUS_ON_NODE))

gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.gpu_plots"
echo "EXECUTION_MODE: $EXECUTION_MODE"

train_command="${runtime_prefix:+$runtime_prefix} accelerate launch \
    --multi-gpu \
    --machine_rank $SLURM_NODEID \
    --rdzv_backend c10d \
    --main_process_ip $HEAD_NODE \
    --main_process_port $HEAD_PORT \
    --num_processes $NUM_PROCS \
    --num_machines $SLURM_STEP_NUM_NODES \
      $TRAIN_MODULE --yaml $1"

prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.prepare --yaml $1"

echo "######################################"
echo "#       Running preparation stage    #"
echo "######################################"
    
$prepare_train_command

echo "######################################"
echo "#     Running  Accelerate-DDP train  #"
echo "######################################"

# Start monitoring in background
$gpu_plots_monitor_command &
monitor_pid=$!

# Optional: give the monitor time to initialize
sleep 5

# Each repeat is a fresh process; a failed repeat does not stop the next ones
exit_code=0
for (( repeatid=0; repeatid<${REPEATS:-1}; repeatid++ )); do
    echo "Running repeatid: $repeatid"
    REPEAT_ID=$repeatid $train_command || {
        rc=$?
        echo "repeat $repeatid failed with exit code $rc"
        exit_code=$rc
    }
done

# Wait for the monitor to clean up and exit
kill -SIGTERM "$monitor_pid"
wait "$monitor_pid"

echo "DDP Job Completed."
exit "$exit_code"
