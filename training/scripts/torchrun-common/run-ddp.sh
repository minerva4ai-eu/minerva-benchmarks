#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################

echo yaml_path=$1

source shared/runtime_environment.sh
training_activate_runtime_environment

##################################################
###             Torchrun Setup                 ###
##################################################
gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m shared.gpu_plots"

export MASTER_PORT=29500

train_command="${runtime_prefix:+$runtime_prefix} torchrun \
      --nnodes $SLURM_STEP_NUM_NODES \
      --nproc_per_node $SLURM_GPUS_ON_NODE \
      --rdzv_id $SLURM_JOB_ID.$SLURM_STEP_ID \
      --rdzv_backend c10d \
      --rdzv_endpoint ${HEAD_NODE}:${MASTER_PORT} \
      $TRAIN_SCRIPT --yaml $1"

prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m shared.prepare --yaml $1"

echo "######################################"
echo "#       Running preparation stage    #"
echo "######################################"
    
srun --nodes=1 --ntasks=1 --export=ALL $prepare_train_command

echo "######################################"
echo "#     Running  Torchrun-DDP train    #"
echo "######################################"

# Launch Run
# Start monitoring in background
$gpu_plots_monitor_command &
monitor_pid=$!

# Optional: give the monitor time to initialize
sleep 5
$train_command

kill -SIGTERM "$monitor_pid"

# Wait for the monitor to clean up and exit
wait "$monitor_pid"

echo "DDP Job Completed."

