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

NUM_PROCS=$((SLURM_STEP_NUM_NODES * SLURM_GPUS_ON_NODE))

NNODES=${SLURM_NNODES}
NPROC_PER_NODE=$GPUS_PER_NODE
NUM_PROCS=$((NNODES * NPROC_PER_NODE))

# ------------------------------------------------------------------
# Generate DeepSpeed hostfile from SLURM node list
# DeepSpeed requires a hostfile when num_nodes > 1
# ------------------------------------------------------------------# 1. Rank 0 creates directory and performs setup

# Define a barrier file on a SHARED filestem accessible by all nodes
BARRIER_DIR="$SRUN_LOGS"
BARRIER_FILE="${BARRIER_DIR}/.deepspeed-rak0-$SLURM_JOB_ID.$SLURM_STEP_ID.lock"

# Use SLURM_STEP_NODELIST (step allocation) instead of SLURM_NODELIST (job allocation)
# Fallback GPUS_PER_NODE to SLURM_GPUS_ON_NODE
GPUS_PER_NODE=${GPUS_PER_NODE:-$SLURM_GPUS_ON_NODE}

# HOSTFILE must be on a shared filesystem so all nodes can read it
HOSTFILE="${SRUN_LOGS}/.deepspeed-hostfile-$SLURM_JOB_ID.$SLURM_STEP_ID"
deepspeed_config_path="scripts/deepspeed_common/configs/${ZERO_STAGE}.json"
tmp_deepspeed_config_path="${SRUN_LOGS}/.deepspeed-config-$SLURM_JOB_ID.$SLURM_STEP_ID.json"

if [ "${SLURM_PROCID}" -eq 0 ]; then
    touch "$BARRIER_FILE"
    cp "$deepspeed_config_path" "$tmp_deepspeed_config_path"

    scontrol show hostnames "${SLURM_STEP_NODELIST}" | while read -r hostname; do
        echo "${hostname} slots=${GPUS_PER_NODE}" >> "$HOSTFILE"
    done
    echo "Generated DeepSpeed hostfile at: $HOSTFILE"
    cat "$HOSTFILE"

    # Update hpZ partition size for parallelism of stage2 and stage3 configurations only, as stage1 does not use hpZ
    HPZ_PARTITION_SIZE=$((SLURM_STEP_NUM_NODES * GPUS_PER_NODE))
    sed -i "s/\"zero_hpz_partition_size\": \"{{HPZ_PARTITION_SIZE}}\"/\"zero_hpz_partition_size\": $HPZ_PARTITION_SIZE/g" "$tmp_deepspeed_config_path"
    echo "Using hpZ partition size: $HPZ_PARTITION_SIZE"
    rm "$BARRIER_FILE"
else
    echo "[Rank ${SLURM_PROCID}] Waiting for Rank 0..."
    sleep 2
    # Loop and check every 2 seconds until Rank 0 finishes
    while [ -f "${BARRIER_FILE}" ]; do
        sleep 2
    done
    echo "[Rank ${SLURM_PROCID}] Rank 0 unlocked, resuming..."
fi

runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix $runtime_prefix"

gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.gpu_plots"

train_command="${runtime_prefix:+$runtime_prefix} deepspeed \
    --no_ssh \
    --hostfile $HOSTFILE \
    --num_nodes $SLURM_NNODES \
    --num_gpus $SLURM_GPUS_ON_NODE \
    --node_rank $SLURM_NODEID \
    --master_addr $HEAD_NODE \
    --master_port $HEAD_PORT \
    $TRAIN_SCRIPT --yaml $1 --deepspeed_config_file $tmp_deepspeed_config_path"

echo "ENABLE_COMPILE: $ENABLE_COMPILE"
if [[ $ENABLE_COMPILE == "True" || $ENABLE_COMPILE == "true" ]]; then
    echo "Compile enabled!"
    train_command="$train_command --enable_compile"
fi

prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.prepare --yaml $1"

echo "NODE_RANK: {$NODE_RANK}"
echo "NNODES: {$NNODES}"
echo "NUM_PROCS: {$NUM_PROCS}"
echo "HEAD_NODE: {$HEAD_NODE}"
echo "HEAD_PORT: {$HEAD_PORT}"
echo "train_command: {$train_command}"
echo "prepare_train_command: {$prepare_train_command}"

echo "######################################"
echo "#       Running preparation stage    #"
echo "######################################"
    
$prepare_train_command

echo "######################################"
echo "#     Running DeepSpeen train        #"
echo "######################################"

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

if [ "${SLURM_PROCID}" -eq 0 ]; then
    rm "$tmp_deepspeed_config_path"
    rm "$HOSTFILE"
fi
echo "Deepspeed Job Completed."

