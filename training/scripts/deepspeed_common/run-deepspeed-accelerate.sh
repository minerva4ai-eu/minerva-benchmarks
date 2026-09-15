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


# Define a barrier file on a SHARED filestem accessible by all nodes
BARRIER_DIR="$SRUN_LOGS"
BARRIER_FILE="${BARRIER_DIR}/.deepspeed-accelerate-rak0-$SLURM_JOB_ID.$SLURM_STEP_ID.lock"

# Use SLURM_STEP_NODELIST (step allocation) instead of SLURM_NODELIST (job allocation)
# Fallback GPUS_PER_NODE to SLURM_GPUS_ON_NODE
GPUS_PER_NODE=${GPUS_PER_NODE:-$SLURM_GPUS_ON_NODE}

deepspeed_config_path="scripts/deepspeed_common/configs/${ZERO_STAGE}.json"
tmp_deepspeed_config_path="${BARRIER_DIR}/.deepspeed-config-$SLURM_JOB_ID.$SLURM_STEP_ID.json"

if [ "${SLURM_PROCID}" -eq 0 ]; then
    cp "$deepspeed_config_path" "$tmp_deepspeed_config_path"

    # Update hpZ partition size for parallelism of stage2 and stage3 configurations only, as stage1 does not use hpZ
    HPZ_PARTITION_SIZE=$((NNODES * SLURM_GPUS_ON_NODE))
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

train_command="${runtime_prefix:+$runtime_prefix} accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file $tmp_deepspeed_config_path \
    --machine_rank $SLURM_NODEID \
    --rdzv_backend c10d \
    --main_process_ip $HEAD_NODE \
    --main_process_port $HEAD_PORT \
    --num_processes $NUM_PROCS \
    --num_machines $SLURM_STEP_NUM_NODES \
    $TRAIN_SCRIPT --yaml $1 --deepspeed_config_file  $tmp_deepspeed_config_path"
      
prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.prepare --yaml $1"

echo "ENABLE_COMPILE: $ENABLE_COMPILE"
if [[ $ENABLE_COMPILE == "True" || $ENABLE_COMPILE == "true" ]]; then
    echo "Compile enabled!"
    train_command="$train_command --enable_compile"
fi

echo "######################################"
echo "#       Running preparation stage    #"
echo "######################################"
    
$prepare_train_command

echo "#######################################"
echo "# Running  Accelerate-Deepspeed train #"
echo "#######################################"


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
echo "Accelerate-Deepspeed Job Completed."

