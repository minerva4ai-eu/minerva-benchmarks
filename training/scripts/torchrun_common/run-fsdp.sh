#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################
module purge
if [ ! -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi

echo yaml_path=$1

source scripts/shared/runtime_environment.sh
training_activate_runtime_environment

echo "EXECUTION_MODE: $EXECUTION_MODE"
runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix: $runtime_prefix"


# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}"
echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"


# Export environment variables
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
###################################################

##################################################
###             Torchrun Setup                 ###
##################################################
gpu_plots_monitor_command="python -m gpu_plots"


# Torchrun args
export JOB_ID=${SLURM_JOB_ID}
export NNODES=${SLURM_NNODES}
export NPROC_PER_NODE=$GPUS_PER_NODE
export NUM_PROCS=$((NNODES * NPROC_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID

gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.gpu_plots"


train_command="${runtime_prefix:+$runtime_prefix} torchrun \
    --nnodes $SLURM_STEP_NUM_NODES \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --rdzv_id $SLURM_JOB_ID.$SLURM_STEP_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint ${HEAD_NODE}:${MASTER_PORT} \
    $TRAIN_MODULE --yaml $1 --max_comm_comp_overlap"


prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.prepare --yaml $1 "

echo "ENABLE_COMPILE: $ENABLE_COMPILE"
if [[ $ENABLE_COMPILE == "True" || $ENABLE_COMPILE == "true" ]]; then
    train_command="$train_command --enable_compile"
fi

echo "######################################"
echo "#       Running preparation stage    #"
echo "######################################"
    
$prepare_train_command

echo "######################################"
echo "#     Running  Torchrun-FSDP train    #"
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


# Terminate the monitor
kill -SIGTERM "$monitor_pid"
# Wait for the monitor to clean up and exit
wait "$monitor_pid"


echo "FSDP Job Completed."
exit "$exit_code"
