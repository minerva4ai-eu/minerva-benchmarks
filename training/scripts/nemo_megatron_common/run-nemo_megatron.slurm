#!/bin/bash

## ENV ACTIVATION
##################################################
###           Load HPC modules                 ###
##################################################
module purge
if [ ! -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi
export NUMEXPR_MAX_THREADS=256

source scripts/shared/runtime_environment.sh
training_activate_runtime_environment

echo "EXECUTION_MODE: $EXECUTION_MODE"
runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix: $runtime_prefix"
echo yaml_path=$1

runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix $runtime_prefix"
gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.gpu_plots"

## CODE EXECUTION
set -x

prepare_train_command="${runtime_prefix:+$runtime_prefix} \
    python $TRAIN_MODULE --yaml $1 --prepare"


train_command="${runtime_prefix:+$runtime_prefix} \
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=$SLURM_GPUS_ON_NODE \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:$HEAD_PORT \
        $TRAIN_MODULE --yaml $1 --enable-flop-counter"



    export HF_HUB_OFFLINE="1" 
    export SINGULARITY_HF_HUB_OFFLINE="1" 

if [[ "${PREPARE_ONLY:-}" == "1" ]]; then
    if [[ ! -e "$MODEL_PATH/nemo_ckpt" || ! -e "$DATASET_PATH/training.jsonl" || ! -e "$DATASET_PATH/validation.jsonl" ]]; then
        echo "######################################"
        echo "#       Running preparation stage    #"
        echo "######################################"

        CUDA_VISIBLE_DEVICES="0" SINGULARITY_CUDA_VISIBLE_DEVICES="0" $prepare_train_command
        exit_code=$?
    fi
    echo "[prepare] PREPARE_ONLY=1, skipping training"
    exit "${exit_code:-0}"
fi


echo "######################################"
echo "#     Running Megatron-Nemo train    #"
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

