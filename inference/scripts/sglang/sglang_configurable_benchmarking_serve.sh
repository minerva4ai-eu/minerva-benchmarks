#!/bin/bash

scontrol show step $SLURM_JOB_ID.$SLURM_STEP_ID
nvidia-smi

##################################################
###        Environment Variables Setup         ###
##################################################

# Get Arguments
LAUNCH_FOLDER=$1
BENCHMARK_FILE=$2
export DATASET=$3
export DATASET_PATH=$4
export MODEL_PATH=$5
export MAX_MODEL_LENGTH=$6

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}"
echo "BENCHMARK_FILE: {$BENCHMARK_FILE}"
echo "DATASET: {$DATASET}"
echo "DATASET_PATH: {$DATASET_PATH}"
echo MODEL_PATH: $MODEL_PATH
echo MAX_MODEL_LENGTH: $MAX_MODEL_LENGTH

# NCCL variables
source $SCRIPTS_DIR/activate-env-variables-per-supercomputer.sh

##################################################

##################################################
###           Get IPs for all NODES            ###
##################################################

# Get current hostnames and Head Node
echo "[INFO] SLURM_STEP_NODELIST: $SLURM_STEP_NODELIST"
nodes=$(scontrol getaddrs "$SLURM_STEP_NODELIST" | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}')
echo "[INFO] Nodes: $nodes"
nodes_array=($nodes)
echo "[INFO] Nodes Array: ${nodes_array[@]}"
head_node_ip=${nodes_array[0]}
echo "[INFO] Head Node: $head_node_ip"


##################################################

##################################################
###       Start SGLang Inference Server        ###
##################################################

echo "[INFO] Running SGLang inference"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP Size: $TENSOR_PARALLEL"
echo "[INFO] NODES: $NODES"

# Set NCCL initialization address using the hostname of the head node
export NCCL_INIT_ADDR="${head_node_ip}:8080"
echo "[INFO] NCCL_INIT_ADDR: $NCCL_INIT_ADDR"

# Launch the model server on each node using SLURM
bash $SCRIPTS_DIR/sglang/serve.sh &

SRUN_PID=$!

sleep 100

echo "SGLang Server started with PID $SRUN_PID"

# Run until `srun` ends.
while kill -0 "$SRUN_PID" 2>/dev/null; do
    echo "Don't kill process $SRUN_PID, it is still running"
    sleep 10
done

echo "Process $SRUN_PID has finished, exiting script"

exit 0
