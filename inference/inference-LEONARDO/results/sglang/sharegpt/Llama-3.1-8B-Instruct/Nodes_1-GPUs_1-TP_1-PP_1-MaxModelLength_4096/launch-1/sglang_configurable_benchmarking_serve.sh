#!/bin/bash

#SBATCH --job-name=SGLANG_DYNAMIC
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00


##################################################
###        Environment Variables Setup         ###
##################################################

# Get Arguments
LAUNCH_FOLDER=$1
BENCHMARK_FILE=$2
DATASET=$3
DATASET_PATH=$4

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}"
echo "BENCHMARK_FILE: {$BENCHMARK_FILE}"
echo "DATASET: {$DATASET}"
echo "DATASET_PATH: {$DATASET_PATH}"


# Export environment variables
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# NCCL variables
source activate-env-variables-per-supercomputer.sh

##################################################

##################################################
###           Get IPs for all NODES            ###
##################################################

# Get current hostnames and Head Node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Get the IP addresses for each node
for node in "${nodes_array[@]}"; do
    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address)
    echo "Node: $node - IP: $node_ip"
done


##################################################

##################################################
###       Start SGLang Inference Server        ###
##################################################

echo "[INFO] Running SGLang inference"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP Size: $TENSOR_PARALLEL"
echo "[INFO] NODES: $NODES"

# Set NCCL initialization address using the hostname of the head node
NCCL_INIT_ADDR="${head_node_ip}:8080"
echo "[INFO] NCCL_INIT_ADDR: $NCCL_INIT_ADDR"

# Set Current Directory
CUR_DIR=$(pwd)
echo "[INFO] CUR_DIR=$CUR_DIR"

# Launch the model server on each node using SLURM
srun -n $NODES --nodes=$NODES --gres=gpu:$GPU_NODE -c $TOTAL_CPUS --cpu-bind=none \
    --export=ALL,MODEL_PATH=$MODEL_PATH,TENSOR_PARALLEL=$TENSOR_PARALLEL,NCCL_INIT_ADDR=$NCCL_INIT_ADDR,NODES=$NODES,PORT=$PORT,PIPELINE_PARALLEL=$PIPELINE_PARALLEL,MAX_MODEL_LENGTH=$MAX_MODEL_LENGTH,ADDITIONAL_ARGS=$ADDITIONAL_ARGS,CUR_DIR=$CUR_DIR,LAUNCH_FOLDER=$LAUNCH_FOLDER,BENCHMARK_FILE=$BENCHMARK_FILE,DATASET=$DATASET,DATASET_PATH=$DATASET_PATH,ENVIRONMENT_VLLM=$ENVIRONMENT_VLLM,ENVIRONMENT_SGLANG=$ENVIRONMENT_SGLANG,SINGULARITY_MODULE=$SINGULARITY_MODULE,SGLANG_IMAGE="$SGLANG_IMAGE" \
    bash $CUR_DIR/serve.sh &

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