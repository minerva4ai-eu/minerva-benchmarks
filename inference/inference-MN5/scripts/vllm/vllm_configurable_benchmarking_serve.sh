#!/bin/bash

#SBATCH --job-name=VLLM_DYNAMIC
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00



##################################################
###           Activate Environment             ###
##################################################
# Activate virtual environment using vllm v0.5.4
source activate-env-per-supercomputer.sh $ENVIRONMENT_VLLM
# ml miniforge
# source activate $ENVIRONMENT_VLLM
# export PATH=$ENVIRONMENT_VLLM/bin:$PATH
# which pip

##################################################


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
# source activate-env-variables-per-supercomputer.sh
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export NCCL_DEBUG=TRACE
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=0

# RAY vasource activate-env-per-supercomputer.sh $ENVIRONMENT_VLLM
# echo "Run_cluster: PYTHON PATH:"
# which pip
# echo ""riables
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_USAGE_STATS_ENABLED=1

# VLLM variables
export VLLM_ALLOW_ENGINE_USE_RAY=1

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

# Head the head_node_ip and port
export port=6379
export ip_head=$head_node_ip:$port


# VLLM variables
export VLLM_HOST_IP=$head_node_ip


##################################################

##################################################
###           Start the Ray Cluster            ###
##################################################

# Start the Head Node, wait until all ray cluster is initialized and start the vLLM serve
echo "Starting HEAD at $head_node"
srun -n 1 --nodes=1 --gres=gpu:$GPU_NODE -c $TOTAL_CPUS --cpu-bind=none  -w "$head_node" --export=ALL,VLLM_HOST_IP=$head_node_ip bash run_cluster.sh $head_node_ip --head $MODEL_PATH $LAUNCH_FOLDER $BENCHMARK_FILE $DATASET $DATASET_PATH $MACHINE $MACHINE_TYPE $ADDITIONAL_ARGS &
sleep 60
SRUN_PID=$!

# Start the workers nodes
echo "Starting workers Nodes"
worker_num=$((SLURM_NNODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    local_node_ip=$(srun -n 1 -N 1 -c 1 -w "$node_i" hostname --ip-address)
    export VLLM_HOST_IP=$local_node_ip
    ip_local=$local_node_ip:$port
    srun -n 1 --nodes=1 --gres=gpu:$GPU_NODE -c $TOTAL_CPUS --cpu-bind=none -w "$node_i" --export=ALL,VLLM_HOST_IP=$local_node_ip bash run_cluster.sh $head_node_ip --worker $MODEL_PATH $LAUNCH_FOLDER $BENCHMARK_FILE $DATASET $DATASET_PATH $MACHINE $MACHINE_TYPE $ADDITIONAL_ARGS &
    sleep 3
done

echo "Ray Cluster started correctly"

##################################################

# Wait for the Ray Cluster to stabilize
sleep 1000



echo "Ray Cluster started with PID $SRUN_PID"


while kill -0 "$SRUN_PID" 2>/dev/null; do
    echo "Don't kill process $SRUN_PID, it is still running"
    sleep 10
done

echo "Process $SRUN_PID has finished, exiting script"

exit
