#!/bin/bash

# Check for minimum number of required arguments
if [ $# -lt 7 ]; then
    echo "Usage: $0 head_node_address --head|--worker MODEL_PATH LAUNCH_FOLDER BENCHMARK_FILE DATASET DATASET_PATH [additional_args...]"
    exit 1
fi

# Assign the first two arguments and shift them away
HEAD_NODE_ADDRESS="$1"
NODE_TYPE="$2"  # Should be --head or --worker
MODEL_PATH="$3"
LAUNCH_FOLDER="$4"
BENCHMARK_FILE="$5"
DATASET="$6"
DATASET_PATH="$7"
shift 7

# Additional arguments are passed directly to the Docker command
ADDITIONAL_ARGS=("$@")

# Validate node type
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi


module load $MODULES
source $ENVIRONMENT_VLLM/bin/activate
export PATH=$ENVIRONMENT_VLLM/bin:$PATH
which python

export RAY_USAGE_STATS_ENABLED=1
CUR_DIR=$(pwd)

cat <<EOT > config.sh
#!/bin/bash

export MODEL_PATH="${MODEL_PATH}"
export ENVIRONMENT_VLLM="${ENVIRONMENT_VLLM}"
export PORT="${PORT}"
export LAUNCH_FOLDER="${LAUNCH_FOLDER}"
export BENCHMARK_FILE="${BENCHMARK_FILE}"
export DATASET="${DATASET}"
export DATASET_PATH="${DATASET_PATH}"
export TP="${TENSOR_PARALLEL}"
export PP="${PIPELINE_PARALLEL}"
export ADDITIONAL_ARGS="${ADDITIONAL_ARGS[*]}"
export MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH}"
export HIP_VISIBLE_DEVICES="0,1,2,3"
EOT

echo "Configuration written to config.sh"

# Command setup for head or worker node
RAY_START_CMD="ray start"
if [ "${NODE_TYPE}" == "--head" ]; then
    # Head Node Ray start Cluster
    export RAY_ADRESS=HEAD_NODE_ADDRESS:6379
    RAY_START_CMD+=" --disable-usage-stats --block --head --port=6379 --node-ip-address=${HEAD_NODE_ADDRESS}"
    ${RAY_START_CMD} &
    
    sleep 90

    # Ray status
    echo "Ray Status"
    ray status
    
    sleep 2
    
    # Submit the vLLM Model in the Ray Cluster
    echo "VLLM serve loading... Head Node Address: ${HEAD_NODE_ADDRESS}"
    bash serve.sh
    # echo "Ray Job submitted..."

else
    # Worker Node Ray start command
    export RAY_ADRESS=HEAD_NODE_ADDRESS:6379
    RAY_START_CMD+=" --disable-usage-stats --block --address=${HEAD_NODE_ADDRESS}:6379"
    ${RAY_START_CMD}
    echo "Worker Node activated ${HEAD_NODE_ADDRESS}:6379"
fi


exit 0