#!/bin/bash

# Check for minimum number of required arguments
if [ $# -lt 9 ]; then
    echo "Usage: $0 head_node_address --head|--worker MODEL_PATH LAUNCH_FOLDER BENCHMARK_FILE DATASET DATASET_PATH MACHINE MACHINE_TYPE [additional_args...]"
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
MACHINE="$8"
MACHINE_TYPE="$9"
shift 9

# Additional arguments are passed directly to the command
ADDITIONAL_ARGS=("$@")

# Validate node type
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Activate environment
echo "ENVIRONMENT_VLLM: $ENVIRONMENT_VLLM"
echo ""
source activate-env-per-supercomputer.sh $ENVIRONMENT_VLLM
echo "run_cluster: PYTHON PATH:"
which python
echo ""


export RAY_USAGE_STATS_ENABLED=1

CUR_DIR=$(pwd)
echo "run_cluster - MACHINE: $MACHINE"
echo "run_cluster - MACHINE_TYPE: $MACHINE_TYPE"

cat <<EOT > config.json
{
    "env_vars": {
        "MODEL_PATH": "${MODEL_PATH}",
        "ENVIRONMENT_VLLM": "${ENVIRONMENT_VLLM}",
        "PORT": "${PORT}",
        "LAUNCH_FOLDER": "${LAUNCH_FOLDER}",
        "BENCHMARK_FILE": "${BENCHMARK_FILE}",
        "DATASET": "${DATASET}",
        "DATASET_PATH": "${DATASET_PATH}",
        "TP": "${TENSOR_PARALLEL}",
        "PP": "${PIPELINE_PARALLEL}",
        "MAX_MODEL_LENGTH": "${MAX_MODEL_LENGTH}",
        "ADDITIONAL_ARGS": "${ADDITIONAL_ARGS[*]}",
        "MODULES": "${MODULES}",
        "MACHINE": "${MACHINE}",
        "MACHINE_TYPE": "${MACHINE_TYPE}"
    }
}
EOT
echo "Configuration written to config.json"

# Command setup for head or worker node
RAY_START_CMD="ray start"
if [ "${NODE_TYPE}" == "--head" ]; then
    # Head Node Ray start Cluster
    RAY_START_CMD+=" --head --port=6379 --node-ip-address=${HEAD_NODE_ADDRESS}"
    ${RAY_START_CMD}

    sleep 90

    # Ray status
    echo "Ray Status"
    ray status

    sleep 2

    # Submit the vLLM Model in the Ray Cluster
    echo "VLLM serve loading... Head Node Address: ${HEAD_NODE_ADDRESS}"
    ray job submit --runtime-env config.json -- bash $CUR_DIR/serve.sh $MODEL_PATH
    # ray job submit -- bash $CUR_DIR/serve.sh $MODEL_PATH
    
    echo "Ray Job submitted..."

    # Iterate until all jobs are finished
    while true; do
        echo "Checking for running jobs..."

        # Fetch the job list and filter for any 'RUNNING' job
        running_jobs=$(ray job list | grep -c "RUNNING")

        if [ "$running_jobs" -gt 0 ]; then
            echo "There are $running_jobs job(s) running."
        else
            echo "No jobs are currently running."
            break
        fi

        sleep 10
    done

else
    # Worker Node Ray start command
    RAY_START_CMD+=" --block --address=${HEAD_NODE_ADDRESS}:6379"
    ${RAY_START_CMD}
    echo "Worker Node activated ${HEAD_NODE_ADDRESS}:6379"
fi

exit 0