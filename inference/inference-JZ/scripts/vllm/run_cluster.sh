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
. /lustre/fshomisc/sup/pub/miniforge/24.11.3/etc/profile.d/conda.sh;
conda activate $ENVIRONMENT_VLLM
export PATH=$ENVIRONMENT_VLLM/bin:$PATH
which python

export RAY_USAGE_STATS_ENABLED=1
CUR_DIR=$(pwd)

cat <<EOT > config.json
{
    "working_dir": ".",
    "env_vars": {
        "MODEL_PATH": "${MODEL_PATH}",
        "ENVIRONMENT_VLLM": "${ENVIRONMENT_VLLM}",
        "PORT": "${PORT}",
        "LAUNCH_FOLDER": "${LAUNCH_FOLDER}",
        "BENCHMARK_FILE": "${BENCHMARK_FILE}",
        "DATASET": "${DATASET}",
        "DATASET_PATH": "${DATASET_PATH}",
        "TP": "${TENSOR_PARALLEL}",
        "ADDITIONAL_ARGS": "${ADDITIONAL_ARGS[*]}",
        "MODULES": "${MODULES}"
    }
}
EOT

echo "Configuration written to config.json"

# Command setup for head or worker node
RAY_START_CMD="ray start"
if [ "${NODE_TYPE}" == "--head" ]; then
    # Head Node Ray start Cluster
    #RAY_START_CMD+=" --head --port=6379 --num-cpus 80 --num-gpus 4 --node-ip-address=${HEAD_NODE_ADDRESS} --include-dashboard=True"
    #RAY_START_CMD+=" --head --port=6379 --num-cpus 80 --num-gpus 4 --node-ip-address=${HEAD_NODE_ADDRESS} --include-dashboard=True --dashboard-port=8265"
    #${RAY_START_CMD}
    ray stop

    ray start --head \
        --num-cpus=80 --num-gpus=4 \
        --node-ip-address=${HEAD_NODE_ADDRESS} \
        --include-dashboard=True \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

    
    sleep 120

    # Ray status
    echo "Ray Status"
    ray status
    
    sleep 2
    
    # Submit the vLLM Model in the Ray Cluster
    echo "VLLM serve loading... Head Node Address: ${HEAD_NODE_ADDRESS}"
    ray job submit --address=$HEAD_NODE_ADDRESS:6379 --runtime-env config.json -- bash $CUR_DIR/serve.sh $MODEL_PATH
    curl http://${HEAD_NODE_ADDRESS}:8265/api/jobs/

    
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

        # Sleep for some interval before checking again
        sleep 10
    done

else
    # Worker Node Ray start command
    RAY_START_CMD+=" --block --address=${HEAD_NODE_ADDRESS}:6379 --node-ip-address=${VLLM_HOST_IP} --num-cpus 80 --num-gpus 4"
    echo ${RAY_START_CMD}
    ${RAY_START_CMD}
    echo "Worker Node ${VLLM_HOST_IP} activated ${HEAD_NODE_ADDRESS}"
fi


exit 0
