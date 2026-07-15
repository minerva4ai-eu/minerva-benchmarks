#!/bin/bash

# ANSI Color Codes
GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
BLUE='\033[94m'
MAGENTA='\033[95m'
CYAN='\033[96m'
GRAY='\033[90m'
RESET='\033[0m'

# Unicode Icons
SUCCESS="✓"
SUCCESS_HEAVY="✔"
FAILURE="✗"
FAILURE_HEAVY="✘"
WARNING="⚠"
INFO="ℹ"
PROGRESS="⟳"
SKIPPED="↷"

POINT_DIAMOND="◆"
POINT_SQUARE="▪"

get_config() {
    
    CONFIG_FILE=$CONFIGS_PATH/benchmark.json
    echo -e "\nUsing configuration from: $CONFIG_FILE"
    # Machine config check
    if [ ! -f $CONFIG_FILE ]; then
        echo -e "${RED}Configuration file not found: $CONFIG_FILE${RESET}" >&2
        exit 1
    fi
    
    # Read config data from file
    CONFIG_DATA=$(jq -r '.' $CONFIG_FILE)

    # Store arrays
    mapfile -t model_names < <(echo ${CONFIG_DATA[@]} | jq -r '.models | keys[]')
    mapfile -t dataset_names < <(echo ${CONFIG_DATA[@]} | jq -r '.training.datasets[]')
    mapfile -t node_configs < <(echo ${CONFIG_DATA[@]} | jq -r '.training.node_configs[]')
    mapfile -t batch_sizes < <(echo ${CONFIG_DATA[@]} | jq -r '.training.batch_sizes[]')
    mapfile -t grad_accums < <(echo ${CONFIG_DATA[@]} | jq -r '.training.grad_accums[]')
    mapfile -t enable_compile < <(echo ${CONFIG_DATA[@]} | jq -r '.training.enable_compile[]')
    mapfile -t precisions < <(echo ${CONFIG_DATA[@]} | jq -r '.training.precisions[]')
    mapfile -t lrs < <(echo ${CONFIG_DATA[@]} | jq -r '.training.lrs[]')
    # enable_flash_attention=$(echo ${CONFIG_DATA[@]} | jq -r '.training.enable_flash_attention[]')

    # Store scalars
    export STEPS=$(echo ${CONFIG_DATA[@]} | jq '.training.steps')
    export EPOCHS=$(echo ${CONFIG_DATA[@]} | jq '.training.epochs')
    trials=$(echo ${CONFIG_DATA[@]} | jq '.training.trials')


    # Mini Mode for testing
    if [[ $MINI_MODE == true ]]; then
        echo "Running in mini mode"
        model_names=(${model_names[0]})
        dataset_names=(${dataset_names[0]})
        batch_sizes=(${batch_sizes[-1]})
        grad_accums=(${grad_accums[-1]})
        precisions=(${precisions[0]})
        enable_bf16=(${enable_bf16[0]})
        STEPS=2
        lrs=(${lrs[0]})
        trials=1
    fi

}

check_config() {
    # For Dry Run
    if [[ -z $FRAMEWORK ]]; then
        FRAMEWORK=$2
    fi
    if [[ -z $PARALLELISM ]]; then
        PARALLELISM=$2
    fi
    if [[ -z $NUMBER_OF_NODES ]]; then
        NUMBER_OF_NODES=$2
    fi
    
    echo Checking configs

    # Rule Check: Parallelism-Node
    if [[ $PARALLELISM == "none" && $NUMBER_OF_NODES -gt 1 ]]; then
        echo "Skipping parallelism $PARALLELISM on multiple-node (requires only 1 node)"
        return 0
    fi

    # Rule Check: Framework-Parallelism
    if [[ $PARALLELISM == "none" && $FRAMEWORK != "torchrun" ]]; then
        echo "Skipping parallelism $PARALLELISM with distributed framework $FRAMEWORK (setup only in torchrun)"
        return 0
    fi
    if [[ $PARALLELISM == "zero"* && $FRAMEWORK != "deepspeed"* ]]; then
        echo "Skipping parallelism $PARALLELISM with Non-DeepSpeed framework"
        return 0
    fi
    if [[ $FRAMEWORK == "deepspeed"* && $PARALLELISM != "zero"*  ]]; then
        echo "Skipping Non-Zero Parallelism with framework $FRAMEWORK"
        return 0
    fi

    # Rule Check: Slurm-Node
    if [[ $SLURM_JOB_NUM_NODES -gt 0 && $SLURM_JOB_NUM_NODES < $NUMBER_OF_NODES ]]; then
        echo "Skipping num_node=$NUMBER_OF_NODES on job with only $SLURM_JOB_NUM_NODES"
        return 0
    fi

    # Rule Check: gemma3-1b,accelerate,fsdp,N1,"Attempting to unscale FP16 gradients."
    # Rule Check: FlashAttention only support fp16 and bf16 data type
    # Rule Check: CPU-Precision (only f32 supported; b16,fp16 not supported)
    # 43303049-202/216/224: "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"

    return 1

}
export -f check_config