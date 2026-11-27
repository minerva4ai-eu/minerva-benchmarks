#!/bin/bash

#######################################################
# ENVIRONMENT VARIABLES TO CHANGE
#######################################################
# RUN ALL BENCHMARKS
#######################################################
FRAMEWORKS=("vllm" "sglang" "deepspeed") # Add other frameworks if needed
DATASETS=("sharegpt" "sonnet")  # Add more datasets if needed
MODELS=("Llama-3.1-8B-Instruct" "Llama-3.3-70B-Instruct" "Llama-3.1-405B-Instruct" "gemma-3-12b-it" "Mistral-7B-Instruct-v0.3" "ALIA-40b") # Add your models here
NUMBER_OF_NODES=(1 2 4)
MAX_MODEL_LENGTHS=(4096 16384 32768) # (2048 4096 8192 16384 32768)
REPEATS=3               # Number of runs per configuration
export BASEBASE_DIR="results-$SLURM_JOB_ID"
export SCRIPTS_DIR="scripts"
#######################################################
# Set environment variables
#######################################################
set -a  # Automatically export all variables
source .env-$MACHINE
set +a  # Stop automatically exporting

# Load utility functions
source scripts/utils.sh

module load $MODULES
#######################################################

runner_fn () {

  echo -e "\n\nStarting runner: ..."
  # which python
  # echo

  framework=$1
  dataset=$2
  model=$3
  NODES=$4
  MAX_MODEL_LENGTH=$5
  TRIAL=$6

  GPU_NODE=$SLURM_GPUS_ON_NODE
  echo FRAMEWORK=$framework DATASET=$dataset MODEL=$model NODES=$NODES MAX_MODEL_LENGTH=$MAX_MODEL_LENGTH TRIAL=$TRIAL GPU_NODE=$GPU_NODE

  # GENERAL PART (Common for all Frameworks).
  TOTAL_GPUS=$((NODES*GPU_NODE))
  TOTAL_CPUS=$((GPUS_PER_NODE*CPUS_PER_GPU))
  #TODO: condition on framework? pass as args?
  export TENSOR_PARALLEL=$GPU_NODE
  export PIPELINE_PARALLEL=$NODES

  BASE_FOLDER="$BASEBASE_DIR/${framework}/${dataset}/${model}"
  RUN_FOLDER="Nodes_${NODES}-GPUs_${TOTAL_GPUS}-TP_${TENSOR_PARALLEL}-PP_${PIPELINE_PARALLEL}-MaxModelLength_${MAX_MODEL_LENGTH}"
  FULL_FOLDER="${BASE_FOLDER}/${RUN_FOLDER}"
  export LAUNCH_FOLDER="${FULL_FOLDER}/launch-${TRIAL}"

  # Define a unique MODEL_PATH per configuration
  MODEL_TYPE=$(get_model_type "$model" "configs-$MACHINE/model_type_map.json")
  MODEL_DIRECTORY=$(get_model_directory "$MODEL_TYPE" "configs-$MACHINE/model_type_directories_map.json")
  MODEL_PATH="${MODEL_DIRECTORY}/${model}"

  if [ -z "$MODEL_DIRECTORY" ]; then
    echo "Unknown model type '$MODEL_TYPE' or missing directory mapping. Exiting."
    exit 1
  fi

  DATASET_PATH=$(get_dataset_path "$dataset" "configs-$MACHINE/config_datasets_paths_map.json")

  ADDITIONAL_ARGS=""
  RUN_SCRIPT="run_mp_vllm.sh"  # Default script, can be overridden per framework

  # Framework-specific configurations
  if [[ "$framework" == "vllm" ]]; then
    # vLLM
    echo "FrameWork vLLM"
    RUN_SCRIPT="run_mp_vllm.sh"

    # If Model is Llama-3.1-405B.
    if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" ]]; then
      # Skip if model is Llama-3.1-405B and NODES < 4.
      if [[ "$NODES" -lt 4 ]]; then
        echo "Skipping $model with $NODES nodes (requires at least 4 nodes)"
        exit 1
      fi
    fi
    ADDITIONAL_ARGS="--disable-log-requests"
  elif [[ "$framework" == "deepspeed" ]]; then
    # DeepSpeed-MII

    if [[ $MACHINE_TYPE != "rocm" ]]; then
      echo "DeepSpeed-MII"
      RUN_SCRIPT="deepspeed-mii_configurable_benchmarking_serve.sh"

      # If Model is Llama-3.1-405B, avoid it.
      if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" || "$model" == "Llama-3.3-70B-Instruct" ]]; then
        exit 1
      fi
      # If Model is gemma-3-12b-it, avoid it.
      if [[ "$model" == "gemma-3-12b-it" ]]; then
        exit 1
      fi
      # If Model is Mistral-7B-Instruct-v0.3, avoid it.
      if [[ "$model" == "Mistral-7B-Instruct-v0.3" ]]; then
        exit 1
      fi
      # If iteration has more than 1 Node, avoid it.
      if [[ "$NODES" != 1 ]]; then
        echo "Skipping deepspeed-mii $model with $NODES nodes (requires maximum 1 node)"
        exit 1
      fi
    else
      echo "Deepspeed MII not supported on ROCM architectures"
      exit 1
    fi
  elif [[ "$framework" == "sglang" ]]; then
    # SGLang
    echo "FrameWork SGLang"
    RUN_SCRIPT="sglang_configurable_benchmarking_serve.sh"
    
    # If Model is Llama-3.1-405B.
    if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" ]]; then
      # Skip if model is Llama-3.1-405B and NODES < 4.
      if [[ "$NODES" -lt 4 ]]; then
        echo "Skipping $model with $NODES nodes (requires at least 4 nodes)"
        exit 1
      fi
      # ADDITIONAL_ARGS="--mem-fraction-static 0.80 --chunked-prefill-size 4096"
    fi
    # If Model is 'gemma-3-12b-it'
    if [[ "$model" == "gemma-3-12b-it" ]]; then
      # Skip if model is 'gemma-3-12b-it' and NODES > 1.
      if [[ "$NODES" -gt 1 ]]; then
        echo "Skipping $model with $NODES nodes (tp and pp cannot be set in SGLang Framework) for 'gemma-3-12b-it' model"
        exit 1
      fi
    fi
    
  fi

  mkdir -p "$LAUNCH_FOLDER"

  # TODO: fix logging
  # TODO: fix backgrounded srun and wait?
  srun --nodes=$NODES --gres=gpu:$GPU_NODE -o $LAUNCH_FOLDER/step-%s.out -e $LAUNCH_FOLDER/step-%s.err \
    $SCRIPTS_DIR/$framework/$RUN_SCRIPT "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$dataset" "$DATASET_PATH" $MODEL_PATH $MAX_MODEL_LENGTH $model &
  SRUN_PID=$!

  # TODO: fix setup for single GPU
  if [[ "$NODES" -eq 1 && "$model" == "gemma-3-12b-it" ]]; then
    echo "Running single GPU..."
    
    GPU_NODE=1
    TOTAL_GPUS=$((NODES*GPU_NODE))
    TOTAL_CPUS=$((GPUS_PER_NODE*CPUS_PER_GPU))
    #TODO: condition on framework? pass as args?
    export TENSOR_PARALLEL=$GPU_NODE

    RUN_FOLDER="Nodes_${NODES}-GPUs_${TOTAL_GPUS}-TP_${TENSOR_PARALLEL}-PP_${PIPELINE_PARALLEL}-MaxModelLength_${MAX_MODEL_LENGTH}"
    FULL_FOLDER="${BASE_FOLDER}/${RUN_FOLDER}"
    export LAUNCH_FOLDER="${FULL_FOLDER}/launch-${TRIAL}"
    srun --nodes=$NODES --gres=gpu:$GPU_NODE -o $LAUNCH_FOLDER/step-%s.out -e $LAUNCH_FOLDER/step-%s.err \
      echo "Single GPU run not implmented yet. Exiting."
  fi

  # Block until this runner's benchmark step has finished.
  wait $SRUN_PID
}
export -f runner_fn

parallel -j 0 runner_fn {} ::: ${FRAMEWORKS[@]} ::: ${DATASETS[@]} ::: ${MODELS[@]} ::: ${NUMBER_OF_NODES[@]} ::: ${MAX_MODEL_LENGTHS[@]} ::: $(seq $REPEATS)