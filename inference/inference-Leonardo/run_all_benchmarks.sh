#!/bin/bash
set -a  # Automatically export all variables
source .env
set +a  # Stop automatically exporting

# Load utility functions
source scripts/utils.sh


#######################################################
# ENVIRONMENT VARIABLES
#######################################################
# DEFAULT
#######################################################
GPUS_PER_NODE=4
FRAMEWORKS=("vllm" "deepspeed") # deepspeed") # vllm")    # Add other frameworks if needed
DATASETS=("sharegpt" "sonnet")  # Add more datasets if needed
MODELS=("Llama-3.1-8B-Instruct" "Llama-3.1-405B" "gemma-3-12b-it" "Mistral-7B-Instruct-v0.3") # "gemma-3-12b-it") # Mistral-7B-Instruct-v0.3") # Add your models here
NUMBER_OF_NODES=(1 2 4) #8 16)
REPEATS=3             # Number of runs per configuration
#######################################################



JOB_IDS=()
CONFIG_INDEX=0
CURRENT_DIR=$(pwd)
TOTAL_CONFIGS=$(( ${#DATASETS[@]} * ${#FRAMEWORKS[@]} * ${#NUMBER_OF_NODES[@]} * ${#MODELS[@]} * REPEATS ))

for framework in "${FRAMEWORKS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for NODES in "${NUMBER_OF_NODES[@]}"; do
        # GENERAL PART (Common for all Frameworks).
        TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
        TENSOR_PARALLEL=$TOTAL_GPUS
        PIPELINE_PARALLEL=1

        BASE_FOLDER="results/${framework}/${dataset}/${model}"
        RUN_FOLDER="Nodes_${NODES}-GPUs_${GPUS_PER_NODE}-TP_${TENSOR_PARALLEL}-PP_${PIPELINE_PARALLEL}"
        FULL_FOLDER="${BASE_FOLDER}/${RUN_FOLDER}"

        # Define a unique MODEL_PATH per configuration
        MODEL_TYPE=$(get_model_type "$model" "configs/model_type_map.json")
        MODEL_DIRECTORY=$(get_model_directory "$MODEL_TYPE" "configs/model_type_directories_map.json")
        MODEL_PATH="${MODEL_DIRECTORY}/${model}"

        if [ -z "$MODEL_DIRECTORY" ]; then
          echo "Unknown model type '$MODEL_TYPE' or missing directory mapping. Exiting."
          exit 1
        fi

        DATASET_PATH=$(get_dataset_path "$dataset" "configs/config_datasets_paths_map.json")
        
        # vLLM
        if [[ "$framework" == "vllm" ]]; then
          # vLLM
          echo "FrameWork vLLM"

          # If Model is Llama-3.1-405B.
          if [[ "$model" == "Llama-3.1-405B" ]]; then
            # Skip if model is Llama-3.1-405B and NODES < 4.
            if [[ "$NODES" -lt 4 ]]; then
              echo "Skipping $model with $NODES nodes (requires at least 4 nodes)"
              continue
            fi
            # Set extra args for Llama-3.1-405B
            ADDITIONAL_ARGS="--max-model-len 32000 --cpu-offload-gb 0.5"
          fi
          # If Model is Llama-3.1-8B-Instruct.
          if [[ "$model" == "Llama-3.1-8B-Instruct" ]]; then
            # Set extra args for Llama-3.1-8B-Instruct
            ADDITIONAL_ARGS="--max-model-len 32000 --cpu-offload-gb 0.5"
          fi

          for (( run_id=1; run_id<=REPEATS; run_id++ )); do
            LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
            echo "Setting up $LAUNCH_FOLDER"
            mkdir -p "$LAUNCH_FOLDER"
            
            cp scripts/vllm/run_cluster.sh "$LAUNCH_FOLDER"
            cp scripts/vllm/vllm_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER"
            cp scripts/vllm/serve.sh "$LAUNCH_FOLDER"
            
            cd "$LAUNCH_FOLDER" || exit 1

            export NODES GPUS_PER_NODE TENSOR_PARALLEL PIPELINE_PARALLEL
            export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
            export MODEL_PATH  # Make available to launched script
            export ADDITIONAL_ARGS
            export MODULES

            if [ "${#JOB_IDS[@]}" -gt 0 ]; then
              DEPENDENCY="--dependency=afterany:${JOB_IDS[-1]}"
            else
              DEPENDENCY=""
            fi

            JOB_ID=$(sbatch --parsable \
                --chdir=$(pwd) \
                --nodes=$NODES \
                --gres=gpu:$GPUS_PER_NODE \
                $DEPENDENCY \
                --output=run-%j.out \
                --error=run-%j.err \
                -A $ACCOUNT \
                -q $QOS \
                vllm_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH")

            echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
            JOB_IDS+=("$JOB_ID")
            ((CONFIG_INDEX++))

            cd - > /dev/null
            sleep 5
          done
        fi

        # DeepSpeed-MII 
        if [[ "$framework" == "deepspeed" ]]; then
          # DeepSpeed-MII
          echo "DeepSpeed-MII"
          
          # If Model is Llama-3.1-405B, avoid it.
          if [[ "$model" == "Llama-3.1-405B" ]]; then
            continue
          fi
          # If Model is gemma-3-12b-it, avoid it.
          if [[ "$model" == "gemma-3-12b-it" ]]; then
            continue
          fi

          # If iteration has more than 1 Node, avoid it.
          if [[ "$NODES" != 1 ]]; then
            echo "Skipping deepspeed-mii $model with $NODES nodes (requires maximum 1 node)"
            continue
          fi

          for (( run_id=1; run_id<=REPEATS; run_id++ )); do
            LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
            echo "Setting up $LAUNCH_FOLDER"
            mkdir -p "$LAUNCH_FOLDER"
            
            cp scripts/deepspeed/serve_deepspeed_mii.py "$LAUNCH_FOLDER"
            cp scripts/deepspeed/deepspeed-mii_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER"
            
            cd "$LAUNCH_FOLDER" || exit 1

            export NODES GPUS_PER_NODE TENSOR_PARALLEL PIPELINE_PARALLEL
            export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
            export MODEL_PATH  # Make available to launched script
            export ADDITIONAL_ARGS
            export MODULES

            if [ "${#JOB_IDS[@]}" -gt 0 ]; then
              DEPENDENCY="--dependency=afterany:${JOB_IDS[-1]}"
            else
              DEPENDENCY=""
            fi

            JOB_ID=$(sbatch --parsable \
                --chdir=$(pwd) \
                --nodes=$NODES \
                --gres=gpu:$GPUS_PER_NODE \
                $DEPENDENCY \
                --output=run-%j.out \
                --error=run-%j.err \
                -A $ACCOUNT \
                -q $QOS \
                deepspeed-mii_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH")

            echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
            JOB_IDS+=("$JOB_ID")

            cd - > /dev/null
            sleep 5
          done
        fi
        
      done
    done
  done
done