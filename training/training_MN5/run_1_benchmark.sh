#!/bin/bash

#######################################################
# COLORS
#######################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color
#######################################################
# ENVIRONMENT VARIABLES TO CHANGE
#######################################################
# SPECIFIC CASE FOR TESTING
#######################################################
#DATASETS=("alpaca" "squadv2")      # ("alpaca" "squadv2") # Add more datasets if needed
#MODELS=("Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.3" "Llama-3.3-70B-Instruct") # "Mistral-7B-Instruct-v0.3" "Llama-3.1-8B-Instruct" "Llama-3.3-70B-Instruct") # Llama-3.3-70B-Instruct (Llama-3.1-8B") # "Mistral-7B-Instruct-v0.3" "Llama-3.3-70B-Instruct" "gemma-3-1b-it") # Add your models here
DATASETS=("alpaca")      # ("alpaca" "squadv2") # Add more datasets if needed
MODELS=("Llama-3.1-8B-Instruct") # "Mixtral-8x7B-v0.1" "Mistral-7B-Instruct-v0.3" "Llama-3.1-8B-Instruct" "Llama-3.3-70B-Instruct") # Llama-3.3-70B-Instruct (Llama-3.1-8B") # "Mistral-7B-Instruct-v0.3" "Llama-3.3-70B-Instruct" "gemma-3-1b-it") # Add your models here
NUMBER_OF_NODES=(2)


declare -A FRAMEWORK_PARALLELISM
#["torchrun"] : "fsdp ddp none"
#FRAMEWORK_PARALLELISM["torchrun"]="fsdp"
#["accelerate"]: "fsdp ddp none"
#FRAMEWORK_PARALLELISM["accelerate"]="fsdp"
#["deepspeed"] :"zero1 zero2 zero3 zero3-offload"
FRAMEWORK_PARALLELISM["deepspeed"]="zero3"

REPEATS=1                 # Number of runs per configuration
MACHINE="bsc-mn5-acc"
MACHINE_TYPE="cuda" # "cuda" or "rocm"
GPU_NAME=$GPU_NAME
#######################################################
# Set environment variables
#######################################################
set -a  # Automatically export all variables
source .env-$MACHINE
set +a  # Stop automatically exporting
#######################################################

# Load utility functions
source scripts/utils.sh
module load $MODULES


JOB_IDS=()
CONFIG_INDEX=0
CURRENT_DIR=$(pwd)
TOTAL_CONFIGS=$(( ${#DATASETS[@]} * ${#FRAMEWORKS[@]} * ${#NUMBER_OF_NODES[@]} * ${#MODELS[@]} * REPEATS ))

for framework in "${!FRAMEWORK_PARALLELISM[@]}"; do
  IFS=' ' read -r -a PARALLELISMS <<< "${FRAMEWORK_PARALLELISM[$framework]}"
  for dataset in "${DATASETS[@]}"; do
    DATASET_PATH=$(get_dataset_path "$dataset" "configs/config_datasets_paths_map.json")
    if [ -z "$DATASET_PATH" ] || [ "$DATASET_PATH" == "null" ]; then
      echo -e "${YELLOW}⚠️  No dataset path found for '$dataset' in configs/config_datasets_paths_map.json - skipping.${NC}"
      continue
    fi
    for model in "${MODELS[@]}"; do
      for NODES in "${NUMBER_OF_NODES[@]}"; do
        # Define which GPU configs to try
        if [[ "$NODES" -eq 1 ]]; then
          GPU_CONFIGS=(1 $GPUS_PER_NODE)   # both 1-GPU and Max-GPU
        else
          GPU_CONFIGS=($GPUS_PER_NODE)  # use default
        fi
        # GPU_CONFIGS=($GPUS_PER_NODE)  # use default

        for GPU_NODE in "${GPU_CONFIGS[@]}"; do
          for parallelism in "${PARALLELISMS[@]}"; do
            CONFIG_JSON=$(get_model_parallelism_config "$model" "$parallelism" "configs/model_parallelism_config.json")

            if [ -z "$CONFIG_JSON" ] || [ "$CONFIG_JSON" == "null" ]; then
              echo -e "${YELLOW}⚠️  No specific config for $model / $parallelism - continue with next configuration.${NC}"
              continue
            else
              # Read values from JSON
              BATCH_SIZES=($(echo "$CONFIG_JSON" | jq -r '.batch_size[]'))
              PRECISIONS=($(echo "$CONFIG_JSON" | jq -r '.precision[]'))
              GRAD_ACCUMS=($(echo "$CONFIG_JSON" | jq -r '.grad_accum[]'))
              LR=$(echo "$CONFIG_JSON" | jq -r '.lr')
              EPOCHS=$(echo "$CONFIG_JSON" | jq -r '.epochs // empty')
              STEPS=$(echo "$CONFIG_JSON" | jq -r '.steps // empty')
              MAX_MODEL_LENGTHS=($(echo "$CONFIG_JSON" | jq -r '.max_model_length // empty | .[]'))
            fi

            for batch in "${BATCH_SIZES[@]}"; do
              for precision in "${PRECISIONS[@]}"; do
                GPU_ARCHS_CONFIG="configs/gpu_archs.json"
                if [[ "$precision" == "fp32" ]]; then
                  export GPU_PEAK_TFLOPS=$(get_gpu_peak_tflops "$GPU_NAME" "theoretical_peak_fp32_tflops" "$GPU_ARCHS_CONFIG")
                elif [[ "$precision" == "fp16" ]]; then
                  export GPU_PEAK_TFLOPS=$(get_gpu_peak_tflops "$GPU_NAME" "theoretical_peak_fp16_tensor_tflops" "$GPU_ARCHS_CONFIG")
                elif [[ "$precision" == "bf16" ]]; then
                  export GPU_PEAK_TFLOPS=$(get_gpu_peak_tflops "$GPU_NAME" "theoretical_peak_bf16_tensor_tflops" "$GPU_ARCHS_CONFIG")
                fi

                for grad_accum in "${GRAD_ACCUMS[@]}"; do
                  for MAX_MODEL_LENGTH in "${MAX_MODEL_LENGTHS[@]}"; do
                    
                    # GENERAL PART (Common for all Frameworks).
                    TOTAL_GPUS=$((NODES * GPU_NODE))
                    TOTAL_CPUS=$((GPUS_PER_NODE * CPUS_PER_GPU))

                    BASE_FOLDER="results/${framework}/${dataset}/${model}"
                    RUN_FOLDER="Nodes_${NODES}-GPUs_${TOTAL_GPUS}-Parallelism_${parallelism}-Precision_${precision}-BS_${batch}-GAS_${grad_accum}-MaxModelLength_${MAX_MODEL_LENGTH}"
                    FULL_FOLDER="${BASE_FOLDER}/${RUN_FOLDER}"

                    MODEL_TYPE=$(get_model_type "$model" "configs/model_type_map.json")
                    MODEL_DIRECTORY=$(get_model_directory "$MODEL_TYPE" "configs/model_type_directories_map.json")
                    MODEL_PATH="${MODEL_DIRECTORY}/${model}"

                    if [ -z "$MODEL_DIRECTORY" ]; then
                      echo -e "${RED}Unknown model type '$MODEL_TYPE' or missing directory mapping. Exiting.${NC}"
                      exit 1
                    fi

                    # ----------------------------
                    #  Framework: PyTorch
                    # ----------------------------
                    if [[ "$framework" == "torchrun" ]]; then
                      echo -e "${BLUE}${BOLD}PyTorch Framework${NC}"

                      # Skip invalid configs - FSDP with less than 2 nodes
                      if [[ "$parallelism" == "fsdp" && "$GPU_NODE" -lt 2 ]]; then
                        echo -e "${YELLOW}Skipping FSDP on single-GPU (requires >1 GPUs)${NC}"
                        continue
                      fi
                      # Skip invalid configs - DDP with less than 2 nodes
                      if [[ "$parallelism" == "ddp" && "$GPU_NODE" -lt 2 ]]; then
                        echo -e "${YELLOW}Skipping DDP on single-GPU (requires >1 GPU)${NC}"
                        continue
                      fi
                      # Skip invalid configs - None parallelism with more than 1 node
                      if [[ "$parallelism" == "none" && "$GPU_NODE" -gt 1 ]]; then
                        echo -e "${YELLOW}Skipping None Parallelism on multiple-node (requires only 1 node)${NC}"
                        continue
                      fi
          
                      for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                        LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
                        echo -e "${CYAN}Setting up $LAUNCH_FOLDER${NC}"
                        mkdir -p "$LAUNCH_FOLDER"
                        
                        cp -R scripts/shared "$LAUNCH_FOLDER"
                        cp scripts/torchrun-common/run-$parallelism.sh "$LAUNCH_FOLDER"
                        cp scripts/torchrun-common/finetune-$parallelism.py "$LAUNCH_FOLDER"
                        cp scripts/torchrun-common/gpu_monitor.py "$LAUNCH_FOLDER"
                        cp scripts/gpu_plots.py "$LAUNCH_FOLDER"
                        cp scripts/torchrun-common/utils.py "$LAUNCH_FOLDER"
                        cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
                        cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"

                        cd "$LAUNCH_FOLDER" || exit 1

                        export CURRENT_DIR NODES GPUS_PER_NODE GPU_NODE MAX_MODEL_LENGTH TOTAL_CPUS EPOCHS STEPS LR
                        export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
                        export MODEL_PATH DATASET_PATH
                        export PARALLELISM="$parallelism"
                        export PRECISION="$precision"
                        export BATCH_SIZE="$batch"
                        export GRAD_ACCUM="$grad_accum"
                        export MODULES
                        export MACHINE
                        export MACHINE_TYPE

                        REMAINING=$((TOTAL_CONFIGS - CONFIG_INDEX))
                        if [ "$REMAINING" -le 5 ] && [ "${#JOB_IDS[@]}" -gt 0 ]; then
                          DEPENDENCY="--dependency=afterany:${JOB_IDS[-1]}"
                        else
                          DEPENDENCY=""
                        fi

                        JOB_ID=$(sbatch --parsable \
                            --chdir=$(pwd) \
                            --nodes=$NODES \
                            --gres=gpu:$GPUS_PER_NODE \
                            --cpus-per-task=$CPUS_PER_GPU \
                            --tasks-per-node=$GPUS_PER_NODE \
                            $DEPENDENCY \
                            --output=run-%j.out \
                            --error=run-%j.err \
                            -A $ACCOUNT \
                            -q $QOS \
                            run-$parallelism.sh "$LAUNCH_FOLDER" "$DATASET" "$DATASET_PATH")

                        echo -e "${GREEN}Submitted job ${BOLD}$JOB_ID ${NC}${GREEN}for $LAUNCH_FOLDER${NC}"
                        JOB_IDS+=("$JOB_ID")
                        ((CONFIG_INDEX++))

                        cd - > /dev/null
                        sleep 5
                      done
                    fi

                    # ----------------------------
                    #  Framework: Accelerate
                    # ----------------------------
                    if [[ "$framework" == "accelerate" ]]; then
                      echo -e "${BLUE}${BOLD}Accelerate Framework${NC}"
                      # Skip invalid configs - FSDP with less than 2 nodes
                      if [[ "$parallelism" == "fsdp" && "$GPU_NODE" -lt 2 ]]; then
                        echo -e "${YELLOW}Skipping FSDP on single-GPU (requires >1 GPUs)${NC}"
                        continue
                      fi
                      # Skip invalid configs - DDP with less than 2 nodes
                      if [[ "$parallelism" == "ddp" && "$GPU_NODE" -lt 2 ]]; then
                        echo -e "${YELLOW}Skipping DDP on single-GPU (requires >1 GPU)${NC}"
                        continue
                      fi
                      # Skip invalid configs - None parallelism with more than 1 node
                      if [[ "$parallelism" == "none" && "$GPU_NODE" -gt 1 ]]; then
                        echo -e "${YELLOW}Skipping None Parallelism on multiple-node (requires only 1 node)${NC}"
                        continue
                      fi
                      # Skip Gemma batch_size=1 when using more than 1 GPU
                      if [[ "$model" == "gemma-3-1b-it" && "$batch" -eq 1 && "$GPU_NODE" -gt 1 ]]; then
                        echo -e "${YELLOW}Skipping Gemma (batch_size=1) with ${GPU_NODE} GPUs.${NC}"
                        continue
                      fi

                      for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                        LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
                        echo -e "${CYAN}Setting up $LAUNCH_FOLDER${NC}"
                        mkdir -p "$LAUNCH_FOLDER"

                        # Copy necessary scripts
                        cp -R scripts/shared "$LAUNCH_FOLDER"
                        cp scripts/accelerate-common/run-$parallelism.sh "$LAUNCH_FOLDER"
                        cp scripts/accelerate-common/finetune-$parallelism.py "$LAUNCH_FOLDER"
                        cp scripts/accelerate-common/gpu_monitor.py "$LAUNCH_FOLDER"
                        cp scripts/gpu_plots.py "$LAUNCH_FOLDER"
                        cp scripts/accelerate-common/utils.py "$LAUNCH_FOLDER"
                        cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
                        cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"

                        cd "$LAUNCH_FOLDER" || exit 1

                        export CURRENT_DIR NODES GPUS_PER_NODE GPU_NODE MAX_MODEL_LENGTH TOTAL_CPUS EPOCHS STEPS LR
                        export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
                        export MODEL_PATH DATASET_PATH
                        export PARALLELISM="$parallelism"
                        export PRECISION="$precision"
                        export BATCH_SIZE="$batch"
                        export GRAD_ACCUM="$grad_accum"
                        export MODULES
                        export MACHINE
                        export MACHINE_TYPE
                        
                        REMAINING=$((TOTAL_CONFIGS - CONFIG_INDEX))
                        if [ "$REMAINING" -le 5 ] && [ "${#JOB_IDS[@]}" -gt 0 ]; then
                          DEPENDENCY="--dependency=afterany:${JOB_IDS[-1]}"
                        else
                          DEPENDENCY=""
                        fi

                        JOB_ID=$(sbatch --parsable \
                            --chdir=$(pwd) \
                            --nodes=$NODES \
                            --gres=gpu:$GPUS_PER_NODE \
                            --cpus-per-task=80 \
                            --tasks-per-node=1 \
                            $DEPENDENCY \
                            --output=run-%j.out \
                            --error=run-%j.err \
                            -A $ACCOUNT \
                            -q $QOS \
                            run-$parallelism.sh "$LAUNCH_FOLDER" "$DATASET" "$DATASET_PATH")

                        echo -e "${GREEN}Submitted job ${BOLD}$JOB_ID${NC}${GREEN} for $LAUNCH_FOLDER${NC}"
                        JOB_IDS+=("$JOB_ID")
                        ((CONFIG_INDEX++))

                        cd - > /dev/null
                        sleep 5
                      done
                    fi

                    # ----------------------------
                    #  Framework: DeepSpeed
                    # ----------------------------
                    # Keep your original deepspeed blocks here unchanged
                    if [[ "$framework" == "deepspeed" ]]; then
                      echo -e "${BLUE}${BOLD}Deepspeed Framework${NC}"
                      PERMITTED_ZERO_STAGES=("zero1" "zero2" "zero3" "zero3-offload")

                      if ! exists_in_list "${PERMITTED_ZERO_STAGES[*]}" " " "$parallelism"; then
                          echo -e "${RED}Error: TYPE_PARALLELISM must be one of: ${PERMITTED_ZERO_STAGES[*]}${NC}"
                          echo -e "${RED}Received: '$parallelism'${NC}"
                          continue
                      fi
                      if [[ "$GPU_NODE" -eq 1 ]]; then
                        echo -e "${YELLOW}Skipping Deepspeed with 1 GPU (requires >1 GPU)${NC}"
                        continue
                      fi

                      # Skip Gemma batch_size=1 when using more than 1 GPU
                      #if [[ "$model" == "gemma-3-1b-it" && "$batch" -eq 1 && "$GPU_NODE" -gt 1 ]]; then
                      #  echo "Skipping Gemma (batch_size=1) with ${GPU_NODE} GPUs."
                      #  continue
                      #fi

                      for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                        LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
                        echo -e "${CYAN}Setting up $LAUNCH_FOLDER${NC}"
                        mkdir -p "$LAUNCH_FOLDER"

                        # Copy necessary scripts
                        cp -R scripts/shared "$LAUNCH_FOLDER"
                        cp scripts/deepspeed-common/run-deepspeed.sh "$LAUNCH_FOLDER"
                        cp scripts/deepspeed-common/finetune-deepspeed.py "$LAUNCH_FOLDER"
                        cp -R scripts/deepspeed-common/configs "$LAUNCH_FOLDER"
                        cp scripts/deepspeed-common/gpu_monitor.py "$LAUNCH_FOLDER"
                        cp scripts/gpu_plots.py "$LAUNCH_FOLDER"
                        cp scripts/deepspeed-common/utils.py "$LAUNCH_FOLDER"
                        cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
                        cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"

                        cd "$LAUNCH_FOLDER" || exit 1

                        export CURRENT_DIR NODES GPUS_PER_NODE GPU_NODE MAX_MODEL_LENGTH TOTAL_CPUS EPOCHS STEPS LR
                        export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
                        export MODEL_PATH DATASET_PATH
                        export PARALLELISM="$parallelism"
                        export PRECISION="$precision"
                        export BATCH_SIZE="$batch"
                        export GRAD_ACCUM="$grad_accum"
                        export MODULES
                        export MACHINE
                        export MACHINE_TYPE
                        
                        REMAINING=$((TOTAL_CONFIGS - CONFIG_INDEX))
                        if [ "$REMAINING" -le 5 ] && [ "${#JOB_IDS[@]}" -gt 0 ]; then
                          DEPENDENCY="--dependency=afterany:${JOB_IDS[-1]}"
                        else
                          DEPENDENCY=""
                        fi

                        JOB_ID=$(sbatch --parsable \
                            --chdir=$(pwd) \
                            --nodes=$NODES \
                            --gres=gpu:$GPUS_PER_NODE \
                            --cpus-per-task=80 \
                            --tasks-per-node=1 \
                            $DEPENDENCY \
                            --output=run-%j.out \
                            --error=run-%j.err \
                            -A $ACCOUNT \
                            -q $QOS \
                            run-deepspeed.sh "$LAUNCH_FOLDER" "$DATASET" "$DATASET_PATH" "$parallelism")

                        echo -e "${GREEN}Submitted job ${BOLD}$JOB_ID${NC}${GREEN} for $LAUNCH_FOLDER${NC}"
                        JOB_IDS+=("$JOB_ID")
                        ((CONFIG_INDEX++))

                        cd - > /dev/null
                        sleep 5
                      done
                    fi


                  done
                done
              done
            done
          done 
        done
      done
    done
  done
done