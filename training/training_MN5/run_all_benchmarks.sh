#!/bin/bash

#######################################################
# ENVIRONMENT VARIABLES TO CHANGE
#######################################################
# SPECIFIC CASE FOR TESTING
#######################################################
FRAMEWORKS=("torchrun")   # ("torchrun" "accelerate" "deepspeed")    # Add other frameworks if needed
DATASETS=("alpaca" "squadv2") # Add more datasets if needed
MODELS=("gemma-3-1b-it" "Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.3" "Llama-3.3-70B-Instruct") #"gemma-3-1b-it") # Add your models here
NUMBER_OF_NODES=(1 4)
TYPE_PARALLELISM=("fsdp" "ddp" "none")
REPEATS=1                 # Number of runs per configuration
MACHINE="bsc-mn5-acc"
MACHINE_TYPE="cuda" # "cuda" or "rocm"
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

for framework in "${FRAMEWORKS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
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
          for parallelism in "${TYPE_PARALLELISM[@]}"; do
            CONFIG_JSON=$(get_model_parallelism_config "$model" "$parallelism" "configs/model_parallelism_config.json")

            if [ -z "$CONFIG_JSON" ] || [ "$CONFIG_JSON" == "null" ]; then
              echo "⚠️ No specific config for $model / $parallelism - continue with next configuration."
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
                      echo "Unknown model type '$MODEL_TYPE' or missing directory mapping. Exiting."
                      exit 1
                    fi

                    DATASET_PATH=$(get_dataset_path "$dataset" "configs/config_datasets_paths_map.json")

                    # ----------------------------
                    #  Framework: PyTorch
                    # ----------------------------
                    if [[ "$framework" == "torchrun" ]]; then
                      echo "PyTorch Framework"

                      # Skip invalid configs - FSDP with less than 2 nodes
                      if [[ "$parallelism" == "fsdp" && "$NODES" -lt 2 ]]; then
                        echo "Skipping FSDP on single-node (requires >1 node)"
                        continue
                      fi
                      # Skip invalid configs - DDP with less than 2 nodes
                      if [[ "$parallelism" == "ddp" && "$NODES" -lt 2 ]]; then
                        echo "Skipping DDP on single-node (requires >1 node)"
                        continue
                      fi
                      # Skip invalid configs - None parallelism with more than 1 node
                      if [[ "$parallelism" == "none" && "$NODES" -gt 1 ]]; then
                        echo "Skipping None Parallelism on multiple-node (requires only 1 node)"
                        continue
                      fi
                      # Skip Gemma batch_size=1 when using more than 1 GPU
                      if [[ "$model" == "gemma-3-1b-it" && "$batch" -eq 1 && "$GPU_NODE" -gt 1 ]]; then
                        echo "Skipping Gemma (batch_size=1) with ${GPU_NODE} GPUs."
                        continue
                      fi

                      for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                        LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
                        echo "Setting up $LAUNCH_FOLDER"
                        mkdir -p "$LAUNCH_FOLDER"

                        cp scripts/torchrun-common/run-$parallelism.sh "$LAUNCH_FOLDER"
                        cp scripts/torchrun-common/finetune-$parallelism.py "$LAUNCH_FOLDER"
                        cp scripts/torchrun-common/custom_train.py "$LAUNCH_FOLDER"
                        cp scripts/torchrun-common/gpu_monitor.py "$LAUNCH_FOLDER"
                        cp scripts/torchrun-common/utils.py "$LAUNCH_FOLDER"
                        cp scripts/gpu_plots.py "$LAUNCH_FOLDER"
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

                        echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
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
                      echo "Accelerate Framework"
                      # Skip invalid configs - FSDP with less than 2 nodes
                      if [[ "$parallelism" == "fsdp" && "$NODES" -lt 2 ]]; then
                        echo "Skipping FSDP on single-node (requires >1 node)"
                        continue
                      fi
                      # Skip invalid configs - DDP with less than 2 nodes
                      if [[ "$parallelism" == "ddp" && "$NODES" -lt 2 ]]; then
                        echo "Skipping DDP on single-node (requires >1 node)"
                        continue
                      fi
                      # Skip invalid configs - None parallelism with more than 1 node
                      if [[ "$parallelism" == "none" && "$NODES" -gt 1 ]]; then
                        echo "Skipping None Parallelism on multiple-node (requires only 1 node)"
                        continue
                      fi
                      # Skip Gemma batch_size=1 when using more than 1 GPU
                      if [[ "$model" == "gemma-3-1b-it" && "$batch" -eq 1 && "$GPU_NODE" -gt 1 ]]; then
                        echo "Skipping Gemma (batch_size=1) with ${GPU_NODE} GPUs."
                        continue
                      fi

                      for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                        LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
                        echo "Setting up $LAUNCH_FOLDER"
                        mkdir -p "$LAUNCH_FOLDER"

                        # Copy necessary scripts
                        cp scripts/accelerate-common/run-$parallelism.sh "$LAUNCH_FOLDER"
                        cp scripts/accelerate-common/finetune-$parallelism.py "$LAUNCH_FOLDER"
                        cp scripts/accelerate-common/custom_train.py "$LAUNCH_FOLDER"
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

                        echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
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