#!/bin/bash

#######################################################
# ENVIRONMENT VARIABLES TO CHANGE
#######################################################
# SPECIFIC CASE FOR TESTING
#######################################################
FRAMEWORKS=("torchrun")
DATASETS=("alpaca")
MODELS=("Llama-3.1-8B-Instruct")
NUMBER_OF_NODES=(1)
TYPE_PARALLELISM=("none")  # "none" "ddp" "fsdp"
REPEATS=1
MACHINE="cines-adastra-mi300"  # cines-adastra-mi250 | cines-adastra-mi300
MACHINE_TYPE="rocm"
#######################################################
# Set environment variables
#######################################################
set -a
source .env-$MACHINE
set +a

# Load utility functions
source scripts/utils.sh
#######################################################

# SLURM args specific to machine
case "$MACHINE" in
    cines-adastra-mi250 | cines-adastra-mi300)
        SLURM_GPU_ARG="--gpus-per-node=$GPUS_PER_NODE"
        SLURM_CONSTRAINT="--constraint=$PARTITION_NAME"
        SLURM_QOS=""
        ;;
    *)
        SLURM_GPU_ARG="--gres=gpu:$GPUS_PER_NODE"
        SLURM_CONSTRAINT=""
        SLURM_QOS="-q $QOS"
        ;;
esac

JOB_IDS=()
CONFIG_INDEX=0
CURRENT_DIR=$(pwd)
TOTAL_CONFIGS=$(( ${#DATASETS[@]} * ${#FRAMEWORKS[@]} * ${#NUMBER_OF_NODES[@]} * ${#MODELS[@]} * REPEATS ))

for framework in "${FRAMEWORKS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for NODES in "${NUMBER_OF_NODES[@]}"; do
        GPU_CONFIGS=($GPUS_PER_NODE)

        for GPU_NODE in "${GPU_CONFIGS[@]}"; do
          for parallelism in "${TYPE_PARALLELISM[@]}"; do
            CONFIG_JSON=$(get_model_parallelism_config "$model" "$parallelism" "configs/model_parallelism_config.json")

            if [ -z "$CONFIG_JSON" ] || [ "$CONFIG_JSON" == "null" ]; then
              echo "⚠️ No specific config for $model / $parallelism - continue with next configuration."
              continue
            else
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

                    # GENERAL PART
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
                    #  Framework: torchrun
                    # ----------------------------
                    if [[ "$framework" == "torchrun" ]]; then
                      echo "PyTorch Framework"

                      # Skip FSDP on single-node
                      if [[ "$parallelism" == "fsdp" && "$NODES" -lt 2 ]]; then
                        echo "Skipping FSDP on single-node (requires >1 node)"
                        continue
                      fi
                      # Skip DDP on single-node
                      if [[ "$parallelism" == "ddp" && "$NODES" -lt 2 ]]; then
                        echo "Skipping DDP on single-node (requires >1 node)"
                        continue
                      fi
                      # Skip None parallelism on multi-node
                      if [[ "$parallelism" == "none" && "$NODES" -gt 1 ]]; then
                        echo "Skipping None Parallelism on multiple-node (requires only 1 node)"
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
                            $SLURM_GPU_ARG \
                            --cpus-per-task=$TOTAL_CPUS \
                            --tasks-per-node=1 \
                            $SLURM_CONSTRAINT \
                            $SLURM_QOS \
                            $DEPENDENCY \
                            --output=run-%j.out \
                            --error=run-%j.err \
                            -A $ACCOUNT \
                            run-$parallelism.sh "$LAUNCH_FOLDER" "$DATASET" "$DATASET_PATH")

                        echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
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