#!/bin/bash

#######################################################
# ENVIRONMENT VARIABLES TO CHANGE
#######################################################
# SPECIFIC CASE FOR TESTING
#######################################################
FRAMEWORKS=("vllm") # ("vllm" "sglang")    # Add other frameworks if needed
DATASETS=("sonnet") # ("sharegpt" "sonnet")  # Add more datasets if needed
MODELS=("Llama-3.1-8B-Instruct" "gemma-3-12b-it" "Mistral-7B-Instruct-v0.3") # ("Llama-3.1-8B-Instruct" "gemma-3-12b-it" "Mistral-7B-Instruct-v0.3" "Llama-3.3-70B-Instruct" "Llama-3.1-405B") # Add your models here
NUMBER_OF_NODES=(1)
MAX_MODEL_LENGTHS=(16384) # (4096 8192 16384 32768)
REPEATS=1                 # Number of runs per configuration
MACHINE="cines-adastra-mi300"  # cines-adastra-mi250 | cines-adastra-mi300
MACHINE_TYPE="rocm" # "cuda" or "rocm"
#######################################################
# Set environment variables
#######################################################
set -a  # Automatically export all variables
source .env-$MACHINE
set +a  # Stop automatically exporting

# Load utility functions
source scripts/utils.sh

# SLURM args specific to machine
case "$MACHINE" in
    cines-adastra-mi250 | cines-adastra-mi300)
        SLURM_GPU_ARG="--gpus-per-node=$GPUS_PER_NODE"
        SLURM_CONSTRAINT="--constraint=$PARTITION_NAME"
        SLURM_QOS="--exclusive"
        ;;
    *)
        SLURM_GPU_ARG="--gres=gpu:$GPUS_PER_NODE"
        SLURM_CONSTRAINT=""
        SLURM_QOS="-q $QOS"
        ;;
esac
#######################################################

JOB_IDS=()
CONFIG_INDEX=0
CURRENT_DIR=$(pwd)
TOTAL_CONFIGS=$(( ${#DATASETS[@]} * ${#FRAMEWORKS[@]} * ${#NUMBER_OF_NODES[@]} * ${#MODELS[@]} * REPEATS ))

for framework in "${FRAMEWORKS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for NODES in "${NUMBER_OF_NODES[@]}"; do
        # Define which GPU configs to try
        # if [[ "$NODES" -eq 1 ]]; then
        #   GPU_CONFIGS=(1 $GPUS_PER_NODE)   # both 1-GPU and Max-GPU
        # else
        GPU_CONFIGS=($GPUS_PER_NODE)  # use default

        for GPU_NODE in "${GPU_CONFIGS[@]}"; do
          for MAX_MODEL_LENGTH in "${MAX_MODEL_LENGTHS[@]}"; do
            # GENERAL PART (Common for all Frameworks).
            TOTAL_GPUS=$((NODES * GPU_NODE))
            TOTAL_CPUS=$((GPUS_PER_NODE * CPUS_PER_GPU))
            TENSOR_PARALLEL=$TOTAL_GPUS
            PIPELINE_PARALLEL=1

            BASE_FOLDER="results/${framework}/${dataset}/${model}"
            RUN_FOLDER="Nodes_${NODES}-GPUs_${TOTAL_GPUS}-TP_${TENSOR_PARALLEL}-PP_${PIPELINE_PARALLEL}-MaxModelLength_${MAX_MODEL_LENGTH}"
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
              if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" ]]; then
                # Skip if model is Llama-3.1-405B and NODES < 4.
                if [[ "$NODES" -lt 4 ]]; then
                  echo "Skipping $model with $NODES nodes (requires at least 4 nodes)"
                  continue
                fi
                # Set extra args for Llama-3.1-405B
                ADDITIONAL_ARGS="--disable-log-requests --enforce-eager"
              fi
              ADDITIONAL_ARGS="--disable-log-requests --enforce-eager"
              
              for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
                echo "Setting up $LAUNCH_FOLDER"
                mkdir -p "$LAUNCH_FOLDER"
                
                cp scripts/vllm/run_cluster.sh "$LAUNCH_FOLDER"
                cp scripts/vllm/vllm_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER"
                cp scripts/vllm/serve.sh "$LAUNCH_FOLDER"
                cp scripts/vllm/gpu_summary_monitor-$MACHINE_TYPE.py "$LAUNCH_FOLDER"
                cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
                cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"

                cd "$LAUNCH_FOLDER" || exit 1

                export NODES GPUS_PER_NODE GPU_NODE TENSOR_PARALLEL PIPELINE_PARALLEL MAX_MODEL_LENGTH TOTAL_CPUS
                export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
                export MODEL_PATH  # Make available to launched script
                export ADDITIONAL_ARGS
                export MODULES
                export MACHINE
                export MACHINE_TYPE

                MAX_PARALLEL=$MAX_JOBS
                RUNNING=${#JOB_IDS[@]}
                if [ "$RUNNING" -ge "$MAX_PARALLEL" ]; then
                    DEPENDENCY="--dependency=afterany:${JOB_IDS[-$MAX_PARALLEL]}"
                else
                    DEPENDENCY=""
                fi

                JOB_ID=$(sbatch --parsable \
                    --chdir=$(pwd) \
                    --nodes=$NODES \
                    $SLURM_GPU_ARG \
                    --cpus-per-task=$TOTAL_CPUS \
                    $SLURM_CONSTRAINT \
                    $SLURM_QOS \
                    $DEPENDENCY \
                    --output=run-%j.out \
                    --error=run-%j.out \
                    -A $ACCOUNT \
                    vllm_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH" "$MACHINE" "$MACHINE_TYPE")

                echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
                JOB_IDS+=("$JOB_ID")
                ((CONFIG_INDEX++))

                cd - > /dev/null
                sleep 5
              done
            fi

            # SGLang
            if [[ "$framework" == "sglang" ]]; then
              # SGLang
              echo "FrameWork SGLang"

              # If Model is Llama-3.1-405B.
              if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" ]]; then
                # Skip if model is Llama-3.1-405B and NODES < 4.
                if [[ "$NODES" -lt 4 ]]; then
                  echo "Skipping $model with $NODES nodes (requires at least 4 nodes)"
                  continue
                fi
                ADDITIONAL_ARGS=""
              fi
              # If Model is 'gemma-3-12b-it'
              if [[ "$model" == "gemma-3-12b-it" ]]; then
                # Skip if model is 'gemma-3-12b-it' and NODES > 1.
                if [[ "$NODES" -gt 1 ]]; then
                  echo "Skipping $model with $NODES nodes (tp and pp cannot be set in SGLang Framework) for 'gemma-3-12b-it' model"
                  continue
                fi
                ADDITIONAL_ARGS=""
              fi
              ADDITIONAL_ARGS=""
              
              for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
                echo "Setting up $LAUNCH_FOLDER"
                mkdir -p "$LAUNCH_FOLDER"
                
                cp scripts/sglang/sglang_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER"
                cp scripts/sglang/gpu_summary_monitor-$MACHINE_TYPE.py "$LAUNCH_FOLDER"
                cp scripts/sglang/serve.sh "$LAUNCH_FOLDER"
                cp scripts/sglang/wrapper_singularity.sh "$LAUNCH_FOLDER"
                cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
                cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"

                cd "$LAUNCH_FOLDER" || exit 1

                export NODES GPUS_PER_NODE GPU_NODE TENSOR_PARALLEL PIPELINE_PARALLEL MAX_MODEL_LENGTH TOTAL_CPUS
                export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
                export MODEL_PATH  # Make available to launched script
                export ADDITIONAL_ARGS
                export MODULES
                export MACHINE
                export MACHINE_TYPE

                MAX_PARALLEL=$MAX_JOBS
                RUNNING=${#JOB_IDS[@]}
                if [ "$RUNNING" -ge "$MAX_PARALLEL" ]; then
                    DEPENDENCY="--dependency=afterany:${JOB_IDS[-$MAX_PARALLEL]}"
                else
                    DEPENDENCY=""
                fi

                JOB_ID=$(sbatch --parsable \
                    --chdir=$(pwd) \
                    --nodes=$NODES \
                    $SLURM_GPU_ARG \
                    --cpus-per-task=$TOTAL_CPUS \
                    $SLURM_CONSTRAINT \
                    $SLURM_QOS \
                    $DEPENDENCY \
                    --output=run-%j.out \
                    --error=run-%j.out \
                    -A $ACCOUNT \
                    sglang_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH")

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