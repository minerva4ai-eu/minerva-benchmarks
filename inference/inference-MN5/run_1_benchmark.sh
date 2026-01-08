#!/bin/bash
<<<<<<< HEAD
set -a  # Automatically export all variables
source .env
set +a  # Stop automatically exporting

# Load utility functions
source scripts/utils.sh
=======
>>>>>>> c9f2946 (Initial commit from GitLab)

#######################################################
# ENVIRONMENT VARIABLES TO CHANGE
#######################################################
# SPECIFIC CASE FOR TESTING
#######################################################
<<<<<<< HEAD
FRAMEWORKS=("vllm") #"vllm") # deepspeed")    # Add other frameworks if needed
DATASETS=("sharegpt")  # Add more datasets if needed
MODELS=("Llama-3.1-8B-Instruct") # ("Llama-3.1-405B" "gemma-3-12b-it" "Mistral-7B-Instruct-v0.3") # Add your models here
NUMBER_OF_NODES=(1)
MAX_MODEL_LENGTHS=(4096) # 4096 8192 16384 32768)
REPEATS=1                 # Number of runs per configuration
#######################################################

=======
FRAMEWORKS=("sglang") #"vllm") # deepspeed")    # Add other frameworks if needed
DATASETS=("sharegpt") #"sonnet")  # Add more datasets if needed
MODELS=("Llama-3.1-8B-Instruct") #Llama-3.3-70B-Instruct") # "Llama-3.1-405B") # ("Llama-3.1-405B" "gemma-3-12b-it" "Mistral-7B-Instruct-v0.3") # Add your models here
NUMBER_OF_NODES=(1)
MAX_MODEL_LENGTHS=(4096) # 16384 32768) # 4096 8192 16384 32768)
REPEATS=1                 # Number of runs per configuration
MACHINE="bsc-mn5-acc"
MACHINE_TYPE="cuda" # "cuda" or "rocm"
#######################################################
# Set environment variables
#######################################################
set -a  # Automatically export all variables
source .env-$MACHINE
set +a  # Stop automatically exporting

# Load utility functions
source scripts/utils.sh
#######################################################
>>>>>>> c9f2946 (Initial commit from GitLab)

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
<<<<<<< HEAD
              if [[ "$model" == "Llama-3.1-405B" ]]; then
=======
              if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" ]]; then
>>>>>>> c9f2946 (Initial commit from GitLab)
                # Skip if model is Llama-3.1-405B and NODES < 4.
                if [[ "$NODES" -lt 4 ]]; then
                  echo "Skipping $model with $NODES nodes (requires at least 4 nodes)"
                  continue
                fi
                # Set extra args for Llama-3.1-405B
<<<<<<< HEAD
                ADDITIONAL_ARGS="--cpu-offload-gb 0.5"
              fi
              # If Model is Llama-3.1-8B-Instruct.
              if [[ "$model" == "Llama-3.1-8B-Instruct" ]]; then
                # Set extra args for Llama-3.1-8B-Instruct
                ADDITIONAL_ARGS="--cpu-offload-gb 0.5"
              fi

=======
                ADDITIONAL_ARGS="--disable-log-requests --enforce-eager"
              fi
              ADDITIONAL_ARGS="--disable-log-requests --enforce-eager"
              
>>>>>>> c9f2946 (Initial commit from GitLab)
              for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
                echo "Setting up $LAUNCH_FOLDER"
                mkdir -p "$LAUNCH_FOLDER"
                
                cp scripts/vllm/run_cluster.sh "$LAUNCH_FOLDER"
                cp scripts/vllm/vllm_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER"
                cp scripts/vllm/serve.sh "$LAUNCH_FOLDER"
<<<<<<< HEAD
                cp scripts/vllm/gpu_summary_monitor.py "$LAUNCH_FOLDER"
=======
                cp scripts/vllm/gpu_summary_monitor-$MACHINE_TYPE.py "$LAUNCH_FOLDER"
                cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
                cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"
>>>>>>> c9f2946 (Initial commit from GitLab)

                cd "$LAUNCH_FOLDER" || exit 1

                export NODES GPUS_PER_NODE GPU_NODE TENSOR_PARALLEL PIPELINE_PARALLEL MAX_MODEL_LENGTH TOTAL_CPUS
                export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
                export MODEL_PATH  # Make available to launched script
                export ADDITIONAL_ARGS
                export MODULES
<<<<<<< HEAD
=======
                export MACHINE
                export MACHINE_TYPE
>>>>>>> c9f2946 (Initial commit from GitLab)

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
                    --cpus-per-task=$TOTAL_CPUS \
                    $DEPENDENCY \
                    --output=run-%j.out \
                    --error=run-%j.err \
                    -A $ACCOUNT \
                    -q $QOS \
<<<<<<< HEAD
                    vllm_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH")
=======
                    vllm_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH" "$MACHINE" "$MACHINE_TYPE")
>>>>>>> c9f2946 (Initial commit from GitLab)

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
              
<<<<<<< HEAD
              # If Model is Llama-3.1-405B.
              if [[ "$model" == "Llama-3.1-405B" ]]; then
=======
              # If Model is Llama-3.1-405B, avoid it.
              if [[ "$model" == "Llama-3.1-405B" || "$model" == "Llama-3.1-405B-Instruct" || "$model" == "Llama-3.70B-Instruct" ]]; then
                continue
              fi
              # If Model is gemma-3-12b-it, avoid it.
              if [[ "$model" == "gemma-3-12b-it" ]]; then
                continue
              fi
              # If Model is Mistral-7B-Instruct-v0.3, avoid it.
              if [[ "$model" == "Mistral-7B-Instruct-v0.3" ]]; then
>>>>>>> c9f2946 (Initial commit from GitLab)
                continue
              fi
              # If iteration has more than 1 Node, avoid it.
              if [[ "$NODES" != 1 ]]; then
                echo "Skipping deepspeed-mii $model with $NODES nodes (requires maximum 1 node)"
                continue
              fi
<<<<<<< HEAD
              REPEATS=1
=======
              ADDITIONAL_ARGS=""
>>>>>>> c9f2946 (Initial commit from GitLab)

              for (( run_id=1; run_id<=REPEATS; run_id++ )); do
                LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
                echo "Setting up $LAUNCH_FOLDER"
                mkdir -p "$LAUNCH_FOLDER"
                
                cp scripts/deepspeed/serve_deepspeed_mii.py "$LAUNCH_FOLDER"
                cp scripts/deepspeed/deepspeed-mii_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER"
<<<<<<< HEAD
                cp scripts/deepspeed/gpu_summary_monitor.py "$LAUNCH_FOLDER"
=======
                cp scripts/deepspeed/gpu_summary_monitor-$MACHINE_TYPE.py "$LAUNCH_FOLDER"
                cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
                cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"
>>>>>>> c9f2946 (Initial commit from GitLab)

                cd "$LAUNCH_FOLDER" || exit 1

                export NODES GPUS_PER_NODE GPU_NODE TENSOR_PARALLEL PIPELINE_PARALLEL MAX_MODEL_LENGTH TOTAL_CPUS
                export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id"
                export MODEL_PATH  # Make available to launched script
                export ADDITIONAL_ARGS
                export MODULES
<<<<<<< HEAD
                
=======
                export MACHINE
                export MACHINE_TYPE

>>>>>>> c9f2946 (Initial commit from GitLab)
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
                    --cpus-per-task=$TOTAL_CPUS \
                    $DEPENDENCY \
                    --output=run-%j.out \
                    --error=run-%j.err \
                    -A $ACCOUNT \
                    -q $QOS \
                    deepspeed-mii_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH")

                echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
                JOB_IDS+=("$JOB_ID")
                ((CONFIG_INDEX++))

                cd - > /dev/null
                sleep 5
              done
            fi
<<<<<<< HEAD
=======

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
                ADDITIONAL_ARGS="" # --mem-fraction-static 0.80 --chunked-prefill-size 4096
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
                    --cpus-per-task=$TOTAL_CPUS \
                    --output=run-%j.out \
                    --error=run-%j.err \
                    -A $ACCOUNT \
                    -q $QOS \
                    sglang_configurable_benchmarking_serve.sh "$LAUNCH_FOLDER" "$BENCHMARK_FILE" "$DATASET" "$DATASET_PATH")

                echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
                JOB_IDS+=("$JOB_ID")
                ((CONFIG_INDEX++))

                cd - > /dev/null
                sleep 5
              done
            fi

>>>>>>> c9f2946 (Initial commit from GitLab)
          done
        done
      done
    done
  done
done