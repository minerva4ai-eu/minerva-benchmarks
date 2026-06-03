#!/bin/bash
# Resubmits the 8 configs that have no valid training data.
# Must be run from the training_MN5 directory.
# All jobs are independent (no dependency chain).

set -a
source .env-leonardo
set +a

CURRENT_DIR=$(pwd)
MACHINE="leonardo"
MACHINE_TYPE="cuda"
MODEL="Llama-3.1-8B-Instruct"
NODES=4
GPUS_PER_NODE=4
TOTAL_CPUS=$((GPUS_PER_NODE * CPUS_PER_GPU))   # 32
PRECISION="bf16"
BATCH_SIZES=(4 8 16)
GRAD_ACCUM=4
MAX_MODEL_LENGTH=2048
LR=2e-05
EPOCHS=3
PARALLELISM="fsdp"
RESULTS_BASE_DIR="/leonardo_work/MNRVA_bench/davide_results"

module load $MODULES

submit_torchrun() {
  local dataset=$1 batch=$2 dataset_path=$3
  local launch_folder="${RESULTS_BASE_DIR}/torchrun/${dataset}/${MODEL}/Nodes_${NODES}-GPUs_$((NODES*GPUS_PER_NODE))-Parallelism_${PARALLELISM}-Precision_${PRECISION}-BS_${batch}-GAS_${GRAD_ACCUM}-MaxModelLength_${MAX_MODEL_LENGTH}/launch-1"

  export CURRENT_DIR NODES GPUS_PER_NODE MAX_MODEL_LENGTH TOTAL_CPUS EPOCHS LR
  export MODEL MODEL_PATH="${MODEL_DIRECTORY:-/leonardo_work/cin_staff/dgentile/models_registry}/${MODEL}"
  export DATASET="$dataset" DATASET_PATH="$dataset_path"
  export PARALLELISM PRECISION BATCH_SIZE="$batch" GRAD_ACCUM MODULES MACHINE MACHINE_TYPE

  cd "$launch_folder" || { echo "ERROR: cannot cd to $launch_folder"; return 1; }
  JOB_ID=$(sbatch --parsable \
    --chdir="$(pwd)" \
    --nodes=$NODES \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_GPU \
    --tasks-per-node=$GPUS_PER_NODE \
    --output=run-%j.out \
    --error=run-%j.err \
    -A llm4e_prod -p $PARTITION_NAME -q $QOS \
    run-fsdp.sh "$launch_folder" "$dataset" "$dataset_path")
  echo "Submitted torchrun/$dataset/BS${batch} → job $JOB_ID"
  cd - > /dev/null
}

submit_accelerate() {
  local dataset=$1 batch=$2 dataset_path=$3
  local launch_folder="${RESULTS_BASE_DIR}/accelerate/${dataset}/${MODEL}/Nodes_${NODES}-GPUs_$((NODES*GPUS_PER_NODE))-Parallelism_${PARALLELISM}-Precision_${PRECISION}-BS_${batch}-GAS_${GRAD_ACCUM}-MaxModelLength_${MAX_MODEL_LENGTH}/launch-1"

  export CURRENT_DIR NODES GPUS_PER_NODE MAX_MODEL_LENGTH TOTAL_CPUS EPOCHS LR
  export MODEL MODEL_PATH="${MODEL_DIRECTORY:-/leonardo_work/cin_staff/dgentile/models_registry}/${MODEL}"
  export DATASET="$dataset" DATASET_PATH="$dataset_path"
  export PARALLELISM PRECISION BATCH_SIZE="$batch" GRAD_ACCUM MODULES MACHINE MACHINE_TYPE

  cd "$launch_folder" || { echo "ERROR: cannot cd to $launch_folder"; return 1; }
  JOB_ID=$(sbatch --parsable \
    --chdir="$(pwd)" \
    --nodes=$NODES \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$TOTAL_CPUS \
    --tasks-per-node=1 \
    --output=run-%j.out \
    --error=run-%j.err \
    -A llm4e_prod -p $PARTITION_NAME -q $QOS \
    run-fsdp.sh "$launch_folder" "$dataset" "$dataset_path")
  echo "Submitted accelerate/$dataset/BS${batch} → job $JOB_ID"
  cd - > /dev/null
}

SHAREGPT_PATH="/leonardo_work/MNRVA_bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
SONNET_PATH="/leonardo_work/MNRVA_bench/datasets/sonnet.txt"

echo "=== Submitting 8 missing configs ==="

# accelerate/sharegpt/BS4 and BS8 (BS16 already has data)
submit_accelerate "sharegpt" 4  "$SHAREGPT_PATH"
submit_accelerate "sharegpt" 8  "$SHAREGPT_PATH"

# torchrun/sharegpt all BS
submit_torchrun "sharegpt" 4  "$SHAREGPT_PATH"
submit_torchrun "sharegpt" 8  "$SHAREGPT_PATH"
submit_torchrun "sharegpt" 16 "$SHAREGPT_PATH"

# torchrun/sonnet all BS
submit_torchrun "sonnet" 4  "$SONNET_PATH"
submit_torchrun "sonnet" 8  "$SONNET_PATH"
submit_torchrun "sonnet" 16 "$SONNET_PATH"

echo "=== Done. Check with: squeue -u $USER ==="
