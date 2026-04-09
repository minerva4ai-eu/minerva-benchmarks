#!/bin/bash
#
# Debug/dry-run script: submits a single FSDP job with only 5 training steps
# using boost_qos_dbg (high priority, 30 min limit, no chained dependencies).
#
# Usage:
#   cd /leonardo_work/cin_staff/dgentile/minerva-benchmarks/training/training_MN5
#   bash debug_run.sh
#

set -a
source .env-leonardo
set +a

CURRENT_DIR=$(pwd)

# --- Config (mirrors BS=4 entry from model_parallelism_config.json) ---
NODES=4
GPUS_PER_NODE=4
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
TOTAL_CPUS=$((GPUS_PER_NODE * CPUS_PER_GPU))

MODEL="Llama-3.1-8B-Instruct"
DATASET="alpaca"
PARALLELISM="fsdp"
PRECISION="bf16"
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_MODEL_LENGTH=2048
LR=2e-5
STEPS=5
EPOCHS=""

MODEL_PATH="/leonardo_work/cin_staff/dgentile/models_registry/${MODEL}"
DATASET_PATH="/leonardo_work/cin_staff/dgentile/minerva-benchmarks/training/training_MN5/datasets/alpaca-cleaned/alpaca_data_cleaned.json"

# --- Launch folder (separate from production results) ---
LAUNCH_FOLDER="${CURRENT_DIR}/results/accelerate/${DATASET}/${MODEL}/Nodes_${NODES}-GPUs_${TOTAL_GPUS}-Parallelism_${PARALLELISM}-Precision_${PRECISION}-BS_${BATCH_SIZE}-GAS_${GRAD_ACCUM}-MaxModelLength_${MAX_MODEL_LENGTH}/debug"
mkdir -p "$LAUNCH_FOLDER"

cp scripts/accelerate-common/run-fsdp.sh "$LAUNCH_FOLDER"
cp scripts/accelerate-common/finetune-fsdp.py "$LAUNCH_FOLDER"
cp scripts/accelerate-common/custom_train.py "$LAUNCH_FOLDER"
cp scripts/accelerate-common/gpu_monitor.py "$LAUNCH_FOLDER"
cp scripts/gpu_plots.py "$LAUNCH_FOLDER"
cp scripts/accelerate-common/utils.py "$LAUNCH_FOLDER"
cp scripts/activate-env-per-supercomputer.sh "$LAUNCH_FOLDER"
cp scripts/activate-env-variables-per-supercomputer.sh "$LAUNCH_FOLDER"

cd "$LAUNCH_FOLDER" || exit 1

export CURRENT_DIR NODES GPUS_PER_NODE TOTAL_CPUS MAX_MODEL_LENGTH LR EPOCHS STEPS
export MODEL_PATH DATASET_PATH
export PARALLELISM PRECISION BATCH_SIZE GRAD_ACCUM
export MODULES MACHINE_TYPE="cuda"
export MACHINE="leonardo"
export ENVIRONMENT_FINETUNING

JOB_ID=$(sbatch --parsable \
    --chdir="$(pwd)" \
    --nodes=$NODES \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-task=$TOTAL_CPUS \
    --tasks-per-node=1 \
    --time=00:20:00 \
    --output=run-%j.out \
    --error=run-%j.err \
    -A $ACCOUNT \
    -p $PARTITION_NAME \
    -q $QOS \
    run-fsdp.sh "$LAUNCH_FOLDER" "$DATASET" "$DATASET_PATH")

echo "Debug job submitted: $JOB_ID"
echo "Output will be in: $LAUNCH_FOLDER/run-${JOB_ID}.out"
echo "Monitor with: squeue -u \$USER -j $JOB_ID"
