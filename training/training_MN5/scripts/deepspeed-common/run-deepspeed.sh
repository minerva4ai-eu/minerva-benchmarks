#!/bin/bash

#SBATCH --job-name=DEEPSPEED_DYNAMIC
#SBATCH --time=1:00:00

set -e
set -o pipefail

##################################################
###           Activate Environment             ###
##################################################
# Activate virtual environment using conda
module load $MODULES
source activate $ENVIRONMENT_FINETUNING
export PATH=$ENVIRONMENT_FINETUNING/bin:$PATH
which python

##################################################


# --- Pre-stage model to node-local scratch to avoid GPFS I/O contention ---
#if [[ -n "$TMPDIR" ]]; then
#    echo "Node-local scratch detected at $TMPDIR"
#else
#    echo "Error: No node-local scratch directory found. Please ensure \$TMPDIR is set."
#    exit 1
#fi
#LOCAL_MODEL_PATH="${TMPDIR}/model_$(basename $MODEL_PATH)"
#
#echo "Pre-staging model from $MODEL_PATH to $LOCAL_MODEL_PATH ..."
#srun --ntasks="$SLURM_NNODES" \
#     --ntasks-per-node=1 \
#     --export=ALL \
#     bash -c "
#         if [ ! -d '$LOCAL_MODEL_PATH' ]; then
#             cp -r '$MODEL_PATH' '$LOCAL_MODEL_PATH'
#             echo \"Node \$(hostname): model staged to $LOCAL_MODEL_PATH\"
#         else
#             echo \"Node \$(hostname): model already cached at $LOCAL_MODEL_PATH\"
#         fi
#     "
#echo "Model pre-staging complete."
#
## Override MODEL_PATH to point to local copy
#MODEL_PATH="$LOCAL_MODEL_PATH"

##################################################
###        Environment Variables Setup         ###
##################################################

function exists_in_list() {
    LIST=$1
    DELIMITER=$2
    VALUE=$3
    LIST_WHITESPACES=$(echo "$LIST" | tr "$DELIMITER" ' ')
    
    for x in $LIST_WHITESPACES; do
        if [ "$x" = "$VALUE" ]; then
            return 0
        fi
    done
    return 1
}

# Get Arguments
LAUNCH_FOLDER=$1
#DATASET=$2
#DATASET_PATH=$3
#ZERO_STAGE=$4
## Validate ZERO_STAGE
#PERMITTED_ZERO_STAGES=("zero1" "zero2" "zero3" "zero3-offload")
#STAGES_WITH_HPZ=("zero2" "zero3" "zero3-offload")
#
#if [[ -z "$ZERO_STAGE" ]]; then
#    echo "Error: ZERO_STAGE is required. Please provide one of: ${PERMITTED_ZERO_STAGES[*]}"
#    exit 1
#fi
#
#if ! exists_in_list "${PERMITTED_ZERO_STAGES[*]}" " " "$ZERO_STAGE"; then
#    echo "Error: ZERO_STAGE must be one of: ${PERMITTED_ZERO_STAGES[*]}"
#    echo "Received: $ZERO_STAGE"
#    exit 1
#fi

OUTPUT_DIR="${LAUNCH_FOLDER}/output"
mkdir -p $OUTPUT_DIR

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}, ZERO_STAGE: {$ZERO_STAGE}, HPZ_PARTITION_SIZE: {$HPZ_PARTITION_SIZE}"
echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"


# Export environment variables
# export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=none
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True

export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=$GPUS_PER_NODE
export NUM_PROCS=$((NNODES * NPROC_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NODE_RANK=$SLURM_PROCID

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname --ip-address)
export NCCL_DEBUG=INFO


deepspeed_config_path="configs/${ZERO_STAGE}.json"
accelerate_config_path="configs/accelerate_config.yaml"

# Update Accelerate config placeholders
sed -i "s/{{MASTER_IP}}/$head_node_ip/g" "$accelerate_config_path"
sed -i "s/{{NUM_NODES}}/$NNODES/g" "$accelerate_config_path"
sed -i "s/{{NUM_GPUS}}/$NUM_PROCS/g" "$accelerate_config_path"
sed -i "s/machine_rank: 0/machine_rank: $NODE_RANK/g" "$accelerate_config_path"

# Update DeepSpeed config path in Accelerate config
sed -i "s|{{path to ds_config.json}}|$deepspeed_config_path|g" "$accelerate_config_path"

# Update hpZ partition size for parallelism of stage2 and stage3 configurations only, as stage1 does not use hpZ
if exists_in_list "${STAGES_WITH_HPZ[*]}" " " "$ZERO_STAGE"; then
    HPZ_PARTITION_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
    #HPZ_PARTITION_SIZE=4
    sed -i "s/\"zero_hpz_partition_size\": \"{{HPZ_PARTITION_SIZE}}\"/\"zero_hpz_partition_size\": $HPZ_PARTITION_SIZE/g" "$deepspeed_config_path"
    echo "Using hpZ partition size: $HPZ_PARTITION_SIZE"
fi

gpu_plots_monitor_command="python -m gpu_plots"

SCALED_LR=$(python3 -c "import math; print(${LR} * math.sqrt(${NUM_PROCS} ))")

train_command="accelerate launch \
    --config_file $accelerate_config_path \
    --machine_rank $SLURM_NODEID \
    --rdzv_backend c10d \
    $TRAIN_SCRIPT \
      --model $MODEL_PATH \
      --data $DATASET_PATH \
      --output_dir $OUTPUT_DIR \
      --max_length $MAX_MODEL_LENGTH \
      --batch_size $BATCH_SIZE \
      ${EPOCHS:+--epochs "$EPOCHS"} \
      ${STEPS:+--max_steps "$STEPS"} \
      --precision $PRECISION \
      --lr $SCALED_LR \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --dataloader_num_workers 32 \
      --dataset '$DATASET' \
      --warmup_ratio 0.1 \
      --deepspeed_config_file  $deepspeed_config_path \
      --logging_steps 1 \
      --gradient_checkpointing"
train_command="deepspeed \
    --num_nodes $SLURM_NNODES \
    --num_gpus $GPU_NODE \
    --node_rank $SLURM_NODEID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $TRAIN_SCRIPT \
      --model $MODEL_PATH \
      --data $DATASET_PATH \
      --output_dir $OUTPUT_DIR \
      --max_length $MAX_MODEL_LENGTH \
      --batch_size $BATCH_SIZE \
      ${EPOCHS:+--epochs "$EPOCHS"} \
      ${STEPS:+--max_steps "$STEPS"} \
      --precision $PRECISION \
      --lr $SCALED_LR \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --dataloader_num_workers 8 \
      --dataset '$DATASET' \
      --warmup_ratio 0.1 \
      --deepspeed_config_file  $deepspeed_config_path \
      --logging_steps 1 \
      --gradient_checkpointing"
# Launch Run
srun -l --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --export=ALL bash -c "
    # Start monitoring in background
    $gpu_plots_monitor_command &
    monitor_pid=\$!

    # Optional: give the monitor time to initialize
    sleep 5

    # Run training in foreground (this blocks until done)
    $train_command

    kill -SIGTERM \"\$monitor_pid\"

    # Wait for the monitor to clean up and exit
    wait \"\$monitor_pid\"
"

echo "Deepspeed Job Completed."

