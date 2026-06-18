#!/bin/bash

#SBATCH --job-name=DEEPSPEED_DYNAMIC


##################################################
###           Environment Setup                ###
##################################################
if [ -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi

source shared/runtime_environment.sh
training_activate_runtime_environment

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
OUTPUT_DIR="${LAUNCH_FOLDER}/output"
mkdir -p $OUTPUT_DIR

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}, ZERO_STAGE: {$ZERO_STAGE}, HPZ_PARTITION_SIZE: {$HPZ_PARTITION_SIZE}"
echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"


# Export environment variables
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=none
export JOB_ID=${SLURM_JOB_ID}
export NNODES=${SLURM_NNODES}
export NPROC_PER_NODE=$GPUS_PER_NODE
export NUM_PROCS=$((NNODES * NPROC_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname --ip-address)

# ------------------------------------------------------------------
# Generate DeepSpeed hostfile from SLURM node list
# DeepSpeed requires a hostfile when num_nodes > 1
# ------------------------------------------------------------------
HOSTFILE="$OUTPUT_DIR/hostfile"
scontrol show hostnames "${SLURM_NODELIST}" | while read -r hostname; do
    echo "${hostname} slots=${GPU_NODE}" >> "$HOSTFILE"
done
echo "Generated DeepSpeed hostfile at: $HOSTFILE"
cat "$HOSTFILE"

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

runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix $runtime_prefix"

gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix }python -m gpu_plots"

#train_command="accelerate launch \
#    --config_file $accelerate_config_path \
#    --machine_rank $SLURM_NODEID \
#    --rdzv_backend c10d \
#    $TRAIN_SCRIPT \
#      --model $MODEL_PATH \
#      --data $DATASET_PATH \
#      --output_dir $OUTPUT_DIR \
#      --max_length $MAX_MODEL_LENGTH \
#      --batch_size $BATCH_SIZE \
#      ${EPOCHS:+--epochs "$EPOCHS"} \
#      ${STEPS:+--max_steps "$STEPS"} \
#      --precision $PRECISION \
#      --lr $LR \
#      --gradient_accumulation_steps $GRAD_ACCUM \
#      --dataloader_num_workers 8 \
#      --dataset '$DATASET' \
#      --warmup_ratio 0.1 \
#      --deepspeed_config_file  $deepspeed_config_path \
#      --logging_steps 1 "
train_command="${runtime_prefix:+$runtime_prefix }deepspeed \
    --hostfile $HOSTFILE \
    --num_nodes $SLURM_NNODES \
    --num_gpus $GPU_NODE \
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
      --lr $LR \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --dataloader_num_workers 8 \
      --dataset '$DATASET' \
      --warmup_ratio 0.1 \
      --deepspeed_config_file  $deepspeed_config_path \
      --logging_steps 1 "
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

