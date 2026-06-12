#!/bin/bash

#SBATCH --job-name=DEEPSPEED_DYNAMIC

##################################################
###           Load HPC modules                 ###
##################################################
eval "$LOAD_MODULES"
##################################################

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

OUTPUT_DIR="${LAUNCH_FOLDER}/output"
mkdir -p $OUTPUT_DIR

echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}, ZERO_STAGE: {$ZERO_STAGE}, HPZ_PARTITION_SIZE: {$HPZ_PARTITION_SIZE}"
echo "MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION}, BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"

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

##################################################
###        Generate DeepSpeed Hostfile         ###
##################################################
HOSTFILE="${LAUNCH_FOLDER}/hostfile"
rm -f "$HOSTFILE"
for node in $(scontrol show hostnames ${SLURM_NODELIST}); do
    echo "$node slots=$GPUS_PER_NODE" >> "$HOSTFILE"
done
echo "Generated hostfile:"
cat "$HOSTFILE"

##################################################
###        Update DeepSpeed Config             ###
##################################################
deepspeed_config_path="configs/${ZERO_STAGE}.json"

if exists_in_list "${STAGES_WITH_HPZ[*]}" " " "$ZERO_STAGE"; then
    HPZ_PARTITION_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
    sed -i "s/\"zero_hpz_partition_size\": \"{{HPZ_PARTITION_SIZE}}\"/\"zero_hpz_partition_size\": $HPZ_PARTITION_SIZE/g" "$deepspeed_config_path"
    echo "Using hpZ partition size: $HPZ_PARTITION_SIZE"
fi

##################################################
###        Build Singularity Prefix            ###
##################################################
singularity_prefix="singularity exec \
    $SINGULARITY_ARGS \
    $SINGULARITY_BINDS \
    --home $LAUNCH_FOLDER \
    $SINGULARITY_CONTAINER"
echo "Singularity prefix: $singularity_prefix"

##################################################
###        GPU Monitor Command                 ###
##################################################
gpu_plots_monitor_command="$singularity_prefix python -m gpu_plots"

##################################################
###        Training Command                    ###
##################################################
# DeepSpeed launcher reads the hostfile and SSHes into each node.
# It must be launched ONCE from the head node only (not via srun).
train_command="$singularity_prefix \
    deepspeed \
    --hostfile $HOSTFILE \
    --master_addr $head_node_ip \
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
      --deepspeed_config_file $deepspeed_config_path \
      --logging_steps 1"

##################################################
###        Launch (head node only)             ###
##################################################
# Start GPU monitor on all nodes via srun in background
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --export=ALL \
    bash -c "$gpu_plots_monitor_command" &
monitor_pid=$!

sleep 5

# DeepSpeed launcher handles multi-node itself via hostfile — run once, not via srun
$train_command

kill -SIGTERM "$monitor_pid"
wait "$monitor_pid"

echo "DeepSpeed Job Completed."