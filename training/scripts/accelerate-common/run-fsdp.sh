#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################
module purge
if [ ! -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi
source scripts/shared/runtime_environment.sh
training_activate_runtime_environment

echo "EXECUTION_MODE: $EXECUTION_MODE"
runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix: $runtime_prefix"
echo yaml_path=$1

MASTER_PORT=29500
# export MASTER_ADDR=$HEAD_NODE
NUM_PROCS=$(($SLURM_STEP_NUM_NODES * SLURM_GPUS_ON_NODE))

gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.gpu_plots"

# accelerate_config_path="scripts/accelerate-common/accelerate_config.yaml"


# Export environment variables
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}


# Torchrun args
export JOB_ID=${SLURM_JOB_ID}
export NNODES=${SLURM_NNODES}
export NPROC_PER_NODE=$GPUS_PER_NODE
export NUM_PROCS=$((NNODES * NPROC_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID

runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix $runtime_prefix"

gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m shared.gpu_plots"


head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname --ip-address)
accelerate_config_path="accelerate_config.yaml"

# Update Accelerate config placeholders
sed -i "s/{{MASTER_IP}}/$head_node_ip/g" "$accelerate_config_path"
sed -i "s/{{NUM_NODES}}/$NNODES/g" "$accelerate_config_path"
sed -i "s/{{NUM_GPUS}}/$NUM_PROCS/g" "$accelerate_config_path"
sed -i "s/machine_rank: 0/machine_rank: $NODE_RANK/g" "$accelerate_config_path"

echo "EXECUTION_MODE: $EXECUTION_MODE"

train_command="${runtime_prefix:+$runtime_prefix} accelerate launch \
    --multi-gpu \
    --machine_rank $SLURM_NODEID \
    --rdzv_backend c10d \
    --main_process_ip $HEAD_NODE \
    --main_process_port $MASTER_PORT \
    --num_processes $NUM_PROCS \
    --num_machines $NNODES \
       $TRAIN_SCRIPT \
        --model $MODEL_PATH \
        --data '$DATASET_PATH' \
        --output_dir $OUTPUT_DIR/$SLURM_JOB_ID \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_MODEL_LENGTH \
        ${EPOCHS:+--epochs $EPOCHS} \
        ${STEPS:+--max_steps $STEPS} \
        --precision $PRECISION \
        --lr $LR \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --dataloader_num_workers 4 \
        --dataset $DATASET \
        --max_comm_comp_overlap"

prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m shared.prepare \
        --model $MODEL_PATH \
        --data $DATASET_PATH \
        --dataset $DATASET \
        --output_dir $OUTPUT_DIR/$SLURM_JOB_ID \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_MODEL_LENGTH "

echo "ENABLE_COMPILE: $ENABLE_COMPILE"
if [[ $ENABLE_COMPILE == "True" || $ENABLE_COMPILE == "true" ]]; then
    echo "Compile enabled!"
    train_command="$train_command --enable_compile"
fi

prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.prepare --yaml $1"

echo "NODE_RANK: {$NODE_RANK}"
echo "NNODES: {$NNODES}"
echo "NUM_PROCS: {$NUM_PROCS}"
echo "MASTER_ADDR: {$MASTER_ADDR}"
echo "MASTER_PORT: {$MASTER_PORT}"
echo "train_command: {$train_command}"
echo "prepare_train_command: {$prepare_train_command}"


echo "######################################"
echo "#       Running preparation stage    #"
echo "######################################"
    
$prepare_train_command

echo "######################################"
echo "#     Running Accelerate-FSDP train  #"
echo "######################################"


    # Optional: give the monitor time to initialize
    sleep 5
    export FSDP_CPU_RAM_EFFICIENT_LOADING=True

    # Run training in foreground (this blocks until done)
    $train_command



echo "FSDP Job Completed."

