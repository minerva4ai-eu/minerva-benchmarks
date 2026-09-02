#!/bin/bash

#SBATCH --job-name=PYTORCH_DYNAMIC


##################################################
###            Setup Environment               ###
##################################################
module purge
if [ ! -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi
source shared/runtime_environment.sh
training_activate_runtime_environment

# Get Arguments
OUTPUT_DIR="${LAUNCH_FOLDER}/output"
mkdir -p $OUTPUT_DIR

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}"
echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"


# Export environment variables
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
###################################################

##################################################
###             Torchrun Setup                 ###
##################################################
runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix $runtime_prefix"


# Torchrun args
export JOB_ID=${SLURM_JOB_ID}
export NNODES=${SLURM_NNODES}
export NPROC_PER_NODE=$GPUS_PER_NODE
export NUM_PROCS=$((NNODES * NPROC_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID

gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m shared.gpu_plots"


train_command="${runtime_prefix:+$runtime_prefix} torchrun \
    --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
    --rdzv_id $JOB_ID --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    $TRAIN_SCRIPT \
      --model $MODEL_PATH \
      --data '$DATASET_PATH' \
      --output_dir $OUTPUT_DIR/$SLURM_JOB_ID \
      --batch_size $BATCH_SIZE \
      --max_length $MAX_MODEL_LENGTH \
      ${EPOCHS:+--epochs "$EPOCHS"} \
      ${STEPS:+--max_steps "$STEPS"} \
      --precision $PRECISION \
      --lr $LR \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --dataloader_num_workers 4 \
      --dataset $DATASET \
      --max_comm_comp_overlap "


prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m shared.prepare \
        --model $MODEL_PATH \
        --data $DATASET_PATH \
        --dataset $DATASET \
        --output_dir $OUTPUT_DIR/$SLURM_JOB_ID \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_MODEL_LENGTH "

echo "ENABLE_COMPILE: $ENABLE_COMPILE"
if [[ $ENABLE_COMPILE == "True" || $ENABLE_COMPILE == "true" ]]; then
    train_command="$train_command --enable_compile"
fi

echo "######################################"
echo "#       Running preparation stage    #"
echo "######################################"
    
srun --nodes=1 --ntasks=1 --export=ALL $prepare_train_command

echo "######################################"
echo "#     Running  Torchrun-FSDP train    #"
echo "######################################"

# Launch Run
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --export=ALL bash -c "
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

echo "FSDP Job Completed."

