#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################
module purge
if [ ! -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi

echo yaml_path=$1

source scripts/shared/runtime_environment.sh
training_activate_runtime_environment

echo "EXECUTION_MODE: $EXECUTION_MODE"
runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix: $runtime_prefix"

##################################################
###             Torchrun Setup                 ###
##################################################
gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.gpu_plots"

export MASTER_PORT=29500

# Adjust precision for CPU runs - bf16 is not supported on CPU
adjusted_precision=$PRECISION
if [ "$GPUS_PER_NODE" -eq "0" ] && [ "$PRECISION" = "bf16" ]; then
    adjusted_precision="fp32"
    echo "⚠️  WARNING: bf16 not supported on CPU, switching to fp32"
fi

train_command="${runtime_prefix:+$runtime_prefix} torchrun \
      --nnodes $SLURM_STEP_NUM_NODES --nproc_per_node $SLURM_GPUS_ON_NODE \
      --rdzv_id $SLURM_JOB_ID.$SLURM_STEP_ID --rdzv_backend c10d --rdzv_endpoint ${HEAD_NODE}:${MASTER_PORT} \
      $TRAIN_SCRIPT --yaml $1"

prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.prepare --yaml $1"

echo "######################################"
echo "#       Running preparation stage    #"
echo "######################################"
    
$prepare_train_command

echo "######################################"
echo "#     Running  Torchrun-DDP train    #"
echo "######################################"

# Launch Run
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --export=ALL bash -c "
    # Start monitoring in background
    $gpu_plots_monitor_command &
    monitor_pid=\$!

    # Optional: give the monitor time to initialize
    sleep 5
    $train_command
    
    kill -SIGTERM \"\$monitor_pid\"

    # Wait for the monitor to clean up and exit
    wait \"\$monitor_pid\"
"

# # Launch Run
# srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --export=ALL \
#   torchrun \
#     --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
#     --rdzv_id $JOB_ID --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
#     finetune-ddp.py \
#       --minerva_dir "${CURRENT_DIR}" \
#       --model "${MODEL_PATH}" \
#       --data "${DATASET_PATH}" \
#       --output_dir "${OUTPUT_DIR}" \
#       --batch_size $BATCH_SIZE \
#       --max_length $MAX_MODEL_LENGTH \
#       ${EPOCHS:+--epochs "$EPOCHS"} \
#       ${STEPS:+--max_steps "$STEPS"} \
#       --precision $PRECISION \
#       --lr $LR \
#       --gradient_accumulation_steps $GRAD_ACCUM \
#       --dataloader_num_workers 8 \
#       --dataset $DATASET


echo "DDP Job Completed."

