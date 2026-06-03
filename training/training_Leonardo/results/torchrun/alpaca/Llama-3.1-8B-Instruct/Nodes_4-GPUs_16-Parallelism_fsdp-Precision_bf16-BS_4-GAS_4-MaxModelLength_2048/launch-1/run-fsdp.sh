#!/bin/bash

#SBATCH --job-name=PYTORCH_DYNAMIC
#SBATCH --time=24:00:00


##################################################
###           Activate Environment             ###
##################################################
# Activate virtual environment using conda
source activate-env-per-supercomputer.sh $ENVIRONMENT_FINETUNING
# module load $MODULES
# source activate $ENVIRONMENT_FINETUNING
# export PATH=$ENVIRONMENT_FINETUNING/bin:$PATH
# which python

##################################################


##################################################
###        Environment Variables Setup         ###
##################################################

# Get Arguments
LAUNCH_FOLDER=$1
DATASET=$2
DATASET_PATH=$3
OUTPUT_DIR="${LAUNCH_FOLDER}/output"
mkdir -p $OUTPUT_DIR

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}"
echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"


# Export environment variables
# export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=none
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True
###################################################

##################################################
###             Torchrun Setup                 ###
##################################################
gpu_plots_monitor_command="python -m gpu_plots"


# Torchrun args
JOB_ID=${SLURM_JOB_ID}
NNODES=${SLURM_NNODES}
NPROC_PER_NODE=$GPUS_PER_NODE
MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -n 1)
MASTER_PORT=29500


# Launch Run
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --export=ALL bash -c "
    # Start monitoring in background
    $gpu_plots_monitor_command &
    monitor_pid=\$!

    # Optional: give the monitor time to initialize
    sleep 5

    # Run training in foreground (this blocks until done)
    torchrun \
      --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
      --rdzv_id $JOB_ID --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
      finetune-fsdp.py \
        --minerva_dir "${CURRENT_DIR}" \
        --model "${MODEL_PATH}" \
        --data '${DATASET_PATH}' \
        --output_dir "${OUTPUT_DIR}/$SLURM_JOB_ID" \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_MODEL_LENGTH \
        ${EPOCHS:+--epochs "$EPOCHS"} \
        ${STEPS:+--max_steps "$STEPS"} \
        --precision $PRECISION \
        --lr $LR \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --dataloader_num_workers 2 \
        --dataset $DATASET

    kill -SIGTERM \"\$monitor_pid\"

    # Wait for the monitor to clean up and exit
    wait \"\$monitor_pid\"
"

# srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --export=ALL \
#   torchrun \
#     --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
#     --rdzv_id $JOB_ID --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
#     finetune-fsdp.py \
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
#       --dataloader_num_workers 2 \
#       --dataset $DATASET


echo "FSDP Job Completed."

