#!/bin/bash


##################################################
###            Setup Environment               ###
##################################################
module purge
if [ ! -z "$LOAD_MODULES" ]; then
    eval "$LOAD_MODULES"
fi 
source shared/runtime_environment.sh
training_activate_runtime_environment

echo yaml_path=$1

####################################################

##################################################
###             Torchrun Setup                 ###
##################################################
gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.gpu_plots"

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

if [[ $ENABLE_COMPILE == "True" || $ENABLE_COMPILE == "true" ]]; then
    train_command="$train_command --enable_compile"
fi

# Launch Run
# Start monitoring in background
$gpu_plots_monitor_command &
monitor_pid=\$!

# Optional: give the monitor time to initialize
sleep 5
$train_command

kill -SIGTERM \"\$monitor_pid\"

# Wait for the monitor to clean up and exit
wait \"\$monitor_pid\"

echo "DDP Job Completed."


#!/bin/bash

#SBATCH --job-name=PYTORCH_DYNAMIC
#SBATCH --time=1:00:00


##################################################
###           Activate Environment             ###
##################################################
# Activate virtual environment using conda
# source activate-env-per-supercomputer.sh $ENVIRONMENT_FINETUNING
eval "$LOAD_MODULES"
# source activate $ENVIRONMENT_FINETUNING
# export PATH=$ENVIRONMENT_FINETUNING/bin:$PATH
# which python

##################################################


##################################################
###        Environment Variables Setup         ###
##################################################

# Get Arguments
OUTPUT_DIR="${LAUNCH_FOLDER}/output"
# Deprecated script arguments - provided by launch environment
#LAUNCH_FOLDER=$1
#DATASET=$2
#DATASET_PATH=$3
#TRAIN_SCRIPT=$4
mkdir -p $OUTPUT_DIR

# Print Arguments Received
echo "LAUNCH_FOLDER: {$LAUNCH_FOLDER}, DATASET: {$DATASET}, DATASET_PATH: {$DATASET_PATH}"
echo "LAUNCH FOLDER CONTENTS: MAX_MODEL_LENGTH: ${MAX_MODEL_LENGTH}, GPUS_PER_NODE: {$GPUS_PER_NODE}, MODEL_PATH: {$MODEL_PATH}, PARALLELISM: {$PARALLELISM}, PRECISION: {$PRECISION} BATCH_SIZE: {$BATCH_SIZE}, GRAD_ACCUM: {$GRAD_ACCUM}"


# Export environment variables
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=none
####################################################

##################################################
###             Torchrun Setup                 ###
##################################################
runtime_prefix="$(training_build_runtime_prefix)"
echo "runtime prefix $runtime_prefix"
gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m shared.gpu_plots"


# Torchrun args
export JOB_ID=${SLURM_JOB_ID}
export NNODES=${SLURM_NNODES}
export NPROC_PER_NODE=$GPUS_PER_NODE
export NUM_PROCS=$((NNODES * NPROC_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID

singularity_prefix="singularity exec \
    $SINGULARITY_ARGS \
    $SINGULARITY_BINDS \
    --home $LAUNCH_FOLDER \
    $SINGULARITY_CONTAINER"
echo "singularity prefix $singularity_prefix"

gpu_plots_monitor_command="$singularity_prefix  python -m gpu_plots"
echo "PATH to Singularity Container: $SINGULARITY_CONTAINER"

echo "NODE_RANK: {$NODE_RANK}"
echo "NNODES: {$NNODES}"
echo "NUM_PROCS: {$NUM_PROCS}"
echo "MASTER_ADDR: {$MASTER_ADDR}"
echo "MASTER_PORT: {$MASTER_PORT}"
echo "train_command: {$train_command}"

source activate-env-variables-per-supercomputer.sh
# Launch Run
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --export=ALL bash -c "
    # Start monitoring in background
    $singularity_prefix $gpu_plots_monitor_command &
    monitor_pid=\$!

    # Optional: give the monitor time to initialize
    sleep 5

    $singularity_prefix torchrun \
      --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
      --rdzv_id $JOB_ID --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
      $TRAIN_SCRIPT \
        --model "${MODEL_PATH}" \
        --data '${DATASET_PATH}' \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_MODEL_LENGTH \
        ${EPOCHS:+--epochs "$EPOCHS"} \
        ${STEPS:+--max_steps "$STEPS"} \
        --precision $PRECISION \
        --lr $LR \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --dataloader_num_workers 4 \
        --dataset '$DATASET'"

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

