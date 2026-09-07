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


###################################################

##################################################
###             Torchrun Setup                 ###
##################################################

export MASTER_PORT=29500

gpu_plots_monitor_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.gpu_plots"


train_command="${runtime_prefix:+$runtime_prefix} torchrun \
    --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
    --rdzv_id $JOB_ID --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    $TRAIN_SCRIPT --yaml $1 --max_comm_comp_overlap"


prepare_train_command="${runtime_prefix:+$runtime_prefix} python -m scripts.shared.prepare \
        --model $MODEL_PATH \
        --data $DATASET_PATH \
        --dataset $DATASET \
        --output_dir $OUTPUT_DIR/$SLURM_JOB_ID \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_MODEL_LENGTH "


echo "######################################"
echo "#       Running preparation stage    #"
echo "######################################"
    
$prepare_train_command

echo "######################################"
echo "#     Running  Torchrun-FSDP train   #"
echo "######################################"

# Launch Run
# Start monitoring in background
$gpu_plots_monitor_command &
monitor_pid=$!

# Optional: give the monitor time to initialize
sleep 5

# Run training in foreground (this blocks until done)
$train_command

kill -SIGTERM "$monitor_pid"

# Wait for the monitor to clean up and exit
wait "$monitor_pid"

echo "FSDP Job Completed."

