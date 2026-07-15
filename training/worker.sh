#!/bin/bash

echo -e "\n\nStarting worker: ..."

# Environment Variables
echo Config in worker script:
echo "  MODEL: $MODEL (path: $MODEL_PATH)" \
"  FRAMEWORK: $FRAMEWORK" \
"  PARALLELISM: $PARALLELISM" \
"  NUMBER_OF_NODES: $NUMBER_OF_NODES" \
"  DATASET: $DATASET (path: $DATASET_PATH)" \
"  BATCH_SIZE: $BATCH_SIZE" \
"  GRAD_ACCUM: $GRAD_ACCUM" \
"  ENABLE_COMPILE: $ENABLE_COMPILE" \
"  PRECISION: $PRECISION" \
"  TRIAL: $TRIAL" \
"  LR: $LR" \
"  EPOCHS: $EPOCHS" \
"  STEPS: $STEPS" \
"  USECPU: $USECPU" \
"  SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE" \
"  SLURM_STEP_NUM_NODES: $SLURM_STEP_NUM_NODES" \
"  SLURM_STEP_NUM_NODES: $SLURM_STEP_NUM_NODES" \
"  SLURM_NODEID: $SLURM_NODEID" \
"  RESULTSDIR: $RESULTSDIR"

# pwd
# env | grep SLURM

if [[ "$FRAMEWORK" == "torchrun" ]]; then
    echo "Using torchrun for distributed training"
    scripts/torchrun-common/run-any.sh > $RESULTSDIR/step-$SLURM_JOB_ID.$SLURM_STEP_ID-rank$SLURM_NODEID.log 2>&1
elif [[ "$FRAMEWORK" == "accelerate" ]]; then
    echo "Using accelerate for distributed training"
    scripts/accelerate-common/run-any.sh > $RESULTSDIR/step-$SLURM_JOB_ID.$SLURM_STEP_ID-rank$SLURM_NODEID.log 2>&1
elif [[ "$FRAMEWORK" == "deepspeed" ]]; then
    echo "Using deepspeed for distributed training"
    # Add deepspeed execution here
    echo "Deepspeed execution would be here"
elif [[ "$FRAMEWORK" == "deepspeed-accelerate" ]]; then
    echo "Using deepspeed-accelerate for distributed training"
    # Add deepspeed-accelerate execution here
    echo "Deepspeed execution would be here"
else
    echo "Unknown framework: $FRAMEWORK"
    exit 1
fi

echo "Ending worker_fn: `date`"
echo -e "Finished work: \n\n"

# original YAML: ddp--bs8-grad_accum8-compileFalse-precbf16-steps2 
# output from multple configurations will be stored in same directory => name output/log files with stepid