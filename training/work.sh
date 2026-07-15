#!/bin/bash

echo -e "\n\nStarting worker..."

##################################################
###             Environment Setup              ###
##################################################

# Get Arguments
export MODEL=$1
export FRAMEWORK=$2
export PARALLELISM=$3
export NUMBER_OF_NODES=$4
export DATASET=$5
export BATCH_SIZE=$6
export GRAD_ACCUM=$7
export ENABLE_COMPILE=$8
export PRECISION=$9
export LR=${10}
export TRIAL=${11}

# Rules Check - First Thing
if check_config; then
    echo Invalid configuration: $MODEL / $FRAMEWORK / $PARALLELISM / $NUMBER_OF_NODES / $DATASET / $BATCH_SIZE / $GRAD_ACCUM / $ENABLE_COMPILE / $PRECISION / $LR / $TRIAL / $SLURM_JOB_NUM_NODES
    exit 0
fi

eval "$MINERVAPATHS"
export MODEL_PATH=${paths[$MODEL]}
export DATASET_PATH=${paths[$DATASET]}

export USECPU=1
NGPU=0
NCPU=$SLURM_CPUS_PER_TASK
if [[ $SLURM_GPUS_ON_NODE -gt 0 ]]; then
    USECPU=0
    if [[ $PARALLELISM == "none" ]]; then
        NGPU=1
    else
        NGPU=$SLURM_GPUS_ON_NODE
    fi
fi

# export RESULTSDIR=$OUT_DIR/bsc-mn5-acc/$SLURM_JOB_ID/$MODEL/$FRAMEWORK/$PARALLELISM/$DATASET/nodes-$NUMBER_OF_NODES/trial-$TRIAL/output
export RESULTSDIR=$OUT_DIR/$SLURM_CLUSTER_NAME-$SLURM_JOB_PARTITION/$SLURM_JOB_ID/$MODEL/$FRAMEWORK/$PARALLELISM/$DATASET/nodes-$NUMBER_OF_NODES/run_id-$TRIAL/launch-1/output
if [[ ! -d $RESULTSDIR ]]; then
    mkdir -p $RESULTSDIR
fi

echo Config in work script:
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
"  NGPU: $NGPU" \
"  NCPU: $NCPU" \
"  OUT_DIR: $OUT_DIR" \
"  CONFIG_ENV: $CONFIG_ENV"  \
"  CONTAINER_ARGS: $CONTAINER_ARGS" \
"  CONTAINER_PATH: $CONTAINER_PATH"\
"  RESULTSDIR: $RESULTSDIR" \
"  SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES" \
"  SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE" \
"  SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK" \
"  SLURM_NODEID: $SLURM_NODEID" \
"  SLURM_PROCID: $SLURM_PROCID"

runner_fn () {

    echo -e "\n\nStarting runner: ..."

    # Check Environment Variables
    echo Config in runner function:
    echo "  MODEL: $MODEL (path: $MODEL_PATH)" \
    "  FRAMEWORK: $FRAMEWORK" \
    "  PARALLELISM: $PARALLELISM" \
    "  NUMBER_OF_NODES: $NUMBER_OF_NODES"  \
    "  SLURM_STEP_NUM_NODES: $SLURM_STEP_NUM_NODES"  \
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
    "  RESULTSDIR: $RESULTSDIR" \
    "  CONFIG_ENV: $CONFIG_ENV" \
    "  CONTAINER_ARGS: $CONTAINER_ARGS" \
    "  CONTAINER_PATH: $CONTAINER_PATH" \
    "  SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE" \
    "  SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK" \
    "  SLURM_NODEID: $SLURM_NODEID" \
    "  SLURM_PROCID: $SLURM_PROCID"

    # env | grep SLURM

    # For python scripts which read with os.environment.get
    export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
    export GPU_NODE=$SLURM_GPUS_ON_NODE

    # No slurm commands inside the container (environment variables yes)
    export HEAD_NODE=$(scontrol show hostnames ${SLURM_STEP_NODELIST} | head -n 1)

    if [[ $CONFIG_ENV == "singularity" ]]; then
        echo Using container: $CONTAINER_PATH
        CWD=`pwd`
        singularity exec $CONTAINER_ARGS $CONTAINER_PATH $CWD/worker.sh >> $RESULTSDIR/run-$SLURM_JOB_ID.$SLURM_STEP_ID.log 2>&1
    else
        ./worker.sh >> $RESULTSDIR/run-$SLURM_JOB_ID.$SLURM_STEP_ID.log 2>&1
    fi
    
    echo "Ending runner_fn: `date`"
    echo -e "Finished run: \n\n"
}
export -f runner_fn

 
# ----------------------------
#  Submit Job Step
# ----------------------------

srun \
    --nodes=$NUMBER_OF_NODES \
    --ntasks-per-node=1 \
    --gres=gpu:$NGPU \
    --cpus-per-task=$NCPU \
    --output=./job_outputs/%j/slurm-%J.out \
    --error=./job_outputs/%j/slurm-%J.err \
    bash -c "runner_fn"
    
