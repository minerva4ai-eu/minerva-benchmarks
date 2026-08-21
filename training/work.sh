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
export MAX_MODEL_LENGTH=${11}
export TRIAL=${12}

# Rules Check - First Thing
if check_config; then
    echo Invalid configuration: $MODEL / $FRAMEWORK / $PARALLELISM / $NUMBER_OF_NODES / $DATASET / $BATCH_SIZE / $GRAD_ACCUM / $ENABLE_COMPILE / $PRECISION / $LR / $MAX_MODEL_LENGTH / $TRIAL / $SLURM_JOB_NUM_NODES
    exit 0
fi

eval "$MINERVAPATHS"
export MODEL_PATH=${paths[$MODEL]}
export DATASET_PATH=${paths[$DATASET]}

NGPU=0
NCPU=$SLURM_CPUS_ON_NODE
export NPROC=$SLURM_NTASKS_PER_NODE
if [[ $PARALLELISM == "none" ]]; then
    NPROC=1
    NCPU=$SLURM_CPUS_PER_TASK
fi

if [[ $SLURM_GPUS_ON_NODE -gt 0 ]]; then
    if [[ $PARALLELISM == "none" ]]; then
        NGPU=1
    else
        NGPU=$SLURM_GPUS_ON_NODE
    fi
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
"  MAX_MODEL_LENGTH: $MAX_MODEL_LENGTH" \
"  EPOCHS: $EPOCHS" \
"  STEPS: $STEPS" \
"  NGPU: $NGPU" \
"  NCPU: $NCPU" \
"  NPROC: $NPROC" \
"  OUT_DIR: $OUT_DIR" \
"  CONFIG_ENV: $CONFIG_ENV"  \
"  CONTAINER_ARGS: $CONTAINER_ARGS" \
"  CONTAINER_PATH: $CONTAINER_PATH"\
"  SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES" \
"  SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE" \
"  SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE" \
"  SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK" \
"  SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE" \
"  SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" \
"  SLURM_NODEID: $SLURM_NODEID" \
"  SLURM_PROCID: $SLURM_PROCID"
# "  RESULTSDIR: $RESULTSDIR" \

runner_fn () {

    echo -e "\n\nStarting runner: ..."

    scontrol show step $SLURM_JOB_ID.$SLURM_STEP_ID

    # export RESULTSDIR=$OUT_DIR/bsc-mn5-acc/$SLURM_JOB_ID/$MODEL/$FRAMEWORK/$PARALLELISM/$DATASET/nodes-$NUMBER_OF_NODES/trial-$TRIAL/output
    export RESULTSDIR=$OUT_DIR/$SLURM_CLUSTER_NAME-$SLURM_JOB_PARTITION/$SLURM_JOB_ID/$MODEL/$FRAMEWORK/$PARALLELISM/$DATASET/nodes-$SLURM_STEP_NUM_NODES/run_id-$TRIAL/launch-1/output
    if [[ ! -d $RESULTSDIR ]]; then
        mkdir -p $RESULTSDIR
    fi

    # Cache Setup
    export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor_$USER/$SLURM_JOB_ID.$SLURM_STEP_ID
    
    export TORCH_LOGS="recompiles"
    # export TORCH_LOGS="+recompiles,+dynamo"
    export TORCHDYNAMO_VERBOSE=1

    # Check Environment Variables
    echo Config in runner function:
    echo "  MODEL: $MODEL (path: $MODEL_PATH)" \
    "  FRAMEWORK: $FRAMEWORK" \
    "  PARALLELISM: $PARALLELISM" \
    "  NUMBER_OF_NODES: $NUMBER_OF_NODES"  \
    "  SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES" \
    "  SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE" \
    "  SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" \
    "  SLURM_STEP_NUM_NODES: $SLURM_STEP_NUM_NODES"  \
    "  SLURM_STEP_NUM_TASKS: $SLURM_STEP_NUM_TASKS"  \
    "  SLURM_STEP_TASKS_PER_NODE: $SLURM_STEP_TASKS_PER_NODE"\
    "  DATASET: $DATASET (path: $DATASET_PATH)" \
    "  BATCH_SIZE: $BATCH_SIZE" \
    "  GRAD_ACCUM: $GRAD_ACCUM" \
    "  ENABLE_COMPILE: $ENABLE_COMPILE" \
    "  PRECISION: $PRECISION" \
    "  TRIAL: $TRIAL" \
    "  LR: $LR" \
    "  MAX_MODEL_LENGTH: $MAX_MODEL_LENGTH" \
    "  EPOCHS: $EPOCHS" \
    "  STEPS: $STEPS" \
    "  RESULTSDIR: $RESULTSDIR" \
    "  CONFIG_ENV: $CONFIG_ENV" \
    "  CONTAINER_ARGS: $CONTAINER_ARGS" \
    "  CONTAINER_PATH: $CONTAINER_PATH" \
    "  SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE" \
    "  SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE" \
    "  SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK" \
    "  SLURM_NODEID: $SLURM_NODEID" \
    "  SLURM_PROCID: $SLURM_PROCID" \
    "  SLURM_STEP_NODELIST: $SLURM_STEP_NODELIST" \
    "  NPROC: $NPROC" \
    "  TORCHINDUCTOR_CACHE_DIR: $TORCHINDUCTOR_CACHE_DIR"

    # env | grep SLURM

    # For python scripts which read with os.environment.get
    export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
    export GPU_NODE=$SLURM_GPUS_ON_NODE

    # No slurm commands inside the container (environment variables yes)
    export HEAD_NODE=$(scontrol show hostnames ${SLURM_STEP_NODELIST} | head -n 1)

    if [[ $CONFIG_ENV == "singularity" ]]; then
        echo Using container: $CONTAINER_PATH
        CWD=`pwd`
        singularity exec $CONTAINER_ARGS $CONTAINER_PATH $CWD/train.sh >> $RESULTSDIR/bash-job$SLURM_JOB_ID-step$SLURM_STEP_ID-task$SLURM_PROCID.log 2>&1
    else
        ./train.sh >> $RESULTSDIR/bash-job$SLURM_JOB_ID-step$SLURM_STEP_ID-task$SLURM_PROCID.log 2>&1
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
    
