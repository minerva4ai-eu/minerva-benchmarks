#!/bin/bash

echo "Starting train: ..."

HEAD_PORT=29500

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
"  MAX_MODEL_LENGTH: $MAX_MODEL_LENGTH" \
"  EPOCHS: $EPOCHS" \
"  STEPS: $STEPS" \
"  OUT_DIR: $OUT_DIR" \
"  CONFIG_ENV: $CONFIG_ENV"  \
"  CONTAINER_ARGS: $CONTAINER_ARGS" \
"  CONTAINER_PATH: $CONTAINER_PATH"\
"  RESULTSDIR: $RESULTSDIR" \
"  SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES" \
"  SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE" \
"  SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" \
"  SLURM_STEP_NUM_NODES: $SLURM_STEP_NUM_NODES" \
"  SLURM_STEP_NUM_TASKS: $SLURM_STEP_NUM_TASKS"  \
"  SLURM_STEP_TASKS_PER_NODE: $SLURM_STEP_TASKS_PER_NODE"\
"  SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE" \
"  SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK" \
"  SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE" \
"  SLURM_NODEID: $SLURM_NODEID" \
"  SLURM_PROCID: $SLURM_PROCID" \
"  SLURM_STEP_NODELIST: $SLURM_STEP_NODELIST" \
"  HEAD_NODE: $HEAD_NODE" \
"  HEAD_PORT: $HEAD_PORT" \
"  TORCHINDUCTOR_CACHE_DIR: $TORCHINDUCTOR_CACHE_DIR"
# "  TENSORBOARD_LOGGING_DIR: $TENSORBOARD_LOGGING_DIR"
# "  USECPU: $USECPU" \

# pwd
# env | grep SLURM

ft_args="--model $MODEL_PATH \
        --data $DATASET_PATH \
        --output_dir $RESULTSDIR \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_MODEL_LENGTH \
        ${STEPS:+--max_steps "$STEPS"} \
        --precision $PRECISION \
        --lr $LR \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --dataloader_num_workers 4 \
        --dataset $DATASET"
        
if [[ $ENABLE_COMPILE == true ]]; then
    ft_args+=" --enable_compile"
fi

if [[ $FRAMEWORK == torchrun ]]; then
    echo "Using torchrun for distributed training"
    dist_args="--nnodes $SLURM_STEP_NUM_NODES --nproc_per_node $NPROC \
        --rdzv_id $SLURM_JOB_ID.$SLURM_STEP_ID --rdzv_backend c10d --rdzv_endpoint ${HEAD_NODE}:${HEAD_PORT}"
    if [[ $PARALLELISM == "none" ]]; then
        echo Running none
        python scripts/torchrun-common/finetune-$PARALLELISM.py $ft_args
        # python finetune.py $ft_args > $RESULTSDIR/py-job$SLURM_JOB_ID-step$SLURM_STEP_ID-task$SLURM_PROCID.out 2>&1
    else
        torchrun $dist_args scripts/accelerate-common/finetune-$PARALLELISM.py $ft_args
        # torchrun $dist_args finetune.py $ft_args > $RESULTSDIR/py-job$SLURM_JOB_ID-step$SLURM_STEP_ID-task$SLURM_PROCID.out 2>&1
    fi

elif [[ $FRAMEWORK == accelerate ]]; then
    echo "Using accelerate for distributed training"
    dist_args="--num_machines $SLURM_STEP_NUM_NODES \
            --num_processes $(( $SLURM_STEP_NUM_NODES * $NPROC )) \
            --multi-gpu \
            --machine_rank $SLURM_NODEID \
            --rdzv_backend c10d \
            --main_process_ip $HEAD_NODE \
            --main_process_port $HEAD_PORT \
             --dynamo_backend inductor --dynamo_use_dynamic"
    accelerate launch $dist_args scripts/accelerate-common/finetune-$PARALLELISM.py $ft_args
elif [[ $FRAMEWORK == deepspeed ]]; then
    echo "Using deepspeed for distributed training"
    # Add deepspeed execution here
    echo "Deepspeed execution would be here"
elif [[ $FRAMEWORK == deepspeed-accelerate ]]; then
    echo "Using deepspeed-accelerate for distributed training"
    # Add deepspeed-accelerate execution here
    echo "Deepspeed execution would be here"
else
    echo "Unknown framework: $FRAMEWORK"
    exit 1
fi

if [[ -d $TORCHINDUCTOR_CACHE_DIR ]]; then
    echo "ENABLE_COMPILE = $ENABLE_COMPILE... Checking TORCHINDUCTOR_CACHE_DIR = ($TORCHINDUCTOR_CACHE_DIR)...."
    echo Disk Usage of TORCHINDUCTOR_CACHE_DIR:
    du -h --max-depth=0 $TORCHINDUCTOR_CACHE_DIR
    if [[ -d $TORCHINDUCTOR_CACHE_DIR/triton ]]; then
        echo Listing contents of TORCHINDUCTOR_CACHE_DIR/triton:
        ls -lha $TORCHINDUCTOR_CACHE_DIR/triton
    fi
    if [[ -d $TORCHINDUCTOR_CACHE_DIR/fxgraph ]]; then
        echo Listing contents of TORCHINDUCTOR_CACHE_DIR/fxgraph:
        ls -lha $TORCHINDUCTOR_CACHE_DIR/fxgraph
    fi
    if [[ -d $TORCHINDUCTOR_CACHE_DIR/aotautograd ]]; then
        echo Listing contents of TORCHINDUCTOR_CACHE_DIR/aotautograd:
        ls -lha $TORCHINDUCTOR_CACHE_DIR/aotautograd
    fi
    if [[ -d $TORCHINDUCTOR_CACHE_DIR/cache ]]; then
        echo Listing contents of TORCHINDUCTOR_CACHE_DIR/cache:
        ls -lha $TORCHINDUCTOR_CACHE_DIR/cache
    fi
    echo Listing contents of TORCHINDUCTOR_CACHE_DIR:
    ls -lha $TORCHINDUCTOR_CACHE_DIR
fi

echo "Ending worker_fn: `date`"
echo -e "Finished work: \n\n"

# original YAML: ddp--bs8-grad_accum8-compileFalse-precbf16-steps2 
# output from multple configurations will be stored in same directory => name output/log files with stepid