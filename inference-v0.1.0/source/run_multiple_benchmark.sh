#!/bin/bash
# Script to launch a job for running a multiple benchmark 
# Usage: source run_multiple_benchmark.sh <CLUSTER> <FRAMEWORK> <MODEL> <DATASET> <NODES_BATCH> <GPUS_PER_NODE_BATCH> <PIPELINE_PARALLEL_BATCH> <TENSOR_PARALLEL_BATCH> <ITERATIONS> <CPUS_PER_TASK> <NTASKS_PER_NODE>
# Example: bash run_multiple_benchmark.sh CINECA-Leonardo deepspeed-MII Mistral-7B-Instruct-v0.3 sharegpt "1" "1 2" "1 2" "1 2" "2" "1" "1"

CLUSTERS=("CINECA-Leonardo" "BSC-MN5" "IDRIS-JZ")

#### DO NOT MODIFY BELOW THIS LINE

#### ARGUMENTS ASSIGNMENT. To be modified if number of arguments changes
if [ "$#" -ne 11 ]; then
    echo "Usage: source run_single_benchmark.sh <> <>"
    exit 1
fi

# BENCHMARK PARAMETERS
CLUSTER=$1
FRAMEWORK=$2
MODEL=$3
DATASET=$4

# SBATCH PARAMETERS
NODES_BATCH=$5
GPUS_PER_NODE_BATCH=$6
# For this two think about their limits. This should be related to the resources allocated
PIPELINE_PARALLEL_BATCH=$7
TENSOR_PARALLEL_BATCH=$8
ITERATIONS=$9
CPUS_PER_TASK=$10
NTASKS_PER_NODE=$11

PORT=2950 # DO WE NEED THIS OR CAN WE FIX IT?

# Test if CLUSTER is valid
if [[ ! " ${CLUSTERS[*]} " =~ [[:space:]]${CLUSTER}[[:space:]] ]]; then
  echo "Error: Unknown cluster $CLUSTER. Valid options are: ${CLUSTERS[@]}"
  exit 1
fi  

TIME_ID=$(date +"%Y%m%d_%H%M")

ENVIRONMENT_INPUTS="$FRAMEWORK $MODEL"
echo " ================================ Setting environment for $FRAMEWORK benchmark on $CLUSTER ================================ "
source ./$CLUSTER/source/set_env.sh $ENVIRONMENT_INPUTS

# DATASETS PATH MAP, will this be stored in each cluster or in the repo? If in the repo, then save them in inference-v0.1.0/datasets
if [ "$DATASET" = "sharegpt" ]; then
  DATASET_PATH="../datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
elif [ "$DATASET" = "sonnet" ]; then
  DATASET_PATH="../datasets/sonnet.txt"
else
  echo "Error: Unknown dataset $DATASET. Valid options are: sharegpt sonnet"
  exit 1
fi

# Parameter checking for all benchmarks
for NODES in ${NODES_BATCH[@]}; do 
  for GPUS_PER_NODE in ${GPUS_PER_NODE_BATCH[@]}; do 
    for PIPELINE_PARALLEL in ${PIPELINE_PARALLEL_BATCH[@]}; do
      for TENSOR_PARALLEL in ${TENSOR_PARALLEL_BATCH[@]}; do
        
        BENCHMARK_INPUTS="$MODEL $DATASET $NODES"
        echo " ================================ Checking input for $FRAMEWORK benchmark on $CLUSTER ================================ "
        source ./$FRAMEWORK/source/benchmark_input_checker.sh $BENCHMARK_INPUTS
        
      done
    done
  done
done

# Job launching for all benchmarks
for NODES in ${NODES_BATCH[@]}; do 
  for GPUS_PER_NODE in ${GPUS_PER_NODE_BATCH[@]}; do 
    for PIPELINE_PARALLEL in ${PIPELINE_PARALLEL_BATCH[@]}; do
      for TENSOR_PARALLEL in ${TENSOR_PARALLEL_BATCH[@]}; do
        for ITERATION in $(seq 1 $ITERATIONS); do 
    
          BENCHMARK_INPUTS="$MODEL $DATASET $NODES" # I dont like having BENCHMARK_INPUTS here again, but for now leave it
          echo " ================================ Launching $FRAMEWORK benchmark on $CLUSTER ================================ "
          source ./$FRAMEWORK/source/benchmark_run.sh $BENCHMARK_INPUTS
        
        done
      done
    done
  done
done