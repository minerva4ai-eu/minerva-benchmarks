#!/bin/bash
# Script to check input parameters for DeepSpeed-MII benchmarking
# Usage: benchmark_input_checker.sh <MODEL> <DATASET> <NODES>
# Example: benchmark_input_checker.sh Mistral-7B-Instruct-v0.3 sharegpt 1

#### ARGUMENTS ASSIGNMENT
if [ "$#" -ne 3 ]; then
  echo "Usage: benchmark_input_checker.sh <MODEL> <DATASET> <NODES>"
  exit 1
fi
MODEL=$1
DATASET=$2
NODES=$3

# Error exit if model is Llama-3.1-405B.
if [ "$MODEL" = "Llama-3.1-405B" ]; then
  echo "Error: DeepSpeed-MII does not support $MODEL"
  exit 1
fi

# Error exit if model is gemma-3-12b-it.
if [ "$MODEL" = "gemma-3-12b-it" ]; then
  echo "Error: DeepSpeed-MII does not support $MODEL"
  exit 1
fi

# Error exit if NODES is not set to one.
if [ "$NODES" != 1 ]; then
  echo "Error: DeepSpeed-MII does not support NODES=$NODES. Valid options are: NODES=1"
  exit 1
fi

# Check if modules can be loaded otherwise exit with error
# This check might not be possible for example with cuda on login node

echo "Input parameters are valid for DeepSpeed-MII benchmarking"