#!/bin/bash
# Script to set up the environment for running benchmarks on CINECA Leonardo cluster
# Usage: source set_env.sh <FRAMEWORK> <MODEL>
# Example: set_env.sh deepspeed-MII Mistral-7B-Instruct-v0.3

#### ARGUMENTS ASSIGNMENT
if [ "$#" -ne 2 ]; then
    echo "Usage: source set_env.sh <FRAMEWORK> <MODEL>"
    exit 1
fi
FRAMEWORK=$1
MODEL=$2

# DO I NEED TO MODULE PURGE? Can be useful when sending multiple benchmark 
# (ALSO PURGE IN THE SBATCH FILE)

#### MODIFY AS REQUIRED

# Job specific
PARTITION_NAME="boost_usr_prod"
ACCOUNT="MNRVA_bench"
QOS="normal" # TO BE SET 
PYTHON="python/3.11.7"
MODULES="$PYTHON" #"$PYTHON cuda/12.1.105-gcc-11.2.0"

# Cluster specific
USER_PATH="/leonardo_scratch/large/userinternal/$USER"

# Model specific
MODEL_PARENT_PATH="/leonardo_work/MNRVA_bench/models"
if [ "$MODEL" = "Llama-3.1-8B-Instruct" ]; then
  MODEL_PATH="$MODEL_PARENT_PATH/$MODEL"
elif [ "$MODEL" = "gemma-3-12b-it" ]; then
  MODEL_PATH="$MODEL_PARENT_PATH/$MODEL"
elif [ "$MODEL" = "Mistral-7B-Instruct-v0.3" ]; then
  MODEL_PATH="$MODEL_PARENT_PATH/$MODEL"
elif [ "$MODEL" = "Llama-3.1-405B" ]; then
  MODEL_PATH="$MODEL_PARENT_PATH/$MODEL"
else
  echo "Error: Unknown model $MODEL. Valid options are: Llama-3.1-8B-Instruct gemma-3-12b-it Mistral-7B-Instruct-v0.3 Llama-3.1-405B"
  exit 1
fi


#### DO NOT MODIFY BELOW THIS LINE

SOURCE_PATH="$USER_PATH/minerva-benchmarks/inference-v0.1.0/source" # This is incorrect for the new version
echo "SOURCE_PATH: $SOURCE_PATH"
cd $SOURCE_PATH # if cd terminates the job, then add || exit 1 # This is not needed if MINERVA/benchmarks are removed as levels

CLUSTER_PATH="$SOURCE_PATH/$CLUSTER"
echo "CLUSTER_PATH: $CLUSTER_PATH"

FRAMEWORK_PATH="$SOURCE_PATH/$FRAMEWORK"
echo "FRAMEWORK_PATH: $FRAMEWORK_PATH"
VENV_PATH="$FRAMEWORK_PATH/$FRAMEWORK"

# Clean the environment # IS THIS NEEDED IN LOGIN NODE?
module purge > /dev/null 2>&1 

# Creating the venv when needed

# Q: Perhaps is better to exit if venv exists. That way we don't waste compute time in installing it. Create a different script to creat the venv.
# A: This script is ran on login node, so no compute time wasted. The script for creating the venv cannot be outside CINECA-Leonardo because $MODULES are 
# defined in this script. In any case this loop will only run once per framework per user.
if [ ! -d "$VENV_PATH" ]; then 
  echo "Loading modules and creating venv in $VENV_PATH"
  module load $PYTHON # Should we load other modules to create the venv?
  python -m venv $VENV_PATH
  source $VENV_PATH/bin/activate
  # pip install --upgrade pip # This is not needed if requirements.txt has specific pip version
  pip install -r $FRAMEWORK_PATH/venv/requirements.txt
  deactivate
fi


# Saving the venv activation script in a variable to be used in benchmark_run.sh
VENV_ACTIVATION="# Checks if venv is already activated based on VIRTUAL_ENV variable. Does this also work for Conda envs? Not a problem here but in other clusters.
if [ -z \$VIRTUAL_ENV ]; then
  source $VENV_PATH/bin/activate
else # alternately, we can just deactivate any existing venv and activate the correct one
  deactivate #2>/dev/null # I think the 2>/dev/null is only needed if there is no venv activated because it would give an error
  source $VENV_PATH/bin/activate
fi"

echo "Environment set correctly"