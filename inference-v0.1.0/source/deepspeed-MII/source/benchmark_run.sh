#!/bin/bash
# Script to run DeepSpeed-MII benchmarking
# Usage: benchmark_run.sh <MODEL> <DATASET> <NODES>
# Example: benchmark_run.sh Mistral-7B-Instruct-v0.3 sharegpt 1

#### ARGUMENTS ASSIGNMENT
if [ "$#" -ne 3 ]; then
  echo "Usage: benchmark_run.sh <MODEL> <DATASET> <NODES>"
  exit 1
fi
MODEL=$1
DATASET=$2
NODES=$3

#### 
OUTPUT_PATH="$CLUSTER_PATH/results/$FRAMEWORK/$MODEL/$DATASET/$TIME_ID/nodes_${NODES}_gpus_${GPUS_PER_NODE}_pipeline_${PIPELINE_PARALLEL}_tensor_${TENSOR_PARALLEL}/iteration_$ITERATION" 

echo "OUTPUT_PATH: $OUTPUT_PATH" 
mkdir -p $OUTPUT_PATH
SBATCH_FILE_PATH="$OUTPUT_PATH/sbatch_script.sh"

printf "#!/bin/bash\n
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION_NAME
#SBATCH --job-name=Benchmark_DeepSpeed-MII_$TIME_ID
#SBATCH --qos=$QOS # quality of service
#SBATCH --error=$OUTPUT_PATH/sbatch.err
#SBATCH --output=$OUTPUT_PATH/sbatch.out
#SBATCH --time=02:00:00

#SBATCH --nodes=$NODES # number of nodes on which to run (N = min[-max])
#SBATCH --ntasks # number of tasks to run # DELETE IF NOT IN THE ORIGINAL SCRIPT
#SBATCH --ntasks-per-node=$NTASKS_PER_NODE # number of tasks to invoke on each node
#SBATCH --tasks-per-node=1 # THIS IS IN vllm_configurable_benchmarking_serve.sh AND IN deepspeed-mii_configurable_benchmarking_serve.sh / It is no a standard argument to sbatch, perhaps it was a typo?

#SBATCH --cpus-per-task=$CPUS_PER_TASK # number of cpus required per task 
#SBATCH --cpus-per-gpu= # number of CPUs required per allocated GPU # DELETE IF NOT IN THE ORIGINAL SCRIPT

# If gpus is the one used, change GPUS_PER_NODE to GPUS in all scripts
#SBATCH --gres=gpu:$GPUS_PER_NODE # required generic resources
#SBATCH --gpus # count of GPUs required for the job # is it better to use gpus or gres or gpus-per-node?
#SBATCH --gpus-per-node # number of GPUs required per allocated node

#SBATCH --mem # minimum amount of real memory # DELETE IF NOT IN THE ORIGINAL SCRIPT


# ENV VAR used in benchmark_template.sh
MODULES=$MODULES
TENSOR_PARALLEL=$TENSOR_PARALLEL
PIPELINE_PARALLEL=$PIPELINE_PARALLEL
FRAMEWORK_PATH=$FRAMEWORK_PATH
OUTPUT_PATH=$OUTPUT_PATH
SOURCE_PATH=$SOURCE_PATH
PORT=$PORT
MODEL=$MODEL
MODEL_PATH=$MODEL_PATH
DATASET=$DATASET
DATASET_PATH=$DATASET_PATH\n" > $SBATCH_FILE_PATH

printf "\n# Moving to $SOURCE_PATH
cd $SOURCE_PATH 

# Clean the environment 
module purge > /dev/null 2>&1 

# Loading modules
module load $MODULES # only execute if MODULES is not empty

# Activating venv in $VENV_PATH
$VENV_ACTIVATION

# --------- Content of benchmark_template.sh ---------\n" >> $SBATCH_FILE_PATH

cat $FRAMEWORK/source/benchmark_template.sh >> $SBATCH_FILE_PATH # TO BE UNCOMMENTED
         
# sbatch $SBATCH_FILE_PATH

echo "DeepSpeed-MII benchmarking script generation completed"



