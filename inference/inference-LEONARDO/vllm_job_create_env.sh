#!/bin/bash
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH -A MNRVA_bench 
#SBATCH --partition=lrd_all_serial
##SBATCH --ntasks-per-node=1 # 1 task
##SBATCH --gres=gpu:4        # 1 gpus per node out of 4
##SBATCH --exclusive
##SBATCH --cpus-per-task=16
##SBATCH --nodes=2
##SBATCH --qos=boost_qos_dbg
##SBATCH --time=2:00:00  



export GPUS_PER_NODE=4
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export MASTER_PORT=6000
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export MASTER_ADDR_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname --ip-address)
export BNB_CUDA_VERSION=121

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo hostname = `hostname`
echo HOSTNAMES = $HOSTNAMES
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT
echo SLURM_PROCID= $SLURM_PROCID
echo NNODES= $NNODES    
echo WORLD_SIZE= $WORLD_SIZE    
echo NODE_RANK= $NODE_RANK      
echo NODE_NAME = $SLURMD_NODENAME
echo MASTER_ADDR_IP = $MASTER_ADDR_iP

module load gcc
module load cuda

module load python

python -m venv env_vllm
source env_vllm/bin/activate

srun python create_env_vllm.py
