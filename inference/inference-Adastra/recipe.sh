#!/bin/bash

set -e

# --- Loading modules ---

module purge
module load PrgEnv-gnu 
module load craype-accel-amd-gfx942 
module load craype-x86-genoa 
module load rocm/6.3.3
module load aws-ofi-rccl/1.4.0_rocm6 
module load gcc/13.2.0
module load cray-python
module li

# --- Setting up virtual environment ---

VENV_DIR="test"

if [ "$1" == "--reset" ]; then
    rm -rf "$VENV_DIR"
fi

if [ -d "$VENV_DIR" ]; then
    echo "venv '$VENV_DIR' already exists."
else
    echo "Creating '$VENV_DIR'..."
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# --- Python setup packages ---
python -m pip install --upgrade pip setuptools setuptools_scm wheel amdsmi==6.3.3 ninja tenacity

# Pip install bases
pip install numpy==1.26.3 packaging==24.1 typing-extensions==4.12.2 ninja==1.11.1.4 llvmlite==0.44.0 numba==0.61.2 scipy==1.15.2

# Pip install torch + torchvision + torchaudio
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0  --index-url https://download.pytorch.org/whl/rocm6.3

# Build vllm from source for amd gpu or install it from a wheel already compiled for amd gpu
# pip install $WORKDIR/wheels/vllm/v0.9.1/vllm-0.9.1*

# Additional packages
pip install -r ./requirements.txt