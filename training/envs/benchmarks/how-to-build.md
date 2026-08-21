# How to Build — Benchmark / Training Environments

This directory contains Singularity image definitions for the MINERVA training benchmarks. The only available CUDA variant is **CUDA 12.1**, which provides a **runtime** image.

## Directory Structure

```
benchmarks/
└── cuda121-flash-attn/              # CUDA 12.1 environment
    ├── pyproject.toml               # Python dependencies
    ├── uv.lock                      # Locked dependency versions
    ├── singularity_uv-runtime.def   # Singularity definition (runtime)
    └── singularity_uv-devel.def     # Singularity definition (development)
```

## Prerequisites

- **Singularity** (or Apptainer) installed and accessible
- **Docker** installed and running (Singularity uses Docker as the bootstrap source)
- **uv** installed (for local development)
- NVIDIA GPU drivers installed on the host
- Sufficient disk space (~10-15 GB recommended, due to CUDA toolkit + PyTorch)

---

## Automated Installation (Recommended)

Instead of following the steps below manually, you can use the automated scripts:

```bash
# From the training/ directory
cd install

# Install all Python environments (including benchmarks)
bash install-all-envs.sh

# Build all Singularity containers (including benchmarks)
bash build-all-singularity.sh
```

---

## Option 1: Local Development with `uv`

Use this option to develop or test the environment locally before building the Singularity image.

### Step 1 — Install `uv`

```bash
# Using pip
pip install uv

# Or using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2 — Create and activate a local virtual environment

```bash
cd cuda121-flash-attn

# Create and activate the environment
uv venv
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv sync --locked

# Verify
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import flash_attn; print('flash-attn OK')"
```

### Step 3 — Update dependencies (optional)

```bash
# Add a new dependency
uv add <package-name>

# Update all dependencies
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package <package-name>
```

---

## Option 2: Build Singularity Container

Use this option to build the Singularity image for execution on HPC clusters.

### Step 1 — Build the Singularity images

From the `cuda121-flash-attn/` directory:

```bash
# Build the runtime image (recommended for benchmark execution)
singularity build singularity_uv-runtime.sif singularity_uv-runtime.def

# Build the devel image (includes dev dependencies for debugging)
singularity build singularity_uv-devel.sif singularity_uv-devel.def
```

Each build will:
1. Pull the `nvidia/cuda:12.1.1-runtime-ubuntu22.04` (or `devel`) Docker base image
2. Install system dependencies (`python3-pip`, `python3-dev`, `git`, `curl`, `build-essential`, `libibverbs-dev`, `ibverbs-utils`, `cuda-nvcc-12-1`, NCCL from NVIDIA repo)
3. Install `uv` via `pip`
4. Run `uv sync --locked` to install all Python dependencies from `pyproject.toml`
5. Produce the `.sif` image file

### Step 2 — Verify the images

```bash
# Runtime image
singularity exec singularity_uv-runtime.sif python --version
singularity exec singularity_uv-runtime.sif python -c "import torch; print('CUDA:', torch.cuda.is_available())"
singularity exec singularity_uv-runtime.sif nvcc --version

# Devel image
singularity exec singularity_uv-devel.sif python --version
singularity exec singularity_uv-devel.sif python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Dependencies

See `pyproject.toml` for the full list. Key packages include:

| Category | Packages |
|----------|----------|
| **Deep Learning** | `torch==2.5.1`, `transformers==4.57.0`, `accelerate==1.10.1`, `trl==1.4.0`, `torchtune==0.6.0`, `torchao==0.9.0`, `lightning==2.5.5` |
| **Training** | `deepspeed==0.15.4`, `bitsandbytes==0.45.0`, `ray==2.38.0`, `torchmetrics==1.8.2`, `torchdata==0.11.0` |
| **Attention** | `flash-attn==2.8.3` (requires CUDA + compilation) |
| **Data** | `datasets==4.8.5`, `tiktoken==0.9.0`, `tokenizers==0.22.1`, `sentencepiece==0.2.0`, `safetensors==0.5.0` |
| **Config/CLI** | `hydra-core==1.3.2`, `pyyaml==6.0.2`, `python-dotenv==1.1.1`, `rich==13.9.0`, `prompt-toolkit==3.0.52`, `tqdm==4.67.1` |
| **Monitoring** | `psutil==6.1.1`, `pynvml==13.0.1`, `memray==1.14.0` |
| **Visualization** | `matplotlib==3.10.7`, `seaborn==0.13.2`, `pandas==2.2.3`, `numpy==2.4.6`, `scikit-learn==1.8.0`, `scipy==1.17.1` |
| **Other** | `einops==0.8.0`, `evaluate==0.4.3`, `huggingface-hub==0.35.3`, `kagglehub==0.3.10`, `tensorboardx==2.6.2.2`, `triton==3.1.0` |

PyTorch wheels are sourced from `https://download.pytorch.org/whl/cu121` via `[[tool.uv.index]]` in `pyproject.toml`.

---

## Notes

- **`flash-attn`** requires compilation from source. The `devel` image includes `build-essential` and the CUDA toolkit needed for this. The `runtime` image includes `cuda-nvcc-12-1` as well, so `flash-attn` can still be compiled during the Singularity build.
- The `no-build-isolation-package = ["flash-attn"]` setting in `pyproject.toml` ensures `flash-attn` is built with the correct dependencies.
- The `UV_PYTHON_INSTALL_DIR=/app/uv-python` environment variable in the `.def` file controls where `uv` installs the Python interpreter (not under `/root` to avoid permission issues in Singularity).
- The **runtime image** additionally installs NCCL from the NVIDIA apt repository (including InfiniBand plugins) and sets `CUDA_HOME` and `LD_LIBRARY_PATH` in the `%environment` section.
- Build times can be **30–60 minutes** due to `flash-attn` compilation and large PyTorch downloads.
