# Environment Management

This directory contains Python environment definitions for building Singularity containers used in MINERVA training benchmarks. Environments are managed using [uv](https://github.com/astral-sh/uv), a fast Python package and project manager, and packaged into Singularity containers for execution on HPC clusters.

## Philosophy

The MINERVA project operates across two distinct computing contexts:

1. **Local development** — where researchers iterate quickly on code, test dependencies, and debug issues
2. **HPC execution** — where reproducible, isolated environments run on shared cluster resources (MareNostrum5, etc.)

These contexts have different requirements. Local development needs speed and flexibility. HPC execution needs reproducibility, isolation, and compatibility with cluster infrastructure. This directory provides tools to bridge both worlds.

## Table of Contents

- [Philosophy](#philosophy)
- [Overview](#overview)
- [Environment Types](#environment-types)
  - [benchmarks/ — Training Benchmark Environment](#benchmarks--training-benchmark-environment)
  - [cli/ — Command-Line Interface Environment](#cli--command-line-interface-environment)
- [Build Choices](#build-choices)
  - [Why `uv`?](#why-uv)
  - [Why Singularity?](#why-singularity)
  - [The Two Approaches](#the-two-approaches)
- [Quick Start](#quick-start)
  - [Automated Installation (Recommended)](#automated-installation-recommended)
  - [Local Development](#local-development)
  - [Build Singularity Container](#build-singularity-container)
- [Using the Environments](#using-the-environments)
  - [Running Benchmarks](#running-benchmarks)
  - [Direct Singularity Execution](#direct-singularity-execution)
  - [GPU Access](#gpu-access)
- [Directory Structure](#directory-structure)
  - [benchmarks/cuda121-flash-attn/](#benchmarkscuda121-flash-attn)
  - [cli/](#cli)
- [Key Dependencies](#key-dependencies)
  - [Benchmarks Environment](#benchmarks-environment)
  - [CLI Environment](#cli-environment)
- [Dependency Management](#dependency-management)
  - [Updating Dependencies](#updating-dependencies)
  - [Rebuilding Containers](#rebuilding-containers)
- [Adding a New CUDA Version](#adding-a-new-cuda-version)
- [Troubleshooting](#troubleshooting)
- [See Also](#see-also)

---

## Philosophy

```
envs/
├── benchmarks/                    # Training benchmark 
│   ├── cuda121-flash-attn/        # CUDA 12.1 + flash attn
│   ├── cuda128-flash-attn/        # CUDA 12.8 + flash attn 
│   └── cuda130-flash-attn/        # CUDA 13.0 + flash attn 
│       ├── pyproject.toml
│       ├── uv.lock
│       └── singularity_uv-runtime.def
└── cli/                           # CLI environment (minerva-cli.sh)
    ├── pyproject.toml
    ├── uv.lock
    └── singularity-uv.def
```

---

## Environment Types

### `benchmarks/` — Training Benchmark Environment

The primary environment for running training benchmarks. It includes a full deep learning stack with PyTorch, transformers, DeepSpeed, and flash attention support. This environment is designed for GPU-accelerated training workloads on HPC clusters.

**Key features:**
- PyTorch 2.5.1 with CUDA 12.1 support
- Flash attention 2.8.3 (requires compilation)
- DeepSpeed 0.15.4 for distributed training
- Hugging Face transformers 4.57.0
- Additional tools: accelerate, peft, trl, torchtune, lightning, etc.

**Runtime image** (`singularity_uv-runtime.def`): Optimized for benchmark execution. Includes CUDA toolkit, NCCL, and InfiniBand support for multi-GPU communication.

### `cli/` — Command-Line Interface Environment

A minimal environment for the `minerva-cli.sh` tool. This environment provides the Click-based CLI for benchmark submission and management. It does not include CUDA libraries or GPU dependencies.

**Key features:**
- Click 8.4+ for CLI argument parsing
- Hydra 1.3+ for configuration management
- Rich 15+ for terminal output formatting
- Data visualization: matplotlib, seaborn, pandas
- Utilities: psutil, tqdm, pyyaml, python-dotenv

---

## Build Choices

### Why `uv`?

`uv` is used as the primary Python package manager because it:
- Is significantly faster than pip for dependency resolution and installation
- Provides reproducible builds via `uv.lock`
- Supports modern Python project standards (`pyproject.toml`)
- Handles multiple Python versions and virtual environments efficiently

### Why Singularity?

Singularity (Apptainer) is used for HPC containerization because it:
- Integrates seamlessly with SLURM and other HPC job schedulers
- Provides process-level isolation without requiring root privileges
- Preserves host system compatibility (GPU drivers, network interfaces, etc.)
- Creates portable, self-contained images (`.sif` files)

### The Two Approaches

There are **two ways** to build and use these environments:

#### 1. Local Development with `uv` (Python venv)

Creates a Python virtual environment on your local machine using `uv` for dependency management.

**When to use:**
- Developing or modifying dependencies
- Testing new package versions
- Debugging issues before building containers
- Quick iteration without container overhead

**Pros:**
- Fast setup and teardown
- Direct access to your local Python interpreter
- Easy to debug with standard Python tools

**Cons:**
- Environment may differ from HPC cluster
- No system-level isolation
- Requires matching CUDA/toolchain versions locally

#### 2. Singularity Container Build

Builds a Singularity container image that bundles the Python environment with all system dependencies into a portable `.sif` file.

**When to use:**
- Running benchmarks on HPC clusters
- Ensuring reproducible environments across different machines
- Isolating dependencies from the host system
- Submitting jobs via SLURM

**Pros:**
- Exact reproducibility across environments
- System-level isolation
- Includes all system dependencies (CUDA, NCCL, etc.)
- Portable — can be shared and reused

**Cons:**
- Slower build times (30–60 minutes for benchmarks)
- Larger disk footprint
- Less flexible for quick changes

---

## Quick Start

### Automated Installation (Recommended)

The `install/` directory provides scripts to install all environments and build all containers in one step:

```bash
# From the training/ directory
cd install

# Install all Python environments (uv sync)
bash install-all-envs.sh

# Build all Singularity containers
bash build-all-singularity.sh
```

Both scripts can be run from either `training/` or `training/install/`.

### Local Development

```bash
# Benchmarks environment
cd envs/benchmarks/cuda121-flash-attn
uv venv
source .venv/bin/activate
uv sync --locked

# CLI environment
cd envs/cli
uv venv
source .venv/bin/activate
uv sync --locked
```

### Build Singularity Container

```bash
# Benchmarks environment
cd envs/benchmarks/cuda121-flash-attn
singularity build singularity_uv-runtime.sif singularity_uv-runtime.def

# CLI environment
cd envs/cli
singularity build singularity-uv.sif singularity-uv.def
```

---

## Using the Environments

### Running Benchmarks

The `minerva-cli.sh` wrapper automatically runs commands inside the Singularity container:

```bash
bash minerva-cli.sh run --config-name base-MN5
```

### Direct Singularity Execution

```bash
# Execute a Python script inside the container
singularity exec --nv envs/benchmarks/cuda121-flash-attn/singularity_uv-runtime.sif python script.py

# Start an interactive shell
singularity shell --nv envs/benchmarks/cuda121-flash-attn/singularity_uv-runtime.sif
```

### GPU Access

The `--nv` flag enables NVIDIA GPU access inside the container:

```bash
singularity exec --nv ...  # Enables CUDA, cuDNN, NCCL
```

---

## Directory Structure

### `benchmarks/cuda121-flash-attn/`

```
cuda121-flash-attn/
├── pyproject.toml               # Python dependencies (PyTorch, transformers, DeepSpeed, etc.)
├── uv.lock                      # Locked dependency versions
├── singularity_uv-runtime.def   # Singularity definition (runtime image)
└── singularity_uv-devel.def     # Singularity definition (development image)
```

### `cli/`

```
cli/
├── pyproject.toml         # CLI dependencies (click, hydra, rich, etc.)
├── uv.lock                # Locked dependencies
└── singularity-uv.def     # Singularity definition file
```

---

## Key Dependencies

### Benchmarks Environment

| Category | Packages |
|----------|----------|
| Deep Learning | `torch==2.5.1`, `transformers==4.57.0`, `accelerate==1.10.1`, `trl==1.4.0`, `torchtune==0.6.0`, `lightning==2.5.5` |
| Training | `deepspeed==0.15.4`, `bitsandbytes==0.45.0`, `ray==2.38.0`, `torchmetrics==1.8.2`, `torchdata==0.11.0` |
| Attention | `flash-attn==2.8.3` (requires CUDA + compilation) |
| Data | `datasets==4.8.5`, `tiktoken==0.9.0`, `tokenizers==0.22.1`, `sentencepiece==0.2.0`, `safetensors==0.5.0` |
| Monitoring | `psutil==6.1.1`, `pynvml==13.0.1`, `memray==1.14.0` |
| Visualization | `matplotlib==3.10.7`, `seaborn==0.13.2`, `pandas==2.2.3`, `numpy==2.4.6` |

PyTorch wheels are sourced from `https://download.pytorch.org/whl/cu121` via `[[tool.uv.index]]` in `pyproject.toml`.

### CLI Environment

| Category | Packages |
|----------|----------|
| CLI/UX | `click>=8.4.1`, `rich>=15.0.0`, `prompt-toolkit`, `tqdm` |
| Config | `pyyaml`, `omegaconf`, `hydra-core>=1.3.2` |
| Data | `numpy`, `pandas`, `seaborn`, `matplotlib` |
| Utilities | `python-dotenv`, `psutil`, `setuptools` |

---

## Dependency Management

### Updating Dependencies

```bash
# Add a new dependency
cd envs/benchmarks/cuda121-flash-attn/
uv add <package-name>

# Update all dependencies
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package <package-name>
```

### Rebuilding Containers

After updating dependencies, rebuild the Singularity container:

```bash
cd envs/benchmarks/cuda121-flash-attn/
singularity build singularity_uv-runtime.sif singularity_uv-runtime.def
```

---

## Adding a New CUDA Version

To add a new CUDA version (e.g., CUDA 12.4):

1. **Copy existing directory:**
   ```bash
   cp -r envs/benchmarks/cuda121-flash-attn envs/benchmarks/cuda124-flash-attn
   ```

2. **Update Singularity definition files:**
   - Change `From:` line to `nvidia/cuda:12.4.x-runtime-ubuntu22.04`
   - Update `cuda-nvcc-12-x` package name
   - Update `%environment` paths if needed

3. **Update pyproject.toml** if CUDA version affects dependency versions.

4. **Rebuild lock file:**
   ```bash
   cd envs/benchmarks/cuda124-flash-attn/
   uv lock
   ```

5. **Build container:**
   ```bash
   singularity build singularity_uv-runtime.sif singularity_uv-runtime.def
   ```

---

## Troubleshooting

### Container fails to find Python

Ensure `uv sync --locked` completed successfully during build. Check `%post` section of the `.def` file.

### GPU not accessible

Verify `--nv` flag is passed to `singularity exec/run`. Ensure NVIDIA drivers are installed on the host.

### NCCL errors

Check `activate-env-variables-per-supercomputer.sh` for correct NCCL settings for your machine. Verify InfiniBand interfaces are accessible.

### Dependency conflicts

Run `uv lock --upgrade` to resolve conflicts. Check that `pyproject.toml` dependencies are compatible with the CUDA version.

---

## See Also

- [install/install-all-envs.sh](../install/install-all-envs.sh) — Install all Python environments with `uv`
- [install/build-all-singularity.sh](../install/build-all-singularity.sh) — Build all Singularity containers
- [envs/benchmarks/how-to-build.md](benchmarks/how-to-build.md) — Detailed build instructions for training environment
- [envs/cli/how-to-build.md](cli/how-to-build.md) — Detailed build instructions for CLI environment
- [configs_hydra/README.md](../configs_hydra/README.md) — Configuration system (references container path)
- [scripts/README.md](../scripts/README.md) — Training scripts (run inside containers)
- [scripts/slurm/README.md](../scripts/slurm/README.md) — SLURM submission (uses containers)
- [training_MN5/README.md](../../README.md) — Root project overview
