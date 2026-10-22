# Environment Management

This directory contains Python environment definitions for building Singularity containers used in training benchmarks. Environments are managed using [uv](https://github.com/astral-sh/uv), a fast Python package and project manager, and packaged into Singularity containers for execution on HPC clusters.

## Overview

Training benchmarks run inside Singularity containers that contain a complete Python environment with all dependencies. The containers are built from definition files (`.def`) using `pyproject.toml` and `uv.lock` for reproducible dependency resolution.

```
envs/
├── benchmarks/                    # Training benchmark 
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

### `uv/` — Primary Training Environments

The main environment system for training benchmarks. Each subdirectory targets a specific CUDA version with flash attention support.

#### Directory Structure

```
uv/
└── cuda121-flash-attn/              # CUDA 12.1 environment
    ├── pyproject.toml               # Python dependencies
    ├── uv.lock                      # Locked dependency versions
    ├── singularity_uv-devel.def     # Singularity definition (development)
    ├── singularity_uv-runtime.def   # Singularity definition (runtime)
    ├── singularity_uv.sif           # Built container (development)
    └── .venv/                       # Virtual environment (local, not committed)
```

#### CUDA 12.1

| Version | Base Image | Use Case |
|---------|------------|----------|
| `cuda121-flash-attn/` | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` | CUDA 12.1 environment for training benchmarks |

#### Container Features

The environment includes:

- **Python 3.11** — Installed via uv from source
- **uv package manager** — For dependency resolution and installation
- **CUDA toolkit** — `cuda-nvcc-12.1` for compilation
- **NCCL libraries** — `libnccl2`, `libnccl-dev` for multi-GPU communication
- **Build tools** — `build-essential`, `python3-dev`, `git`, `curl`
- **Flash attention** — Included via `pyproject.toml` dependencies
- **Training stack** — PyTorch, transformers, accelerate, trl, deepspeed, etc.

#### Singularity Definition Files

Two definition files are provided per CUDA version:

**`singularity_uv-runtime.def`** — Production runtime container (optimized for benchmark execution).

**`singularity_uv-devel.def`** — Development container (includes dev dependencies for debugging).

### `cli/` — CLI Environment

A separate environment for the `minerva-cli.sh` tool. Uses the same structure as `uv/`:

```
cli/
├── pyproject.toml         # CLI dependencies (click, prompt_toolkit, etc.)
├── uv.lock                # Locked dependencies
└── singularity-uv.def    # Singularity definition file

```

This environment is used by `minerva-cli.sh` to provide the Click-based CLI for benchmark submission.

---

## Building Containers

### Prerequisites

- Singularity/Apptainer installed on the build machine
- Docker/Podman available (Singularity builds from Docker images)
- Internet access for downloading base images and packages

### Build Commands

```bash
# From the training/ directory
cd install

# Install all Python environments (uv sync)
bash install-all-envs.sh

# Build all Singularity containers
bash build-all-singularity.sh
```

Both scripts can be run from either `training/` or `training/install/`.


---

## Directory Structure

### `benchmarks/cuda130-flash-attn/`

```
cuda130-flash-attn/
├── pyproject.toml               # Python dependencies (PyTorch, transformers, DeepSpeed, etc.)
├── uv.lock                      # Locked dependency versions
└── ingularity_uv-runtime.def   # Singularity definition (runtime image)
```

### `cli/`

```
cli/
├── pyproject.toml         # CLI dependencies (click, hydra, rich, etc.)
├── uv.lock                # Locked dependencies
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

### `pyproject.toml`

Defines project dependencies:

```toml
[project]
name = "minerva-training-benchmarks"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.38.0",
    "accelerate>=0.25.0",
    "trl>=0.7.0",
    "deepspeed>=0.14.0",
    "flash-attn>=2.5.0",
    "peft>=0.8.0",
    "datasets>=2.16.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "pandas>=2.1.0",
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.0",
    "rich>=13.0.0",
    "click>=8.1.0",
    "prompt-toolkit>=3.0.0",
]
```

### `uv.lock`

Locked dependency tree generated by `uv lock`. Ensures reproducible builds across environments.

### Updating Dependencies

```bash
# Add a new dependency
cd envs/benchmarks/cuda130-flash-attn/
uv add <package-name>

# Update all dependencies
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package <package-name>

### Rebuilding Containers

After updating dependencies, rebuild the Singularity container:

```bash
cd envs/benchmarks/cuda130-flash-attn/
singularity build singularity_uv-runtime.sif singularity_uv-runtime.def
```

---

## Adding a New CUDA Version (or ROCM version)

To add a new CUDA version (e.g., CUDA 13.3):

1. **Copy existing directory:**
   ```bash
   cp -r envs/benchmarks/cuda130-flash-attn envs/benchmarks/cuda130-flash-attn
   ```

2. **Update Singularity definition files:**
   - Change `From:` line to `nvidia/cuda:13.3.x-runtime-ubuntu22.04`
   - Update `cuda-nvcc-13-x` package name
   - Update `%environment` paths if needed

3. **Update pyproject.toml** if CUDA version affects dependency versions.

4. **Rebuild lock file:**
   ```bash
   cd envs/uv/cuda124-flash-attn/
   uv lock
   ```

5. **Build container:**
   ```bash
   singularity build singularity_uv-runtime.sif singularity_uv-runtime.def
   ```

6. **Update references** in config files (e.g., `configs_hydra/configs/base-MN5.yaml` → `singularity_container` path).

---

## Container Path Configuration

The Singularity container path is configured in:

- **`minerva-cli.sh`**: `CLI_CONTAINER_PATH` variable
- **`configs_hydra/configs/base-MN5.yaml`**: `machine.singularity_container` field

```yaml
machine:
  singularity_container: /path/to/singularity_uv-runtime.sif
```

---

A similar process should be followed to add any new environment necessary.

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
- [configs_hydra/README.md](../configs_hydra/README.md) — Configuration system (references container path)
- [scripts/README.md](../scripts/README.md) — Training scripts (run inside containers)
- [scripts/slurm/README.md](../scripts/slurm/README.md) — SLURM submission (uses containers)
- [training/README.md](../../README.md) — Root project overview
