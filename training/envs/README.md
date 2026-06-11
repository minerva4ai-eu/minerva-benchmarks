# Environment Management

This directory contains Python environment definitions for building Singularity containers used in training benchmarks. Environments are managed using [uv](https://github.com/astral-sh/uv), a fast Python package and project manager, and packaged into Singularity containers for execution on HPC clusters.

## Overview

Training benchmarks run inside Singularity containers that contain a complete Python environment with all dependencies. The containers are built from definition files (`.def`) using `pyproject.toml` and `uv.lock` for reproducible dependency resolution.

```
envs/
├── uv/                              # Primary environment system
│   └── cuda121-flash-attn/          # CUDA 12.1 + flash attention
└── cli/                             # CLI environment (minerva-cli.sh)
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
# Build runtime container
cd envs/uv/cuda121-flash-attn/
singularity build singularity_uv-runtime.sif singularity_uv-runtime.def

# Build development container
singularity build singularity_uv-devel.sif singularity_uv-devel.def
```

### Build Process

The Singularity definition file executes these steps in `%post`:

1. **Install system dependencies** — Python dev headers, build tools, NCCL, CUDA toolkit
2. **Install uv** — Via pip
3. **Sync Python packages** — `uv sync --locked` reads `pyproject.toml` + `uv.lock`
4. **Create symlinks** — Python 3.11 linked to `/usr/local/bin/python`
5. **Set environment** — PATH, locale, uv project environment

### Output

The build produces a `.sif` (Singularity Image Format) file:

```
singularity_uv-runtime.sif   # ~6 GB depending on dependencies
```

---

## Using Containers

### Via CLI

The `minerva-cli.sh` wrapper automatically runs commands inside the Singularity container:

```bash
bash minerva-cli.sh run --config-name base-MN5
```


### Bind Mounts

Key bind mounts used by `minerva-cli.sh`:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `$HOME` | `$HOME` | User home directory |
| `$PWD` | `$PWD` | Current working directory |
| `/etc/passwd` | `/etc/passwd` | User identity |
| `/etc/group` | `/etc/group` | Group membership |
| `$(which sbatch)` | `/usr/local/bin/sbatch` | SLURM job submission |
| `$(which sacct)` | `/usr/local/bin/sacct` | SLURM account info |
| `/var/run/munge` | `/var/run/munge` | SLURM authentication socket |
| `/etc/munge` | `/etc/munge` | Munge config |
| `/etc/slurm` | `/etc/slurm` | SLURM daemon config |
| `/usr/lib64/slurm` | `/usr/lib64/slurm` | SLURM libraries |
| `/usr/lib64/libmunge.so.2` | `/usr/lib64/libmunge.so.2` | Munge library |
| `/lib64/libc.so.6` | `/lib64/libc.so.6` | C library |
| `/lib64/libm.so.6` | `/lib64/libm.so.6` | Math library |
| `/lib64/libresolv.so.2` | `/lib64/libresolv.so.2` | DNS resolver library |

### GPU Access

The `--nv` flag enables NVIDIA GPU access inside the container:

```bash
singularity exec --nv ...  # Enables CUDA, cuDNN, NCCL
```

For AMD ROCm GPUs, use `--rocm` instead (if supported by the container).

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
cd envs/uv/cuda121-flash-attn/
uv add <package-name>

# Update all dependencies
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package <package-name>

# Rebuild container
singularity build singularity_uv-runtime.sif singularity_uv-runtime.def
```

---

## Adding a New CUDA Version

To add a new CUDA version (e.g., CUDA 12.4):

1. **Copy existing directory:**
   ```bash
   cp -r envs/uv/cuda121-flash-attn envs/uv/cuda124-flash-attn
   ```

2. **Update Singularity definition files:**
   - Change `From:` line to `nvidia/cuda:12.4.x-runtime-ubuntu22.04`
   - Update `cuda-nvcc-12-x` package name
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

- [configs_hydra/README.md](../configs_hydra/README.md) — Configuration system (references container path)
- [scripts/README.md](../scripts/README.md) — Training scripts (run inside containers)
- [scripts/slurm/README.md](../scripts/slurm/README.md) — SLURM submission (uses containers)
- [training_MN5/README.md](../../README.md) — Root project overview
