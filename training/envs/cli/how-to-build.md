# How to Build — CLI Environment

This directory contains the Singularity image definition for the MINERVA training CLI tool. The image bundles a minimal Python 3.11 environment with all dependencies needed to run the `minerva-cli.sh` script.

## Directory Structure

```
cli/
├── pyproject.toml         # CLI dependencies
├── uv.lock                # Locked dependencies
└── singularity-uv.def     # Singularity definition file
```

## Prerequisites

- **Singularity** (or Apptainer) installed and accessible via `module load singularity` or directly in PATH
- **Docker** installed and running (Singularity uses Docker as the bootstrap source)
- **uv** installed (for local development)
- Sufficient disk space (~5–10 GB recommended)

---

## Automated Installation (Recommended)

Instead of following the steps below manually, you can use the automated scripts:

```bash
# From the training/ directory
cd install

# Install all Python environments (including CLI)
bash install-all-envs.sh

# Build all Singularity containers (including CLI)
bash build-all-singularity.sh
```

---

## Option 1: Local Development with `uv`

Use this option to develop or test the CLI locally before building the Singularity image.

### Step 1 — Install `uv`

```bash
# Using pip
pip install uv

# Or using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2 — Create and activate a local virtual environment

```bash
# Create and activate the environment
uv venv
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv sync --locked

# Verify
python -c "import hydra; print(hydra.__version__)"
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

### Step 1 — Build the Singularity image

```bash
singularity build singularity-uv.sif singularity-uv.def
```

This will:
1. Pull the `python:3.11-slim` Docker base image
2. Install system dependencies (`git`, `curl`, `build-essential`)
3. Install `uv` via `pip`
4. Run `uv sync --locked` to install all Python dependencies from `pyproject.toml`
5. Produce `singularity-uv.sif`

### Step 2 — Verify the image

```bash
singularity exec singularity-uv.sif python --version
singularity exec singularity-uv.sif uv --version
```

---

## Dependencies

See `pyproject.toml` for the full list:

| Category | Packages |
|----------|----------|
| Data | `numpy`, `pandas`, `seaborn`, `matplotlib` |
| Config | `pyyaml`, `omegaconf`, `hydra-core>=1.3.2` |
| CLI/UX | `click>=8.4.1`, `rich>=15.0.0`, `prompt-toolkit`, `tqdm` |
| Utilities | `python-dotenv`, `psutil`, `setuptools` |

---

## Notes

- The Singularity image uses `python:3.11-slim` as the base — no CUDA libraries are included (this is a CLI-only environment).
- The `uv.lock` file pins exact versions for reproducible builds. Always keep it in sync with `pyproject.toml`.
- To update dependencies: `uv sync --locked` (after editing `pyproject.toml`).
