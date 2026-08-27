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

# Or maybe 'uv' is already installed in the form of a module e.g.
module load uv
```

### Step 2 — Create and activate a local virtual environment

```bash
# make sure you are inside folder 'training/envs/cli'
cd training/envs/cli

# Install dependencies from pyproject.toml
uv sync --locked

# Verify
python -c "import omegaconf; print(omegaconf.__version__)"
python -c "import hydra; print(hydra.__version__)"
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

