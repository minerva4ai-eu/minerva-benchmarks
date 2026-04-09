# Leonardo Setup Guide — Fine-Tuning Benchmarks

**Date:** 2025-03-31  
**System:** CINECA Leonardo (NVIDIA A100 80GB, 4 GPUs/node, 32 CPUs/node)  
**Account:** `mnrva_bench`  
**Partition:** `boost_usr_prod`  
**Base path:** `/leonardo_work/cin_staff/dgentile/minerva-benchmarks`

---

## Prerequisites

1. **Clone the repository**
   ```bash
   cd /leonardo_work/cin_staff/dgentile
   git clone --branch leonardo --single-branch https://github.com/minerva4ai-eu/minerva-benchmarks.git
   ```

2. **Install Miniforge3** (conda package manager)
   ```bash
   cd /leonardo_work/cin_staff/dgentile
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh -b -p /leonardo_work/cin_staff/dgentile/miniforge3
   ```

---

## Step 1: Create `.env-leonardo`

Created `training/training_MN5/.env-leonardo` with Leonardo-specific SLURM and environment settings:

```bash
# ENVIRONMENT PATH
ENVIRONMENT_FINETUNING=/leonardo_work/cin_staff/dgentile/minerva-benchmarks/training/training_MN5/envs/fine-tune-dev

# ADDITIONAL VARIABLES
PORT=8000

# Modules to load for activating the environment
MODULES="cuda/12.3"

# SUPERCOMPUTER SPECIFIC
SUPCOMPUTER_NAME="Leonardo"
PARTITION_NAME="boost_usr_prod"
ACCOUNT=mnrva_bench
QOS=boost_qos_lprod
GPUS_PER_NODE=4
CPUS_PER_GPU=8
```

Key differences from MN5:
- `GPUS_PER_NODE=4` (MN5 has 4 as well, but partition/account differ)
- `CPUS_PER_GPU=8` (MN5 has 20, since it has 80 CPUs/node)
- `PARTITION_NAME="boost_usr_prod"` (Leonardo-specific)
- `QOS=boost_qos_lprod`

---

## Step 2: Create Leonardo-specific conda environment YAML

Created `training/training_MN5/envs-yaml/fine-tune-dev-env-leonardo.yaml` — a copy of `fine-tune-dev-env.yaml` with two fixes:

1. **Added PyTorch extra index URL** for CUDA 12.1 wheels:
   ```yaml
   - pip:
       - --extra-index-url https://download.pytorch.org/whl/cu121
   ```

2. **Removed `+cu121`/`+cu118` suffixes** from PyTorch packages (the extra-index-url handles this):
   ```yaml
   # Original (won't resolve with extra-index-url):
   - torch==2.5.1+cu121
   - torchaudio==2.5.1+cu118
   - torchvision==0.20.1+cu121

   # Fixed:
   - torch==2.5.1
   - torchaudio==2.5.1
   - torchvision==0.20.1
   ```

---

## Step 3: Create the conda environment

```bash
cd /leonardo_work/cin_staff/dgentile/minerva-benchmarks/training/training_MN5
eval "$(/leonardo_work/cin_staff/dgentile/miniforge3/bin/conda shell.bash hook)"
conda env create --prefix ./envs/fine-tune-dev -f envs-yaml/fine-tune-dev-env-leonardo.yaml
```

**Installed environment location:** `training/training_MN5/envs/fine-tune-dev/`

**Key package versions:**
| Package | Version |
|---|---|
| Python | 3.11.10 |
| PyTorch | 2.5.1+cu121 |
| CUDA (PyTorch) | 12.1 |
| Transformers | 4.57.0 |
| Accelerate | 1.10.1 |
| DeepSpeed | 0.15.4 |
| PEFT | 0.14.0 |
| Datasets | 3.2.0 |

---

## Step 4: Update activation scripts for Leonardo

### `scripts/activate-env-per-supercomputer.sh`

Replaced the placeholder `leonardo` case with actual conda activation logic:

```bash
leonardo)
    module load $MODULES
    eval "$(/leonardo_work/cin_staff/dgentile/miniforge3/bin/conda shell.bash hook)"
    conda activate $ENVIRONMENT
    export PATH=$ENVIRONMENT/bin:$PATH
    export CUDA_HOME=$CUDA_ROOT
    which python
    ;;
```

### `scripts/activate-env-variables-per-supercomputer.sh`

Replaced the placeholder `leonardo` case with NCCL/CUDA/PyTorch environment variables:

```bash
leonardo)
    # NCCL variables for Leonardo InfiniBand
    export NCCL_NET=IB
    export NCCL_DEBUG=TRACE
    export NCCL_NVLS_ENABLE=0
    export NCCL_IB_DISABLE=0

    # CUDA DEVICES
    export CUDA_VISIBLE_DEVICES="0,1,2,3"

    # PYTORCH
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_LAUNCH_BLOCKING=1
    ;;
```

---

## Step 5: Update configuration files

### `configs/config_datasets_paths_map.json`

Updated all dataset paths from BSC MN5 (`/gpfs/scratch/bsc99/...`) to Leonardo paths:

```json
{
  "alpaca": "/leonardo_work/cin_staff/dgentile/minerva-benchmarks/training/training_MN5/datasets/alpaca-cleaned/alpaca_data_cleaned.json",
  "squadv2": {
    "train": ".../datasets/squad_v2/squad_v2/train-00000-of-00001.parquet",
    "validation": ".../datasets/squad_v2/squad_v2/validation-00000-of-00001.parquet"
  }
}
```

### `configs/model_type_directories_map.json`

Updated model registry path:

```json
{
  "Text Generation": "/leonardo_work/cin_staff/dgentile/models_registry",
  "Embedding": "/leonardo_work/cin_staff/dgentile/models_registry",
  "Vision": "/leonardo_work/cin_staff/dgentile/models_registry"
}
```

---

## Step 6: Download datasets

Datasets were downloaded from HuggingFace using `huggingface-cli`:

```bash
# Alpaca Cleaned
huggingface-cli download yahma/alpaca-cleaned \
  --repo-type dataset \
  --local-dir datasets/alpaca-cleaned

# SQuAD v2
huggingface-cli download rajpurkar/SQuAD-explorer \
  --repo-type dataset \
  --local-dir datasets/squad_v2
```

**Dataset locations:**
- `training/training_MN5/datasets/alpaca-cleaned/alpaca_data_cleaned.json` (42 MB)
- `training/training_MN5/datasets/squad_v2/squad_v2/train-00000-of-00001.parquet` (16 MB)
- `training/training_MN5/datasets/squad_v2/squad_v2/validation-00000-of-00001.parquet` (1.3 MB)

---

## Step 7: Download model

Downloaded `meta-llama/Llama-3.1-8B-Instruct` (~15 GB) using `huggingface-cli`:

```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --repo-type model \
  --local-dir /leonardo_work/cin_staff/dgentile/models_registry/Llama-3.1-8B-Instruct
```

> **Note:** Requires a HuggingFace token with access to the Llama model. Token was already configured in `~/.cache/huggingface/token`.

---

## Step 8: Fix SLURM parameters in run scripts

### `run_1_benchmark.sh` and `run_all_benchmarks.sh`

Three changes applied to both files:

1. **Set `MACHINE="leonardo"`** (was `"bsc-mn5-acc"`):
   ```bash
   MACHINE="leonardo"
   ```

2. **Fixed hardcoded `--cpus-per-task=80`** in the Accelerate sbatch block (line ~232). MN5 has 80 CPUs/node; Leonardo has 32. Changed to use the variable:
   ```bash
   # Before:
   --cpus-per-task=80
   # After:
   --cpus-per-task=$TOTAL_CPUS
   ```
   (`$TOTAL_CPUS` = `$GPUS_PER_NODE * $CPUS_PER_GPU` = 4 × 8 = 32)

3. **Added `-p $PARTITION_NAME`** to both sbatch invocations (torchrun and accelerate):
   ```bash
   -A $ACCOUNT \
   -p $PARTITION_NAME \    # <-- added
   -q $QOS \
   ```

---

## Summary of all files changed/created

### New files created
| File | Description |
|---|---|
| `training/training_MN5/.env-leonardo` | Leonardo environment variables |
| `training/training_MN5/envs-yaml/fine-tune-dev-env-leonardo.yaml` | Conda env spec with PyTorch index URL fix |
| `training/training_MN5/envs/fine-tune-dev/` | Installed conda environment (not tracked in git) |
| `training/training_MN5/datasets/alpaca-cleaned/` | Alpaca Cleaned dataset |
| `training/training_MN5/datasets/squad_v2/` | SQuAD v2 dataset |

### Modified files
| File | Change |
|---|---|
| `training/training_MN5/run_1_benchmark.sh` | MACHINE, cpus-per-task, partition flag |
| `training/training_MN5/run_all_benchmarks.sh` | MACHINE, cpus-per-task, partition flag |
| `training/training_MN5/scripts/activate-env-per-supercomputer.sh` | Leonardo conda activation logic |
| `training/training_MN5/scripts/activate-env-variables-per-supercomputer.sh` | Leonardo NCCL/CUDA env vars |
| `training/training_MN5/configs/config_datasets_paths_map.json` | Leonardo dataset paths |
| `training/training_MN5/configs/model_type_directories_map.json` | Leonardo model registry path |

### External assets (outside repo)
| Path | Description |
|---|---|
| `/leonardo_work/cin_staff/dgentile/miniforge3/` | Miniforge3 installation |
| `/leonardo_work/cin_staff/dgentile/models_registry/Llama-3.1-8B-Instruct/` | Llama 3.1 8B model weights |

---

## Running a benchmark

```bash
cd /leonardo_work/cin_staff/dgentile/minerva-benchmarks/training/training_MN5
bash run_1_benchmark.sh
```

This submits SLURM jobs for Llama-3.1-8B-Instruct fine-tuning with FSDP on 4 nodes, testing batch sizes 4/8/16 in bf16 precision.

---

## Notes: Making `generateSummaryTable.py` work on Leonardo

Two things were needed for reliable summary generation.

1. **Run with the benchmark conda environment active**

```bash
cd /leonardo_work/cin_staff/dgentile/minerva-benchmarks/training/training_MN5
eval "$(/leonardo_work/cin_staff/dgentile/miniforge3/bin/conda shell.bash hook)"
conda activate ./envs/fine-tune-dev
python generateSummaryTable.py
```

If `python: command not found` appears, the conda environment is not activated.

2. **Ensure Leonardo env vars are loaded in the shell**

`generateSummaryTable.py` reads `SUPCOMPUTER_NAME` and `PARTITION_NAME` from environment variables. If needed, export them from `.env-leonardo` before running:

```bash
set -a
source .env-leonardo
set +a
python generateSummaryTable.py
```

This produces:

- `results/full_benchmark_training_summary_Leonardo_boost_usr_prod.csv`

### Important bug fix applied

A timing bug in Accelerate FSDP produced negative values in summaries when `max_steps=-1` (Hugging Face default).

File fixed:

- `training/training_MN5/scripts/accelerate-common/finetune-fsdp.py`

Change applied:

```python
# before
if training_args.max_steps:

# after
if training_args.max_steps and training_args.max_steps > 0:
```

Why: `-1` is truthy in Python, so the old condition wrongly computed per-step time as `total_time / -1`.

After this fix, future runs produce correct timing metrics in the JSON summaries and final CSV.
