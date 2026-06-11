# SLURM Job Submission CLI

This subpackage provides a Click-based command-line interface for generating, validating, and submitting LLM training benchmark jobs to SLURM clusters. It is the primary entry point for running benchmarks.

## Overview

The SLURM subpackage automates the entire job lifecycle:

```
Config Generation → Combo Validation → Launch Folder Creation → Script Staging → sbatch Submission → Job Monitoring
```

## CLI Entry Point

The CLI is invoked via the shell wrapper `minerva-cli.sh`:

```bash
./minerva-cli.sh <command> [options]
```

This script wraps the Python CLI inside a Singularity container with the necessary bind mounts:

```bash
singularity exec --env CWD="$PWD" \
    --bind "$HOME":"$HOME" \
    --bind "$PWD":"$PWD" \
    --bind /etc/passwd:/etc/passwd \
    --bind /etc/group:/etc/group \
    --bind $(which sbatch):/usr/local/bin/sbatch \
    --bind $(which sacct):/usr/local/bin/sacct \
    --bind /var/run/munge:/var/run/munge \
    --bind /etc/munge:/etc/munge \
    --bind /etc/slurm:/etc/slurm \
    --bind /usr/lib64/slurm:/usr/lib64/slurm \
    --bind /usr/lib64/libmunge.so.2:/usr/lib64/libmunge.so.2 \
    --bind /lib64/libc.so.6:/lib64/libc.so.6 \
    --bind /lib64/libm.so.6:/lib64/libm.so.6 \
    --bind /lib64/libresolv.so.2:/lib64/libresolv.so.2 \
    "$CLI_CONTAINER_PATH" bash -c "cd $PWD && python -m scripts.slurm.cli $cli_args"
```

**Bind mounts ensure the container can:**
- Access the user's home directory and current working directory
- Execute SLURM commands (`sbatch`, `sacct`)
- Communicate with the SLURM daemon via munge socket and SLURM config files

## Commands

### `run` — Generate and Submit Jobs

```bash
./minerva-cli.sh run [OPTIONS]
```

Generates all valid benchmark configurations and submits them as SLURM jobs.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | `False` | Generate configs and build launch folders without submitting jobs |
| `--configs-path` | `./configs_hydra/configs` | Path to the Hydra config directory |
| `--config-name` | `base` | Base config name to compose (e.g., `base`, `base-MN5`) |
| `--output` | `benchmark-runs/` | Output directory for generated configs and results |

**Example:**

```bash
# Generate and submit all valid jobs
./minerva-cli.sh run

# Dry run — generate configs without submitting
./minerva-cli.sh run --dry-run

# Use MN5-specific config
./minerva-cli.sh run --config-name base-MN5

# Custom output directory
./minerva-cli.sh run --output /path/to/results/
```

### `monitor` — Monitor Running Jobs

```bash
./minerva-cli.sh monitor [OPTIONS]
```

Monitors the status of submitted SLURM jobs with a color-coded dashboard.

## Subpackage Structure

```
scripts/slurm/
├── __init__.py
├── cli.py              # Click CLI group with run/monitor commands
├── submitter.py        # Job submission logic: folder creation, script copying, env building
├── monitor.py          # SLURM state dashboard with status icons
└── utils.py            # ANSI colors, Unicode icons, JSONL I/O, YAML loading
```

---

## `cli.py` — Click CLI

### CLI Group

```python
@click.group()
def cli():
    pass
```

### `run` Command

```python
@cli.command()
@click.option("--dry-run", is_flag=True)
@click.option("--configs-path", default="./configs_hydra/configs")
@click.option("--config-name", default="base")
@click.option("--output", default="benchmark-runs/")
def run(dry_run, configs_path, config_name, output):
    """Generate all valid configs and submit all pending jobs."""
```

**Detailed execution flow:**

1. **Generate valid configs**: Calls `generate_valid_combos()` from `configs_hydra.hydra_app` to produce `(valid, skipped)` tuples. This iterates over all `(model, framework, dataset)` combinations, validates them with constraint rules, and expands training hyperparameters.

2. **Create monitor directory**: Creates a unique run ID and monitor directory: `benchmark-runs/slurm-monitor/{date}/run_id-{N}/`. This directory tracks all jobs submitted in this run.

3. **For each valid config** (in order):
   a. **`build_launch_folder()`** — Creates the experiment directory structure:
      ```
      benchmark-runs/{machine}/{date}/{model}/{framework}/{dataset}/nodes-{N}/run_id-{N}/launch-{R}/
      ```
      Returns the `Path` to the launch folder.
   
   b. **`copy_scripts()`** — Stages all necessary scripts to the launch folder:
      - Copies `shared/` directory (framework-agnostic code: `custom_train.py`, `data.py`, `utils.py`, `datasets/`)
      - Copies the framework-specific run script (`cfg.framework.scripts.run`)
      - Copies the framework-specific finetune script (`cfg.framework.scripts.finetune`)
      - Copies all files listed in `cfg.framework.scripts.copy_files` (gpu_monitor, utils, etc.)
   
   c. **`build_env()`** — Constructs the environment dictionary (see below for full list of 25+ variables).
   
   d. **Submit job** — If not `--dry-run`, submits via `sbatch`. In dry-run mode, only the YAML config is saved without submitting.

4. **Track job IDs**: All submitted job IDs are recorded in `jobs_submitted.jsonl` for dependency management and monitoring.

**Job dependency chaining**: When submitting multiple jobs, the system can chain dependencies using `--dependency=afterok:<job_id>` to ensure jobs run in a specific order (e.g., per-model-jobs mode where each model's jobs must complete before the next model starts).

**Repeat mechanism**: If `cfg.experiment.repeat` is set to N > 1, the system submits N identical jobs with different `REPEAT_ID` values (0 to N-1) to account for run-to-run variability.

**Dry-run vs actual submission**:
- **Dry-run** (`--dry-run`): Generates configs, builds launch folders, copies scripts, saves YAML configs — but does NOT submit any jobs. Useful for validating the config generation pipeline.
- **Actual submission** (no flag): Does everything in dry-run mode PLUS submits all jobs via `sbatch`.

**Tracking file**: `jobs_submitted.jsonl` records each submitted job with:
```json
{"job_id": "12345", "model": "llama3_8b", "framework": "accelerate", "parallelism": "ddp", "dataset": "alpaca", "config_path": "benchmark-runs/.../config.yaml"}
```

### `monitor` Command

Provides real-time status updates on running jobs using the SLURM state dashboard from `monitor.py`.

---

## `submitter.py` — Job Submission Logic

### `build_launch_folder(cfg, base_dir, runs_dir, run_id, dry, repeat_id)`

Creates the per-experiment directory structure:

```
{runs_dir}/{machine}/{date}/{model}/{framework}/{dataset}/nodes-{N}/run_id-{N}/launch-{R}/
```

**Returns:** `Path` to the launch folder (or YAML config path in dry-run mode).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfg` | `BenchmarkConfig` | Composed benchmark configuration |
| `base_dir` | `Path` | Base directory of the project |
| `runs_dir` | `Path` | Output directory for benchmark results |
| `run_id` | `str` | Unique run identifier (e.g., `run_id-1`) |
| `dry` | `bool` | If True, only generates YAML config without creating launch folder |
| `repeat_id` | `int` | Repeat iteration number |

### `copy_scripts(cfg, dest)`

Copies all necessary scripts to the launch folder:

1. Copies the `shared/` directory (framework-agnostic code).
2. Copies the framework-specific run script (`cfg.framework.scripts.run`).
3. Copies the framework-specific finetune script (`cfg.framework.scripts.finetune`).
4. Copies all files listed in `cfg.framework.scripts.copy_files` (gpu_monitor, utils, etc.).

### `build_env(cfg, launch_folder, run_id)`

Constructs the environment dictionary passed to the SLURM job. Includes:

| Variable | Source | Description |
|----------|--------|-------------|
| `MODULES` | `cfg.machine.modules` | SLURM modules to load |
| `SINGULARITY_CONTAINER` | `cfg.machine.singularity_container` | Path to Singularity image |
| `SINGULARITY_BINDS` | `cfg.machine.singularity_binds` | Bind mount arguments |
| `SINGULARITY_ARGS` | `cfg.machine.singularity_args` | Singularity runtime args (e.g., `--nv`) |
| `NODES` | `cfg.slurm.sbatch.nodes` | Number of compute nodes |
| `GPUS_PER_NODE` | `cfg.slurm.sbatch.gpus_per_node` | GPUs per node |
| `GPU_NODE` | `nodes × gpus_per_node` | Total GPU count |
| `FRAMEWORK` | `cfg.framework.name` | Framework name |
| `DATASET` | `cfg.dataset.name` | Dataset identifier |
| `DATASET_PATH` | `cfg.dataset.path` | Dataset file path |
| `MODEL` | `cfg.model.name` | Model identifier |
| `MODEL_PATH` | `cfg.model.path` | Model weights path |
| `PARALLELISM` | `cfg.framework.parallelism_name` | Parallelism strategy |
| `PRECISION` | Training config | Precision type (bf16, fp16, etc.) |
| `BATCH_SIZE` | Training config | Batch size |
| `GRAD_ACCUM` | Training config | Gradient accumulation steps |
| `MAX_MODEL_LENGTH` | `cfg.dataset.max_seq_len` | Max sequence length |
| `LR` | Training config | Learning rate |
| `STEPS` | Training config | Max training steps |
| `EPOCHS` | Training config | Training epochs |
| `REPEAT_ID` | `run_id` | Repeat iteration number |
| `MACHINE` | `cfg.machine.name` | Machine name |
| `TRAIN_SCRIPT` | Framework script name | Training script filename |
| `ENVIRONMENT_FINETUNING` | Machine config | Python environment path |
| `ZERO_STAGE` | Framework config | DeepSpeed ZeRO stage (if applicable) |

---

## `monitor.py` — SLURM State Dashboard

Provides a comprehensive status dashboard for SLURM jobs with emoji icons and color coding.

### Status States

| State | Code | Icon | Description |
|-------|------|------|-------------|
| `pending` | PD | ⏳ | Queued and waiting for resources |
| `running` | R | 🏃 | Actively executing on compute nodes |
| `completing` | CG | 🚶 | Finishing up and cleaning up |
| `suspended` | S | ⏸️ | Paused, cores released |
| `stopped` | ST | 🛑 | Paused, retaining cores |
| `preempted` | PR | 💥 | Evicted by higher priority job |
| `requeued` | RQ | 🔄 | Kicked out, returned to queue |
| `completed` | CD | ✅ | Finished successfully (exit 0) |
| `failed` | F | ❌ | Terminated with non-zero exit code |
| `timeout` | TO | ⏰ | Killed for exceeding wall-clock limit |
| `cancelled` | CA | 🚫 | Manually killed via scancel |
| `out_of_memory` | OOM | 🚨 | Terminated for exceeding RAM limit |
| `node_fail` | NF | 💀 | Node failure |

---

## `utils.py` — Utilities

### ANSI Colors and Unicode Icons

Provides consistent color coding and iconography across all CLI output:

```python
# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[90m"
RESET = "\033[0m"

# Icons
SUCCESS = "✓"
FAILURE = "✗"
WARNING = "⚠"
INFO = "ℹ"
PROGRESS = "⟳"
POINT_DIAMOND = "◆"
ARROW_RIGHT = "→"
```

### JSONL I/O

```python
def write_jsonl(d: list[dict], p: str):
    """Write list of dicts to JSON Lines file."""

def read_jsonl(path: str) -> list[dict]:
    """Read JSON Lines file into list of dicts."""
```

### YAML Loading

```python
def load_yaml(filepath):
    """Load and parse a YAML file. Returns empty dict on error."""
```

---

## Job Lifecycle

```
1. User runs: ./minerva-cli.sh run
        ↓
2. generate_valid_combos() → (valid_configs, skipped_reasons)
        ↓
3. For each valid config:
   a. build_launch_folder() → creates directory structure
   b. copy_scripts() → stages scripts to launch folder
   c. build_env() → constructs environment dict
   d. sbatch submission (or dry-run YAML save)
        ↓
4. Jobs execute on SLURM cluster
   - GPU monitoring runs in background
   - Training scripts execute
   - Results saved to output directory
        ↓
5. ./minerva-cli.sh monitor → displays job status dashboard
```

## Directory Structure Created

```
benchmark-runs/
└── {machine}/                    # e.g., bsc-mn5-acc
    └── {date}/                   # e.g., 10-06-2026
        └── slurm-monitor/
            └── {date}/
                └── run_id-{N}/
        └── {model}/{framework}/{dataset}/
            └── nodes-{N}/
                └── run_id-{N}/
                    └── launch-{R}/
                        ├── run-{parallelism}.sh      # Launcher script
                        ├── finetune-{parallelism}.py  # Training entry point
                        ├── gpu_monitor.py             # GPU metrics
                        ├── utils.py                   # Shared utilities
                        ├── shared/                    # Shared code directory
                        │   ├── custom_train.py
                        │   ├── data.py
                        │   └── utils.py
                        └── output/                    # Training outputs
                            ├── logs/
                            ├── checkpoints/
                            └── metrics.json
```

## See Also

- [configs_hydra/README.md](../configs_hydra/README.md) — Configuration system
- [scripts/README.md](../README.md) — Training scripts overview
- [training_MN5/README.md](../../README.md) — Root project overview
