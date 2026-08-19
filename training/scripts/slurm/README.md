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
bash minerva-cli.sh <command> [options]
```

The script supports two execution modes controlled by the `USE_SINGULARITY` variable (default: `0`):

### Singularity Mode (`USE_SINGULARITY=1`)

Wraps the Python CLI inside a Singularity container with the necessary bind mounts:

```bash
singularity exec --env CWD="$PWD" \
    --bind "$HOME":"/tmp_home" \
    --bind "$PWD":"$PWD" \
    --bind /etc/passwd:/etc/passwd \
    --bind /etc/group:/etc/group \
    --bind $(which sbatch):/usr/local/bin/sbatch \
    --bind $(which sacct):/usr/local/bin/sacct \
    --bind $(which scancel):/usr/local/bin/scancel \
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
- Execute SLURM commands (`sbatch`, `sacct`, `scancel`)
- Communicate with the SLURM daemon via munge socket and SLURM config files

### Virtualenv Mode (`USE_SINGULARITY=0`, default)

Activates the local Python virtual environment and runs the CLI directly:

```bash
source "envs/cli/.venv/bin/activate"
python -m scripts.slurm.cli $cli_args
```

## Commands

### `run` — Generate and Submit Jobs

```bash
bash minerva-cli.sh run [OPTIONS]
```

Generates all valid benchmark configurations and submits them as SLURM jobs.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | `False` | Generate configs and build launch folders without submitting jobs |
| `--per-model-jobs` | `False` | Chain job dependencies per model (each model's jobs complete before the next starts) |
| `--configs-path` | `./configs_hydra/configs` | Path to the Hydra config directory |
| `--config-name` | `base` | Base config name to compose (e.g., `base`, `base-MN5`) |
| `--runs-dir` | `benchmark-runs/` | Output directory for generated configs and results |
| `--yaml` | `None` | Run a specific benchmark configuration by providing the path to a BenchmarkConfig YAML file. Can be repeated for multiple configs, e.g., `--yaml path1.yaml --yaml path2.yaml` |

**Examples:**

```bash
# Generate and submit all valid jobs
bash minerva-cli.sh run

# Dry run — generate configs without submitting
bash minerva-cli.sh run --dry-run

# Use MN5-specific config
bash minerva-cli.sh run --config-name base-MN5

# Custom output directory
bash minerva-cli.sh run --runs-dir /path/to/results/

# Run specific YAML configs
bash minerva-cli.sh run --yaml config1.yaml --yaml config2.yaml

# Per-model job dependency chaining
bash minerva-cli.sh run --per-model-jobs
```

### `rerun` — Rerun Jobs

```bash
bash minerva-cli.sh rerun --run-date DD-MM-YYYY --run-id N [OPTIONS]
```

Reruns jobs from a previous run. Supports rerunning all, only failed, or only pending jobs.

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--run-date` | Yes | Date of the original run in `DD-MM-YYYY` format |
| `--run-id` | Yes | Serial ID of the run on the provided date |
| `--runs-dir` | No | Output directory (default: `benchmark-runs/`) |
| `--all` | No | Rerun all jobs from the run |
| `--only-failed` | No | Rerun only failed jobs |
| `--only-pending` | No | Rerun only pending jobs |
| `--yaml` | No | Re-run specific YAML configs. Rerun reuses scripts from the original run-id without copying them again. |

**Examples:**

```bash
# Rerun all failed jobs from run_id-1 on 29-06-2026
bash minerva-cli.sh rerun --run-date 29-06-2026 --run-id 1 --only-failed

# Rerun all jobs from a specific run
bash minerva-cli.sh rerun --run-date 29-06-2026 --run-id 1 --all

# Rerun specific YAML configs
bash minerva-cli.sh rerun --run-date 29-06-2026 --run-id 1 --yaml config.yaml
```

### `status` — Check Job Status

```bash
bash minerva-cli.sh status --run-date DD-MM-YYYY --run-id N [OPTIONS]
```

Prints a summary of all run statuses with optional filtering by model, framework, parallelism, nodes, or SLURM state.

**Options:**

| Option | Description |
|--------|-------------|
| `--run-date` | Date of the run (`DD-MM-YYYY`) |
| `--run-id` | Serial ID of the run |
| `--rerun-id` | Optional: check status of a specific rerun |
| `--runs-dir` | Output directory (default: `benchmark-runs/`) |
| `--model` | Filter by model name(s), space-separated |
| `--framework` | Filter by framework name(s), space-separated |
| `--parallelism-type` | Filter by parallelism type(s), space-separated |
| `--nodes` | Filter by number of nodes (exact match), space-separated |
| `--state` | Filter by SLURM job state (e.g., `running`, `pending`, `failed`) |

**Examples:**

```bash
# Status of all jobs in run_id-1
bash minerva-cli.sh status --run-date 29-06-2026 --run-id 1

# Filter by model and state
bash minerva-cli.sh status --run-date 29-06-2026 --run-id 1 --model llama3-7b --state running

# Check a specific rerun
bash minerva-cli.sh status --run-date 29-06-2026 --run-id 1 --rerun-id 2
```

### `cancel` — Cancel Jobs

```bash
bash minerva-cli.sh cancel --run-date DD-MM-YYYY --run-id N [OPTIONS]
```

Cancels all running and pending jobs for a given run. Supports filtering by model, framework, parallelism, and nodes.

**Options:**

| Option | Description |
|--------|-------------|
| `--run-date` | Date of the run (`DD-MM-YYYY`) |
| `--run-id` | Serial ID of the run |
| `--runs-dir` | Output directory (default: `benchmark-runs/`) |
| `--model` | Filter by model name(s), space-separated |
| `--framework` | Filter by framework name(s), space-separated |
| `--parallelism-type` | Filter by parallelism type(s), space-separated |
| `--nodes` | Filter by number of nodes (exact match), space-separated |

**Example:**

```bash
# Cancel all running/pending jobs in run_id-1
bash minerva-cli.sh cancel --run-date 29-06-2026 --run-id 1

# Cancel only llama3-7b jobs
bash minerva-cli.sh cancel --run-date 29-06-2026 --run-id 1 --model llama3-7b
```

### Interactive Mode

Run the CLI without arguments to enter interactive mode:

```bash
bash minerva-cli.sh
```

This launches a REPL loop with prompt-based argument collection:

```
MINERVA benchmarks CLI — interactive mode
Type 'help' for available commands, 'quit' to exit.

minerva-benchmarks > run

  -- run options --
    configs-path (default='./configs_hydra/configs'): 
    config-name (default='base'): 
    runs-dir (default='benchmark-runs/'): 
    dry-run? [y/N]: n
    per-model-jobs? [y/N]: n
```

Available interactive commands: `run`, `rerun`, `status`, `cancel`, `help`, `quit`/`exit`.

You can also pass a single command as an argument to enter interactive mode for that command:

```bash
bash minerva-cli.sh run
```

## Subpackage Structure

```
scripts/slurm/
├── __init__.py
├── cli.py              # Click CLI group with run/rerun/status/cancel commands + interactive mode
├── cli_utils.py        # Interactive prompt utilities, OptionConfig dataclasses, argument parsing
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
@click.option("--per-model-jobs", is_flag=True)
@click.option("--configs-path", default="./configs_hydra/configs")
@click.option("--config-name", default="base")
@click.option("--runs-dir", default="benchmark-runs/")
@click.option("--yaml", "yamls", multiple=True, default=None,
    help='Run a benchmark configuration by providing the path to BenchmarkConfig file.')
def run(dry_run, per_model_jobs, configs_path, config_name, runs_dir, yamls):
    """Generate all valid configs and submit all pending jobs."""
```

**Detailed execution flow:**

1. **Load configurations**: Either loads YAML configs from `--yaml` paths, or calls `generate_valid_combos()` from `configs_hydra.hydra_app` to produce `(valid, skipped)` tuples. This iterates over all `(model, framework, dataset)` combinations, validates them with constraint rules, and expands training hyperparameters.

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

**Job dependency chaining (`--per-model-jobs`)**: When submitting multiple jobs, the system can chain dependencies using `--dependency=afterok:<job_id>` to ensure jobs run in a specific order. With `--per-model-jobs`, jobs are grouped by model and each model's jobs must complete before the next model starts.

**Repeat mechanism**: If `cfg.experiment.repeat` is set to N > 1, the system submits N identical jobs with different `REPEAT_ID` values (1 to N) to account for run-to-run variability.

**Dry-run vs actual submission**:
- **Dry-run** (`--dry-run`): Generates configs, builds launch folders, copies scripts, saves YAML configs — but does NOT submit any jobs. Useful for validating the config generation pipeline.
- **Actual submission** (no flag): Does everything in dry-run mode PLUS submits all jobs via `sbatch`.

**Tracking file**: `jobs_submitted.jsonl` records each submitted job with:
```json
{"job_id": "12345", "cfg_id": "llama3_8b/accelerate/ddp/alpaca", "dependency": "12344", "launch_folder": "benchmark-runs/.../launch-1", "yaml_filename": "config.yaml"}
```

### `rerun` Command

```python
@cli.command()
@click.option("--run-date", required=True)
@click.option("--run-id", required=True)
@click.option("--runs-dir", default="benchmark-runs/")
@click.option("--all", is_flag=True)
@click.option("--only-failed", is_flag=True)
@click.option("--only-pending", is_flag=True)
@click.option("--yaml", "yamls", multiple=True, default=None)
def rerun(run_date, run_id, output, all, only_failed, only_pending, yamls):
    """Rerun all, failed, pending jobs, or a specific run/combo by id."""
```

Reruns jobs from a previous run. Unlike `run`, rerun reuses scripts from the original run-id without copying them again. Supports filtering by `--all`, `--only-failed`, `--only-pending`, or specific `--yaml` configs.

**Rerun tracking**: Resubmitted jobs are logged in `jobs_resubmitted-rerun_id-{N}.jsonl` within the original run's monitor directory.

### `status` Command

```python
@cli.command()
@click.option("--run-date", required=True)
@click.option("--run-id", required=True)
@click.option("--rerun-id", required=False)
@click.option("--runs-dir", default="benchmark-runs/")
@click.option("--model", default=None)
@click.option("--framework", default=None)
@click.option("--parallelism-type", default=None)
@click.option("--nodes", default=None)
@click.option("--state", default=None)
def status(run_date, run_id, rerun_id, runs_dir, model, framework, parallelism, nodes, state):
    """Print a summary of all run statuses."""
```

Prints a summary of all run statuses with optional filtering. Config-based filtering (model, framework, parallelism, nodes) uses AND logic — a job must match ALL specified filters. State filtering uses SLURM job states.

### `cancel` Command

```python
@cli.command()
@click.option("--run-date", required=True)
@click.option("--run-id", required=True)
@click.option("--runs-dir", default="benchmark-runs/")
@click.option("--model", default=None)
@click.option("--framework", default=None)
@click.option("--parallelism-type", default=None)
@click.option("--nodes", default=None)
def cancel(run_date, run_id, runs_id, model, framework, parallelism, nodes):
    """Cancel all running and pending jobs for a given run."""
```

Cancels all running and pending jobs for a given run using `scancel`. Supports the same filtering options as `status`. Only jobs in `running` or `pending` state are cancelled; others are skipped with a message.

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
| `repeat_id` | `int` | Repeat iteration number (required when not in dry-run mode) |

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

## `cli_utils.py` — Interactive Prompt Utilities

Provides interactive prompt-based argument collection using `prompt_toolkit`.

### Key Components

**`OptionConfig` dataclass**: Defines CLI options for interactive prompting:

```python
@dataclass
class OptionConfig:
    name: str                    # CLI flag, e.g. "--configs-path"
    prompt: str                  # Text shown to user
    default: Optional[str] = None
    required: bool = False
    validator: Optional[Callable[[str], bool]] = None
    error_msg: str = "Invalid input."
    transform: Optional[Callable[[str], Any]] = None
```

**`BoolOptionConfig`**: Extends `OptionConfig` for boolean flags with `condition_is_true` callback.

**`CommaSeparatedOptionConfig`** / **`SpaceSeparatedOptionConfig`**: For multi-value options.

**`prompt_options_interactive(options)`**: Iterates over a list of `OptionConfig`, prompts the user for each value, validates input, and returns aggregated CLI args.

**Option configurations**:
- `RUN_OPTIONS`: Options for the `run` command
- `RERUN_OPTIONS`: Options for the `rerun` command
- `STATUS_OPTIONS`: Options for the `status` command
- `CANCEL_OPTIONS`: Options for the `cancel` command

**Utility functions**:
- `read_user_input()`: Interactive prompt with history file (`~/.minerva-history`)
- `is_valid_date(value, fmt)`: Validates date format (`DD-MM-YYYY`)
- `str2date2str(value, fmt)`: Normalizes date strings

## `cli.py` — Click CLI

### CLI Group

```python
@click.group()
def cli():
    pass
```

### Entry Point

```python
def cli_entry():
    if len(sys.argv) == 1:
        interactive_loop()          # No args → interactive mode
    elif len(sys.argv) == 2:
        if sys.argv[1] in ["help", "--help", "-h"]:
            cli()                   # Help → show CLI help
        interactive_loop(sys.argv[1])  # Single command → interactive for that command
    else:
        cli()                       # Multiple args → direct Click invocation
```

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

---

## Job Lifecycle

```
1. User runs: bash minerva-cli.sh run
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
5. bash minerva-cli.sh status --run-date DD-MM-YYYY --run-id N
        ↓
6. bash minerva-cli.sh rerun --run-date DD-MM-YYYY --run-id N --only-failed
        ↓
7. bash minerva-cli.sh cancel --run-date DD-MM-YYYY --run-id N
```

## Directory Structure Created

```
benchmark-runs/
└── {machine}/                    # e.g., bsc-mn5-acc
    └── {date}/                   # e.g., 29-06-2026
        └── slurm-monitor/
            └── {date}/
                └── run_id-{N}/
                    ├── jobs_submitted.jsonl
                    └── jobs_resubmitted-rerun_id-{M}.jsonl
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
