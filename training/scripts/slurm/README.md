# SLURM Job Submission CLI

Click-based CLI for generating, validating, and submitting LLM training benchmark jobs to SLURM clusters.

## Overview

```
Config Generation → Combo Validation → YAML Config Write → sbatch Submission → Job Monitoring
```

## CLI Entry Point

Invoked via the shell wrapper `minerva-cli.sh`:

```bash
bash minerva-cli.sh <command> [options]
```

The wrapper activates the virtualenv at `envs/cli/.venv/bin/activate`, sets up logging to `cli-logs/`, and runs `python -m scripts.slurm.cli`.

## Commands

### `run` — Generate and Submit Jobs

```bash
bash minerva-cli.sh run [OPTIONS]
```

Generates valid benchmark configurations and submits them as SLURM jobs.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | `False` | Generate configs without submitting jobs |
| `--configs-path` | `./configs_hydra/configs` | Path to Hydra config directory |
| `--config-name` | `base` | Base config name to compose (required, e.g., `base-MN5`) |
| `--runs-dir` | `benchmark-runs/` | Output directory for generated configs and results |
| `--per-model-jobs` | `False` | Group experiments per model into separate jobs |
| `--per-nodes-jobs` | `False` | Group experiments per node count into separate jobs |
| `--models` | `""` | Comma-separated model names to filter (e.g., `gemma3-1b,mistral_7b`) |
| `--frameworks` | `""` | Comma-separated framework names to filter |
| `--datasets` | `""` | Comma-separated dataset names to filter |
| `--yaml` | `None` | Run specific BenchmarkConfig YAML file(s). Can be repeated |
| `--nnodes` | `None` | Override number of nodes for the job |

**Examples:**

```bash
# Generate and submit all valid jobs
bash minerva-cli.sh run --config-name base-MN5

# Dry run — generate configs without submitting
bash minerva-cli.sh run --config-name base-MN5 --dry-run

# Filter by model and framework
bash minerva-cli.sh run --config-name base-MN5 --models mistral_7b --frameworks deepspeed-cuda130

# Run specific YAML configs
bash minerva-cli.sh run --config-name base-MN5 --yaml config1.yaml --yaml config2.yaml

# Group jobs per model
bash minerva-cli.sh run --config-name base-MN5 --per-model-jobs
```

### Deprecated Commands

The following commands are currently **commented out** in `cli.py` and not available:
- `rerun` — Rerun jobs from a previous run
- `status` — Check job status
- `cancel` — Cancel running/pending jobs

### Interactive Mode

Run without arguments to enter interactive mode:

```bash
bash minerva-cli.sh
```

Available interactive commands: `run`, `help`, `quit`/`exit`.

## Subpackage Structure

```
scripts/slurm/
├── __init__.py
├── cli.py              # Click CLI group with run command + interactive mode
├── cli_utils.py        # Interactive prompt utilities, OptionConfig dataclasses
├── submitter.py        # Job submission: config writing, env building, sbatch
├── monitor.py          # SLURM state dashboard with status icons
└── utils.py            # ANSI colors, Unicode icons, JSONL I/O, YAML loading
```

---

## `cli.py` — Click CLI

### CLI Group

```python
@click.group()
def cli():
    """MINERVA SLURM job submission CLI for LLM training and fine-tuning benchmarks."""
```

### `run` Command

```python
@cli.command()
@click.option("--dry-run", is_flag=True)
@click.option("--configs-path", default="./configs_hydra/configs")
@click.option("--config-name", default="base", required=True)
@click.option("--runs-dir", default="benchmark-runs/")
@click.option("--per-model-jobs", is_flag=True)
@click.option("--per-nodes-jobs", is_flag=True)
@click.option("--models", type=str, default="")
@click.option("--frameworks", type=str, default="")
@click.option("--datasets", type=str, default="")
@click.option("--yaml", "yamls", multiple=True, default=None)
@click.option("--nnodes", default=None, type=int)
def run(dry_run, configs_path, config_name, runs_dir, per_model_jobs,
        per_nodes_jobs, models, frameworks, datasets, yamls, nnodes):
    """Generate benchmark configurations and submit SLURM job with steps."""
```

**Execution flow:**

1. **Validate `--config-name`**: Must not be the default `base`; exits with error if not provided.

2. **Load configurations**: Either loads YAML configs from `--yaml` paths, or calls `generate_valid_combos()` from `configs_hydra.hydra_app` to produce valid configs. Filters by `--models`, `--frameworks`, `--datasets` if provided.

3. **Write configs**: Calls `write_config()` to save each valid config as a YAML file under `{runs_dir}-{config_name}/{machine}/yaml-configs/...`.

4. **Submit job**: Calls `submit_job()` which:
   - Determines node count via `get_job_nodes()` (max nodes across configs, or `--nnodes` override)
   - Builds sbatch command with partition, QoS, account, constraints from config
   - Creates a manifest file listing all YAML config paths
   - Submits via `sbatch` with environment built by `build_sbatch_env()`

5. **Job execution**: The submitted `MINERVA.job` script reads the manifest, iterates over YAML configs, builds srun environment via `build_srun_env()`, and launches training with `srun`.

**Grouping options:**
- `--per-model-jobs`: Groups configs by model name, submits one job per model
- `--per-nodes-jobs`: Groups configs by node count, submits one job per node count

**Dry-run vs actual submission**:
- **Dry-run** (`--dry-run`): Generates and saves YAML configs only
- **Actual submission**: Saves configs + submits via `sbatch`

---

## `submitter.py` — Job Submission Logic

### `write_config(cfg, base_dir, runs_dir, dry)`

Saves a `BenchmarkConfig` as a YAML file under `{runs_dir}/{machine}/yaml-configs/...`. Returns the config file path.

### `_build_launch_folder(cfg, base_dir, runs_dir, dry, repeat_id, run_date)`

Creates the per-experiment directory structure. Returns `Path` to the launch folder (or YAML config path in dry-run mode).

### `copy_scripts(cfg, dest)`

Copies scripts to the launch folder:
1. Framework-specific run script (`cfg.framework.scripts.run`)
2. Framework-specific finetune script (`cfg.framework.scripts.finetune`)
3. All files listed in `cfg.framework.scripts.copy_files`

### `build_srun_env()`

Builds the environment for the `srun` step inside `MINERVA.job`. Reads `YAML_PATH` and `TEMP_ENV_FILE` from environment, loads the config, and writes export statements to the temp file. Sets variables like `LOAD_MODULES`, `EXECUTION_MODE`, `VENV_PATH`, `SINGULARITY_CONTAINER`, `SRUN_SCRIPT`, `TRAIN_SCRIPT`, `FRAMEWORK`, `PARALLELISM`, `NNODES`, `DATASET_PATH`, `ZERO_STAGE`, `PYTHONPATH`.

### `build_sbatch_env(machine, yamls, results_dir)`

Builds the environment for the initial `sbatch` submission. Creates a manifest file listing all YAML config paths (to bypass ARG_MAX limits), then sets `MODULES`, `EXECUTION_MODE`, `SINGULARITY_BINDS`, `SINGULARITY_ARGS`, `MINERVA_WORKDIR`, `MINERVA_MANIFEST_FILE`.

### `get_job_nodes(cfgs, nnodes)`

Returns the number of nodes to request. If `--nnodes` is provided, validates it is >= max nodes across configs. Otherwise returns the max nodes found in configs.

### `submit_job(cfgs, cfgs_paths, runs_dir, run_date, nnodes)`

Submits the benchmark job to SLURM:
1. Determines node count via `get_job_nodes()`
2. Samples slurm/machine config from first config (common across all)
3. Builds sbatch command with `--nodes`, `--gres`, `--cpus-per-task`, `--tasks-per-node`, `--partition`, optional `--account`/`--qos`/`--constraint`
4. Calls `build_sbatch_env()` to create environment with manifest file
5. Submits via `sbatch --parsable MINERVA.job`
6. Returns the job ID

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
    default: str | None = None
    required: bool = False
    validator: Callable[[str], bool] | None = None
    error_msg: str = "Invalid input."
    transform: Callable[[str], Any] | None = None
    exit_after: bool | None = False
```

**`BoolOptionConfig`**: Extends `OptionConfig` for boolean flags with `condition_is_true` callback.

**`CommaSeparatedOptionConfig`** / **`SpaceSeparatedOptionConfig`**: For multi-value options.

**`prompt_options_interactive(options)`**: Iterates over a list of `OptionConfig`, prompts the user for each value, validates input, and returns aggregated CLI args.

**Option configurations**:
- `RUN_OPTIONS`: Options for the `run` command
- `RERUN_OPTIONS`, `STATUS_OPTIONS`, `CANCEL_OPTIONS`: Defined but commands are deprecated

**Utility functions**:
- `read_user_input()`: Interactive prompt with history file (`~/.minerva-history`)
- `is_valid_date(value, fmt)`: Validates date format (`DD-MM-YYYY`)
- `str2date2str(value, fmt)`: Normalizes date strings

**Constants**:
- `RUNS_DIR = Path("benchmark-runs/")`
- `DEFAULT_CONFIGS_PATH = "./configs_hydra/configs"`
- `DEFAULT_CONFIG_NAME = "base"`
- `BASE_DIR = Path(".")`

---

## `monitor.py` — SLURM State Dashboard

Provides a status dashboard for SLURM jobs with emoji icons and color coding.

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

### Functions

- `load_all(run)`: Reads JSONL job tracking file
- `get_job_info(job_id)`: Queries `sacct` for job state, returns `JobInfo` dataclass
- `print_job_status(job, job_info)`: Prints formatted job status line
- `filter_by_status(runs_dir, status)`: Filters configs by status

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
SKIPPED = "↷"
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
def load_config(path: str) -> BenchmarkConfig:
    """Load BenchmarkConfig from YAML via OmegaConf."""

def load_yaml(filepath):
    """Load and parse a YAML file."""
```

### Path Helpers

- `get_cfg_folder(cfg, base_dir, runs_dir)`: Returns config directory path
- `get_cfg_folder_from_launch(launch_path)`: Extracts config dir from launch path
- `copy_launch_folder(source, target)`: Copies launch folder excluding logs/outputs

---

## Job Lifecycle

```
1. User runs: bash minerva-cli.sh run --config-name base-MN5
        ↓
2. generate_valid_combos() → list of valid BenchmarkConfig objects
        ↓
3. write_config() → saves each config as YAML under yaml-configs/
        ↓
4. submit_job() → builds sbatch command + manifest file
        ↓
5. sbatch MINERVA.job → SLURM allocates nodes
        ↓
6. MINERVA.job reads manifest, for each YAML:
   a. build_srun_env() → sets environment variables
   b. srun bash $SRUN_SCRIPT $YAML_PATH → launches training
        ↓
7. Training executes on compute nodes
   - GPU monitoring runs in background
   - Results saved to output directory
```

## Directory Structure Created

```
benchmark-runs-{config-name}/
└── {machine}/                    # e.g., bsc-mn5-acc
    └── yaml-configs/
        └── {model}/{framework}/{parallelism}/{dataset}/nodes-{N}/
            └── {config-name}.yaml
    └── {date}/
        └── {job_id}/
            ├── sbatch-slurm-logs/
            │   ├── MINERVA-JOB-{job_id}.out
            │   └── MINERVA-JOB-{job_id}.err
            ├── steps-slurm-logs/
            │   ├── srun-{job_id}.{step_id}.out
            │   ├── srun-{job_id}.{step_id}.err
            │   └── srun-{job_id}.{step_id}.nccl
            ├── outputs/
            │   ├── gpus-monitor/
            │   └── training-results/
            └── training-logs/
```

## See Also

- [configs_hydra/README.md](../configs_hydra/README.md) — Configuration system
- [scripts/README.md](../README.md) — Training scripts overview
- [training/README.md](../../README.md) — Root project overview
