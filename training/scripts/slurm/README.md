# SLURM Job Submission CLI

Click CLI for preparing, submitting, and collecting results from LLM training benchmark jobs on SLURM clusters.

## Entry Point

Run the wrapper from the `training/` directory:

```bash
bash minerva-cli.sh <command> [options]
```

The wrapper activates `envs/cli/.venv`, writes logs under `cli-logs/<date>/`, and invokes `python -m scripts.slurm.cli`. The CLI does not provide an interactive prompt mode.

## Available Commands

Only `run`, `prepare`, and `results` are currently registered. `rerun`, `status`, and `cancel` are not available as CLI commands.

### `run` — Generate and Submit Benchmarks

```bash
bash minerva-cli.sh run --config-name MN5-singularity [OPTIONS]
```

Generates valid benchmark configurations from a Hydra machine profile, saves the YAML configs, and submits them through `sbatch`.

| Option | Default | Description |
|---|---|---|
| `--dry-run` | Off | Generate and save configs without submitting. |
| `--configs-path PATH` | `./configs_hydra/configs` | Hydra config directory. |
| `--config-name NAME` | `base` | Profile to compose. Generated combinations require a non-`base` profile. |
| `--profile NAME` | None | Benchmark profile from `configs_hydra/configs/profile/` (file name without `.yaml`), merged on top of `--config-name`. See [Benchmark Profiles](#benchmark-profiles). |
| `--runs-dir PATH` | `benchmark-runs/` | Base directory for generated configs and results; `-<config-name>` and, if set, `-<profile>` are appended. |
| `--per-model-jobs` | Off | Submit one job per model. |
| `--per-nodes-jobs` | Off | Submit one job per node count. |
| `--models NAMES` | All | Comma-separated model-name filter. |
| `--frameworks NAMES` | All | Comma-separated framework-name filter. |
| `--datasets NAMES` | All | Comma-separated dataset-name filter. |
| `--yaml PATH` | None | Run an existing `BenchmarkConfig` YAML; may be repeated. |
| `--nnodes COUNT` | Config maximum | Override node count, if it is not less than the maximum required by the selected configs. |

Examples:

```bash
# Generate and submit the selected profile's valid benchmark combinations
bash minerva-cli.sh run --config-name MN5-singularity

# Generate configs without submitting
bash minerva-cli.sh run --config-name MN5-singularity --dry-run

# Restrict the run to a benchmark profile (models, frameworks, batch sizes, precisions)
bash minerva-cli.sh run --config-name MN5-singularity --profile llama3-megatron-bf16-bf16fp8

# Filter generated combinations
bash minerva-cli.sh run --config-name MN5-singularity --models mistral_7b --frameworks deepspeed

# Submit existing generated YAML configs
bash minerva-cli.sh run --yaml benchmark-runs-MN5-singularity/path/to/config1.yaml --yaml benchmark-runs-MN5-singularity/path/to/config2.yaml
```

`--yaml` cannot be combined with `--dry-run`. The machine config (`--config-name`) is inferred from the YAML path in this mode.

### Benchmark Profiles

Two different things are called "profile" here:

- `--config-name` selects a **machine config** (e.g. `MN5-singularity`), which describes the cluster and environment.
- `--profile` selects an optional machine-independent **benchmark profile**: a YAML file in `configs_hydra/configs/profile/`, e.g. `llama3-megatron-bf16-bf16fp8.yaml`.

How `--profile` is applied:

1. The machine config is composed through Hydra as usual.
2. `profile/<NAME>.yaml` is merged on top (`compose_with_profile` in `configs_hydra/hydra_app.py`), so profile values override machine and model defaults.
3. The profile's `selection.models`, `selection.frameworks` and `selection.datasets` restrict those axes, and `model.combinations` (e.g. `global_batch_sizes`, `batch_sizes`, `precisions`) overrides the combination grid. An axis the profile does not mention is left unrestricted.
4. CLI filters (`--models`, `--frameworks`, `--datasets`) are applied first. A profile `selection` entry that is not among the remaining choices raises an error.
5. If the profile file does not exist, the CLI fails and lists the available profiles.

Example profile:

```yaml
# configs_hydra/configs/profile/llama3-megatron-bf16-bf16fp8.yaml
selection:
  models: [llama3_70b]
  frameworks: [megatron-nemo-2509]

model:
  combinations:
    global_batch_sizes: [512]
    batch_sizes: [2]
    precisions: [bf16, bf16_fp8]
```

Output location: the profile name is appended to the results directory, giving `<runs-dir>-<config-name>-<profile>` (for example `benchmark-runs-MN5-singularity-llama3-megatron-bf16-bf16fp8`). Pass that same directory as `--root` to `results`.

### `prepare` — Megatron-NEMO Preparation

```bash
bash minerva-cli.sh prepare --config-name MN5-singularity [OPTIONS]
```

Composes preparation-only configs for the `megatron-nemo-2509` framework and submits one SLURM job unless `--dry-run` is specified. If `--models` is omitted, supported models are considered.

Options: `--configs-path PATH`, required `--config-name NAME`, `--profile NAME` (see [Benchmark Profiles](#benchmark-profiles)), `--runs-dir PATH`, comma-separated `--models NAMES`, and `--dry-run`. Dry-run still writes the generated configs; it skips submission.

### `results` — Collect Training Results

```bash
bash minerva-cli.sh results --jobids 45917001,45917002 [OPTIONS]
```

Collects summaries for the specified Slurm job IDs and writes a CSV. Additional job IDs can also be passed as positional arguments.

| Option | Default | Description |
|---|---|---|
| `--jobids IDS` | Required | Comma-separated job IDs. |
| `--root PATH` | `benchmark-runs-MN5-singularity` | Results root directory. |
| `--output PATH` | Generated path | CSV output path under `analytics/results/`, named from the job IDs. |

Example:

```bash
bash minerva-cli.sh results --jobids 45917001,45917002 --root benchmark-runs-MN5-singularity
```

## Submission Flow

1. `run` composes valid `BenchmarkConfig` combinations, or loads supplied YAML files.
2. Configs are saved under the profile-specific results directory.
3. `submit_job()` builds the `sbatch` command and writes a manifest of YAML paths.
4. `MINERVA.job` reads the manifest and launches each config as an `srun` step.
5. After jobs finish, `results` collects the training summaries into a CSV.

`prepare` is a separate workflow that composes and submits Megatron-NEMO preparation configs.

## Package Modules

```text
scripts/slurm/
├── cli.py       # Click commands: run, prepare, results
├── submitter.py # Config writing, environment setup, and sbatch submission
├── monitor.py   # Monitoring helpers; no status CLI command is registered
└── utils.py     # Shared YAML, JSONL, path, and terminal-formatting helpers
```

## See Also

- [Hydra configuration guide](../configs_hydra/README.md)
- [Training scripts overview](../README.md)
- [Training project README](../../README.md)
