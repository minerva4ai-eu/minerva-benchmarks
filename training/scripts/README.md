# Training Scripts

This directory contains all training scripts, launchers, and utilities for LLM training and fine-tuning benchmarks. Scripts are organized by framework (Accelerate, TorchRun, DeepSpeed, NeMo/Megatron) with shared code that is common across all frameworks.

## Table of Contents

- [Overview](#overview)
  - [How Launchers Work](#how-launchers-work)
- [Directory Structure](#directory-structure)
- [Shared Code (`shared/`)](#shared-code-shared)
  - [`custom_train.py` — CustomTrainer](#custom_trainpy--customtrainer)
  - [`data.py` — Data Loading](#datapy--data-loading)
  - [`utils.py` — Shared Utilities](#utilspy--shared-utilities)
  - [`datasets/` — Dataset Handlers](#datasets--dataset-handlers)
- [Framework-Specific Scripts](#framework-specific-scripts)
  - [Accelerate (`accelerate_common/`)](#accelerate-accelerate_common)
  - [TorchRun (`torchrun_common/`)](#torchrun-torchrun_common)
  - [DeepSpeed (`deepspeed_common/`)](#deepspeed-deepspeed_common)
  - [NeMo / Megatron (`nemo_megatron_common/`)](#nemo--megatron-nemo_megatron_common)
- [Cross-Cutting Utilities](#cross-cutting-utilities)
  - [`runtime_environment.sh`](#runtime_environmentsh)
  - [`gpu_monitor.py` / `gpu_plots.py` — GPU Monitoring](#gpu_monitorpy--gpu_plotspy--gpu-monitoring)
- [How Launchers Work](#how-launchers-work-1)
  - [Launcher Execution Flow (singularity_prefix pattern)](#launcher-execution-flow-singularity_prefix-pattern)
  - [Training Entry Point Command Signatures](#training-entry-point-command-signatures)
- [Adding a New Framework](#adding-a-new-framework)
- [Using `minerva-cli.sh`](#using-minerva-clish)
- [See Also](#see-also)

## Overview

The scripts directory provides:

- **Framework-specific launchers** — SLURM batch scripts that launch training jobs for each framework and parallelism strategy.
- **Training entry points** — Python scripts that define the training loop, model loading, and data pipeline.
- **Shared code** — Framework-agnostic utilities for data loading, logging, TFLOP computation, and dataset handling.
- **GPU monitoring** — Scripts for collecting and visualizing GPU utilization metrics.

### How Launchers Work

All launcher scripts follow the same execution pattern:

```
1. SLURM directives (#SBATCH)
2. Activate environment (source shared/runtime_environment.sh)
3. Setup output directory and environment variables
4. Start GPU monitoring in background (python -m gpu_plots &)
5. Launch training (accelerate/torchrun/deepspeed/python)
6. Kill GPU monitoring and wait for cleanup
7. Print completion status
```

The environment variables used in step 2 (MODEL, DATASET, BATCH_SIZE, etc.) are injected by the SLURM submitter at job submission time (see [scripts/slurm/README.md](slurm/README.md) for the full list).

## Directory Structure

```
scripts/
├── shared/                          # Framework-agnostic code
│   ├── custom_train.py              # CustomTrainer with MegatronFlopsCallback
│   ├── data.py                      # Dataset loading, parsing, collate functions
│   ├── args.py                      # Shared CLI argument parsing
│   ├── comm_metrics.py              # Communication metrics helpers
│   ├── flops.py                     # FLOP counting utilities
│   ├── logger.py                    # Logging setup
│   ├── prepare.py                   # Pre-run preparation helpers
│   ├── utils.py                     # print_rank, count_parameters, timed decorators
│   ├── gpu_monitor.py               # GPU metrics collection
│   ├── gpu_plots.py                 # GPU utilization plotting
│   ├── runtime_environment.sh       # Machine-specific NCCL/CUDA env vars
│   └── datasets/                    # Dataset handler registry + pretokenize.py
│
├── accelerate_common/               # HuggingFace Accelerate framework
│   ├── run-ddp.sh                   # DDP launcher (multi-GPU)
│   ├── run-fsdp.sh                  # FSDP launcher (multi-GPU)
│   ├── finetune-ddp.py              # DDP training entry point
│   ├── finetune-fsdp.py             # FSDP training entry point
│   └── finetune-fsdp-SFTTrainer.py  # FSDP with TRL SFTTrainer
│
├── torchrun_common/                 # PyTorch native distributed
│   ├── run-none.sh                  # Single-GPU launcher
│   ├── run-ddp.sh                   # DDP launcher
│   ├── run-fsdp.sh                  # FSDP launcher
│   ├── finetune-none.py             # Single-GPU training entry point
│   ├── finetune-ddp.py              # DDP training entry point
│   ├── finetune-fsdp.py             # FSDP training entry point
│   └── finetune-fsdp-SFTTrainer.py  # FSDP with TRL SFTTrainer
│
├── deepspeed_common/                # Microsoft DeepSpeed
│   ├── run-deepspeed.sh             # DeepSpeed launcher (ZeRO stages)
│   ├── run-deepspeed-accelerate.sh  # DeepSpeed via Accelerate launcher
│   ├── finetune-deepspeed-pure.py   # DeepSpeed training entry point
│   ├── finetune-deepspeed-accelerate.py  # DeepSpeed+Accelerate entry point
│   └── configs/                     # DeepSpeed JSON configuration files
│
├── nemo_megatron_common/            # NVIDIA NeMo / Megatron
│   ├── run-nemo_megatron.slurm      # NeMo / Megatron launcher
│   ├── nemo_megatron.py             # NeMo / Megatron training entry point
│   └── utils/                       # NeMo-specific helpers (profiling, comms, dataset)
│
└── slurm/                           # SLURM job submission CLI
    ├── README.md                    # See scripts/slurm/README.md
    ├── cli.py
    ├── submitter.py
    └── utils.py
```

---

## Shared Code (`shared/`)

Framework-agnostic code used by all training frameworks.

### `custom_train.py` — CustomTrainer

Extends HuggingFace `Trainer` and `SFTTrainer` with benchmarking-specific functionality:

**Key features:**

- **MegatronFlopsCallback** — Computes TFLOPs per step using Megatron-style analytical FLOP counting:
  ```python
  def compute_tflops_per_step(
      batch_size, seq_len, num_layers, hidden_size,
      intermediate_size, vocab_size, elapsed_seconds, num_gpus
  ) -> float:
      # Forward: QKV projections + attention + output projection + SwiGLU MLP
      # Total FLOPs = 3 × (forward + lm_head)  (forward + 2× backward)
      # TFLOPs = total_flops / elapsed / num_gpus / 1e12
  ```
- **Rank-aware printing** — `print_rank(rank_or_msg, msg)` prints to all ranks or rank 0 only.
- **Parameter counting** — `count_parameters(model)` returns `(trainable, total, trainable_pct)`.
- **Timed decorators** — `@timed("attr")` and `@perf_timed("attr")` append execution times to `self.<attr>` lists.
- **TFLOP local caching** — `inductor_config.autotune_local_cache = True` for faster subsequent runs.
- **TF32 enabled** — `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`.

### `data.py` — Data Loading

**Key functions:**

- **`parse_dataset_paths(data_arg)`** — Parses dataset path argument:
  - Single string path → train/val split
  - JSON string `'{"train": "...", "validation": "..."}'` → explicit splits
  - Python dict string `"{'train': '...', 'validation': '...'}"` → explicit splits
  - Returns: `(train_path, val_path, is_split)`

- **`load_dataset(dataset_name, dataset_path, tokenizer, max_length)`** — Loads and preprocesses datasets:
  - Dispatches to the appropriate `DatasetHandler` via `DATASET_HANDLER_MAP`.
  - Handles train/val splitting via `random_split` or `train_test_split`.
  - Returns `(train_dataset, val_dataset, train_collate_fn, val_collate_fn)`.

### `utils.py` — Shared Utilities

| Function | Description |
|----------|-------------|
| `print_rank(rank_or_msg, msg)` | Rank-aware printing (all ranks or rank 0 only) |
| `count_parameters(model)` | Returns `(trainable_params, total_params, trainable_pct)` |
| `save_summary_stats_json(summary, output_file)` | Saves training summary as JSON |
| `@timed(attr)` | Decorator: appends `time.time()` delta to `self.<attr>` |
| `@perf_timed(attr)` | Decorator: appends `time.perf_counter()` delta to `self.<attr>` |

### `datasets/` — Dataset Handlers

Registry of dataset handlers that load, preprocess, and format data for training. Handlers are mapped via `DATASET_HANDLER_MAP` and `DATASET_MAP`.

**Available handlers:**

| Handler | Dataset | Task | Format |
|---------|---------|------|--------|
| `AlpacaHandler` | Alpaca | Instruction-Finetuning | `prompt`/`completion` |
| `ShareGPTHandler` | ShareGPT | Instruction-Finetuning | `conversations` list |
| `SonnetHandler` | Sonnet (raw text) | Pretraining | Raw text, line-by-line |
| `SquadV2Handler` | SQuAD v2 | Question-Answering | `question`/`answers` |

**Adding a new dataset handler:**

1. Create a new class inheriting from `DatasetHandler` (or `RawTextDataset` for pretraining):
   ```python
   class MyHandler(DatasetHandler):
       name = "my_dataset"
       task = "Instruction-Finetuning"
       
       def preprocess(self, data, tokenizer):
           # Return list of {"prompt": ..., "completion": ...} dicts
           ...
   ```

2. Register the handler in `DATASET_HANDLER_MAP`:
   ```python
   DATASET_HANDLER_MAP = {
       "alpaca": AlpacaHandler,
       "sharegpt": ShareGPTHandler,
       "sonnet": SonnetHandler,
       "squadv2": SquadV2Handler,
       "my_dataset": MyHandler,
   }
   ```

3. Add the dataset name to `DATASET_MAP` in `config_datasets_handlers_map.py`.

---

## Framework-Specific Scripts

### Accelerate (`accelerate_common/`)

[HuggingFace Accelerate](https://huggingface.co/docs/accelerate) provides a high-level API for multi-GPU training.

**Launchers:**

| Script | Parallelism | GPUs | Command |
|--------|-------------|------|---------|
| `run-ddp.sh` | DDP | ≥2 | `accelerate launch --num_processes <N>` |
| `run-fsdp.sh` | FSDP | ≥2 | `accelerate launch --num_processes <N>` |

**Training entry points:**

| Script | Description |
|--------|-------------|
| `finetune-ddp.py` | DDP fine-tuning |
| `finetune-fsdp.py` | FSDP fine-tuning |
| `finetune-fsdp-SFTTrainer.py` | FSDP with HuggingFace TRL SFTTrainer |

**Key launcher flow (run-ddp.sh example):**
```bash
#!/bin/bash
#SBATCH --job-name=ACCELERATE_DYNAMIC
#SBATCH --time=24:00:00

# 1. Activate environment
source shared/runtime_environment.sh

# 2. Setup
OUTPUT_DIR="${LAUNCH_FOLDER}/output"
mkdir -p $OUTPUT_DIR
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True

# 3. Start GPU monitoring
python -m gpu_plots &
monitor_pid=$!
sleep 5

# 4. Launch training
accelerate launch --num_processes "$SLURM_GPUS_ON_NODE" --mixed_precision "$PRECISION" \
    "$TRAIN_SCRIPT" \
    --model $MODEL_PATH --data $DATASET_PATH --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE --max_length $MAX_MODEL_LENGTH \
    --precision $PRECISION --lr $LR \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --dataloader_num_workers 4 --dataset $DATASET

# 5. Cleanup
kill -SIGTERM "$monitor_pid"
wait "$monitor_pid"
```

### TorchRun (`torchrun_common/`)

PyTorch's native distributed training package provides lower-level control over DDP and FSDP.

**Launchers:**

| Script | Parallelism | GPUs | Command |
|--------|-------------|------|---------|
| `run-none.sh` | Single-GPU | 1 | `python "$TRAIN_SCRIPT"` (no distributed launch) |
| `run-ddp.sh` | DDP | ≥2 | `torchrun --nproc_per_node <N>` |
| `run-fsdp.sh` | FSDP | ≥2 | `torchrun --nproc_per_node <N>` |

**Training entry points:**

| Script | Description |
|--------|-------------|
| `finetune-none.py` | Single-GPU training |
| `finetune-ddp.py` | DDP training |
| `finetune-fsdp.py` | FSDP training |
| `finetune-fsdp-SFTTrainer.py` | FSDP with HuggingFace TRL SFTTrainer |

### DeepSpeed (`deepspeed_common/`)

[Microsoft DeepSpeed](https://www.deepspeed.ai/) provides ZeRO optimization for training very large models across multiple GPUs/nodes.

**Launcher:**

| Script | Parallelism | GPUs | Command |
|--------|-------------|------|---------|
| `run-deepspeed.sh` | ZeRO-1/2/3/3-Offload | ≥2 | `deepspeed --master_port <PORT> "$TRAIN_SCRIPT"` |
| `run-deepspeed-accelerate.sh` | ZeRO via Accelerate | ≥2 | `accelerate launch "$TRAIN_SCRIPT"` |

**Training entry points:**

| Script | Description |
|--------|-------------|
| `finetune-deepspeed-pure.py` | Pure DeepSpeed training (programmatic config) |
| `finetune-deepspeed-accelerate.py` | DeepSpeed training via Accelerate |

**DeepSpeed-specific features:**
- **ZeRO stage configuration** — ZeRO-1 (optimizer sharding), ZeRO-2 (optimizer + gradient sharding), ZeRO-3 (full parameter sharding), ZeRO-3-Offload (CPU offload).
- **Model pre-staging** — Optional pre-staging of model weights to node-local `$TMPDIR` to avoid GPFS I/O contention (commented out by default).
- **DeepSpeed JSON configs** — Located in `configs/`, define ZeRO stage, micro batch size, gradient accumulation, etc.

**DeepSpeed JSON Configs (`deepspeed_common/configs/`):**

| File | Purpose |
|------|---------|
| `zero1.json` | ZeRO-1 config (optimizer state sharding across GPUs) |
| `zero2.json` | ZeRO-2 config (optimizer + gradient sharding) |
| `zero3.json` | ZeRO-3 config (full parameter sharding) |

Each JSON file specifies:
```json
{
  "zero_optimization": {
    "stage": 1,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": 0.0001, "betas": [0.9, 0.999], "eps": 1e-8}
  },
  "scheduler": {...},
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "bf16": {"enabled": true}
}
```

**FSDP Layer Wrapping:**

For FSDP parallelism, the system automatically determines which layers to wrap based on the model type. The `get_fsdp_layer_to_wrap()` function in `shared/utils.py` maps model types to their layer classes:

```python
def get_fsdp_layer_to_wrap(model_type: str) -> List[Type]:
    """Returns the list of layer classes to wrap with FSDP."""
    WRAP_MAPPING = {
        "llama": [LlamaDecoderLayer],
        "mistral": [MistralDecoderLayer],
        "gemma": [GemmaDecoderLayer],
        "phi": [PhiDecoderLayer],
        # ... more mappings
    }
    return WRAP_MAPPING.get(model_type, [PreTrainedModel])
```

This ensures that FSDP shards at the appropriate granularity for each model architecture, balancing memory savings with communication overhead.

### NeMo / Megatron (`nemo_megatron_common/`)

[NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/) with Megatron-LM provides tensor/pipeline/context/data/expert parallelism for very large models.

**Launcher:**

| Script | Parallelism | Command |
|--------|-------------|---------|
| `run-nemo_megatron.slurm` | Megatron (`tp/pp/cp/dp/ep/sp`) | `python "$TRAIN_SCRIPT"` inside the NeMo container |

**Training entry point:**

| Script | Description |
|--------|-------------|
| `nemo_megatron.py` | NeMo / Megatron training (uses the `tulu-3-sft-mixture` dataset) |

**`utils/` helpers:** profiling analysis (`analyze_async_pytorch_profiler.py`, `analyze_sync_pytorch_profiler.py`), communication measurements (`comm_measurements.py`, `nccl_tagger.py`), benchmark callbacks (`my_benchmark_callback.py`, `my_megatron_benchmark_callback.py`), dataset utilities (`dataset.py`), and host/CPU helpers (`hostlist.py`, `cpu_mem_usage.py`, `chrono.py`, `ac_handler.py`).

---

## Cross-Cutting Utilities

### `runtime_environment.sh`

Located in `shared/`. Sets machine-specific environment variables for NCCL, CUDA, and PyTorch. Sourced by every launcher before training:

```bash
source shared/runtime_environment.sh
```


### `gpu_monitor.py` / `gpu_plots.py` — GPU Monitoring

Both live in `shared/`. `gpu_monitor.py` collects GPU metrics during the run; `gpu_plots.py` generates matplotlib/seaborn visualizations from them.

**Metrics collected:**
- Timestamp
- GPU index
- GPU name
- Power draw (watts)
- Memory used (MiB)
- Utilization (%)

**Output:**
- Per-GPU line charts for power, memory, and utilization
- Saved as PNG files in `profiler/{SLURM_JOB_ID}/{nodeid}-{nodename}/`
- Log file: `gpus_monitor.logs`

**Usage:**
```python
# Run as module (launched from training scripts)
python -m gpu_plots
```

---

## How Launchers Work

All launcher scripts follow the same pattern:

```
1. SLURM directives (#SBATCH)
2. Activate environment (source shared/runtime_environment.sh)
3. Setup output directory and environment variables
4. Start GPU monitoring in background
5. Launch training (accelerate/torchrun/deepspeed/python)
6. Kill GPU monitoring and wait for cleanup
7. Print completion status
```

### Launcher Execution Flow (singularity_prefix pattern)

Each launcher script is executed inside a Singularity container. The container path and bind mounts are configured by the SLURM submitter via environment variables:

```bash
# The submitter sets these before launching the job:
SINGULARITY_CONTAINER=/path/to/singularity_uv-runtime.sif
SINGULARITY_BINDS="--bind /gpfs/scratch:/gpfs/scratch --bind $HOME:$HOME"
SINGULARITY_ARGS="--nv"  # or "--rocm" for AMD GPUs

# The launcher script uses them:
singularity exec $SINGULARITY_ARGS $SINGULARITY_BINDS \
    $SINGULARITY_CONTAINER \
    bash -c "cd $LAUNCH_FOLDER && source shared/runtime_environment.sh && python $TRAIN_SCRIPT ..."
```

The `singularity_prefix` pattern ensures that all training runs inside the same container environment, regardless of the framework or parallelism strategy.

### Training Entry Point Command Signatures

Each `finetune-*.py` script accepts the same set of command-line arguments, injected by the launcher script:

```bash
python finetune-*.py \
    --model $MODEL_PATH \
    --data $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_MODEL_LENGTH \
    --precision $PRECISION \
    --lr $LR \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --dataloader_num_workers 4 \
    --dataset $DATASET \
    --steps $STEPS \
    --epochs $EPOCHS \
    --optimizer $OPTIMIZER \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING
```

**Argument mapping from SLURM environment variables:**

| CLI Argument | Environment Variable | Source |
|-------------|---------------------|--------|
| `--model` | `$MODEL_PATH` | `cfg.model.path` |
| `--data` | `$DATASET_PATH` | `cfg.dataset.path` |
| `--output_dir` | `$LAUNCH_FOLDER/output` | Computed by submitter |
| `--batch_size` | `$BATCH_SIZE` | `cfg.trainings.batch_sizes` |
| `--max_length` | `$MAX_MODEL_LENGTH` | `cfg.dataset.max_seq_len` |
| `--precision` | `$PRECISION` | `cfg.trainings.precisions` |
| `--lr` | `$LR` | `cfg.trainings.lr` |
| `--gradient_accumulation_steps` | `$GRAD_ACCUM` | `cfg.trainings.grad_accums` |
| `--dataset` | `$DATASET` | `cfg.dataset.name` |
| `--steps` | `$STEPS` | `cfg.trainings.steps` |
| `--epochs` | `$EPOCHS` | `cfg.trainings.epochs` |
| `--optimizer` | `$OPTIMIZER` | `cfg.trainings.optimizer` |
| `--gradient_checkpointing` | `$GRADIENT_CHECKPOINTING` | `cfg.trainings.gradient_checkpointing` |

---

## Adding a New Framework

To add support for a new training framework:

1. **Create framework directory:** `scripts/<framework-name>_common/`

2. **Create launcher scripts:** `run-none.sh`, `run-ddp.sh`, `run-fsdp.sh` (or equivalent parallelism strategies)

3. **Create training entry points:** `finetune-none.py`, `finetune-ddp.py`, `finetune-fsdp.py`

4. **Register in config:** Add `configs/framework/<framework-name>.yaml` with:
   ```yaml
   name: <framework-name>
   parallelism:
     none: { min_gpus: 1, max_gpus: 1 }
     ddp: { min_gpus: 2 }
     fsdp: { min_gpus: 2 }
   scripts:
     run: scripts/<framework-name>_common/run-${parallelism}.sh
     finetune: scripts/<framework-name>_common/finetune-${parallelism}.py
     shared: scripts/shared
     copy_files:
       - ${scripts.run}
       - ${scripts.finetune}
   ```

5. **Register in Hydra:** Add to `dataclasses_hydra/__init__.py`:
   ```python
   cs.store(group="framework", name="<framework-name>", node=FrameworkConfig)
   ```

6. **Add to generator:** Add `<framework-name>` to `FRAMEWORKS` list in `hydra_app.py`.

---

## Using `minerva-cli.sh`

Scripts in this directory are never invoked directly. For each valid model × framework × dataset × parallelism combination, the CLI wrapper at `training/minerva-cli.sh` copies the matching launcher (`run-*.sh`), training entry point (`finetune-*.py`), and `shared/` code into a per-run launch folder, then submits it to SLURM:

```bash
cd training
bash minerva-cli.sh run --config-name MN5-singularity
```

`minerva-cli.sh` activates `envs/cli/.venv` and runs `python -m scripts.slurm.cli run ...`, which:

1. Composes and validates one `BenchmarkConfig` YAML per combination (Hydra, see [configs_hydra/README.md](../configs_hydra/README.md)).
2. Copies the launcher, training entry point, and shared code into `benchmark-runs-{config-name}/{machine}/{date}/{job_id}/...`.
3. Submits `MINERVA.job` to SLURM, which `srun`s each launcher with the environment variables described in [Training Entry Point Command Signatures](#training-entry-point-command-signatures).

See [training/README.md](../README.md#using-minerva-clish) for the full CLI reference (`run` options, typical workflows) and [scripts/slurm/README.md](slurm/README.md) for job lifecycle details.

---

## See Also

- [configs_hydra/README.md](../configs_hydra/README.md) — Configuration system
- [scripts/slurm/README.md](slurm/README.md) — SLURM job submission CLI
- [envs/README.md](../envs/README.md) — Environment management
- [training/README.md](../README.md) — Root project overview
