# Training Scripts

This directory contains all training scripts, launchers, and utilities for LLM training and fine-tuning benchmarks. Scripts are organized by framework (Accelerate, TorchRun, DeepSpeed) with shared code that is common across all frameworks.

## Overview

The scripts directory provides:

- **Framework-specific launchers** — SLURM batch scripts that launch training jobs for each framework and parallelism strategy.
- **Training entry points** — Python scripts that define the training loop, model loading, and data pipeline.
- **Shared code** — Framework-agnostic utilities for data loading, logging, TFLOP computation, and dataset handling.
- **GPU monitoring** — Scripts for collecting and visualizing GPU utilization metrics.
- **Shell utilities** — Helper functions for environment activation, config parsing, and path resolution.

### How Launchers Work

All launcher scripts follow the same execution pattern:

```
1. SLURM directives (#SBATCH)
2. Activate environment (source activate-env-variables-per-supercomputer.sh)
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
│   ├── utils.py                     # print_rank, count_parameters, timed decorators
│   └── datasets/                    # Dataset handler registry
│
├── accelerate-common/               # HuggingFace Accelerate framework
│   ├── run-none.sh                  # Single-GPU launcher
│   ├── run-ddp.sh                   # DDP launcher (multi-GPU)
│   ├── run-fsdp.sh                  # FSDP launcher (multi-GPU)
│   ├── finetune-none.py             # Single-GPU training entry point
│   ├── finetune-ddp.py              # DDP training entry point
│   ├── finetune-fsdp.py             # FSDP training entry point
│   ├── gpu_monitor.py               # GPU metrics collection
│   └── utils.py                     # Accelerate-specific utilities
│
├── torchrun-common/                 # PyTorch native distributed
│   ├── run-none.sh                  # Single-GPU launcher
│   ├── run-ddp.sh                   # DDP launcher
│   ├── run-fsdp.sh                  # FSDP launcher
│   ├── finetune-none.py             # Single-GPU training entry point
│   ├── finetune-ddp.py              # DDP training entry point
│   ├── finetune-fsdp.py             # FSDP training entry point
│   ├── gpu_monitor.py               # GPU metrics collection
│   └── utils.py                     # TorchRun-specific utilities
│
├── deepspeed-common/                # Microsoft DeepSpeed
│   ├── run-deepspeed.sh             # DeepSpeed launcher (ZeRO stages)
│   ├── finetune-deepspeed.py        # DeepSpeed training entry point
│   ├── configs/                     # DeepSpeed JSON configuration files
│   ├── gpu_monitor.py               # GPU metrics collection
│   ├── metrics.py                   # DeepSpeed-specific metrics
│   └── utils.py                     # DeepSpeed-specific utilities
│
├── slurm/                           # SLURM job submission CLI
│   ├── README.md                    # See scripts/slurm/README.md
│   ├── cli.py
│   ├── submitter.py
│   ├── monitor.py
│   └── utils.py
│
├── activate-env-variables-per-supercomputer.sh  # Machine-specific NCCL/CUDA env vars
├── utils.sh                         # Shell utilities (JSON/YAML parsing)
└── gpu_plots.py                     # GPU utilization plotting
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

### Accelerate (`accelerate-common/`)

[HuggingFace Accelerate](https://huggingface.co/docs/accelerate) provides a high-level API for multi-GPU training.

**Launchers:**

| Script | Parallelism | GPUs | Command |
|--------|-------------|------|---------|
| `run-none.sh` | Single-GPU | 1 | `accelerate launch --num_processes 1` |
| `run-ddp.sh` | DDP | ≥2 | `accelerate launch --num_processes <N>` |
| `run-fsdp.sh` | FSDP | ≥2 | `accelerate launch --num_processes <N>` |

**Training entry points:**

| Script | Description |
|--------|-------------|
| `finetune-none.py` | Single-GPU fine-tuning |
| `finetune-ddp.py` | DDP fine-tuning |
| `finetune-fsdp.py` | FSDP fine-tuning |
| `finetune-fsdp-SFTTrainer.py` | FSDP with HuggingFace TRL SFTTrainer |

**Key launcher flow (run-none.sh example):**
```bash
#!/bin/bash
#SBATCH --job-name=ACCELERATE_DYNAMIC
#SBATCH --time=24:00:00

# 1. Activate environment
source activate-env-per-supercomputer.sh $ENVIRONMENT_FINETUNING

# 2. Setup
OUTPUT_DIR="${LAUNCH_FOLDER}/output"
mkdir -p $OUTPUT_DIR
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True

# 3. Start GPU monitoring
python -m gpu_plots &
monitor_pid=$!
sleep 5

# 4. Launch training
accelerate launch --num_processes 1 --mixed_precision "$PRECISION" \
    "$TRAIN_SCRIPT" \
    --model $MODEL_PATH --data $DATASET_PATH --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE --max_length $MAX_MODEL_LENGTH \
    --precision $PRECISION --lr $LR \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --dataloader_num_workers 32 --dataset $DATASET

# 5. Cleanup
kill -SIGTERM "$monitor_pid"
wait "$monitor_pid"
```

### TorchRun (`torchrun-common/`)

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

### DeepSpeed (`deepspeed-common/`)

[Microsoft DeepSpeed](https://www.deepspeed.ai/) provides ZeRO optimization for training very large models across multiple GPUs/nodes.

**Launcher:**

| Script | Parallelism | GPUs | Command |
|--------|-------------|------|---------|
| `run-deepspeed.sh` | ZeRO-1/2/3/3-Offload | ≥2 | `deepspeed --master_port <PORT> "$TRAIN_SCRIPT"` |

**Training entry points:**

| Script | Description |
|--------|-------------|
| `finetune-deepspeed.py` | DeepSpeed training with config |
| `finetune-deepspeed-pure.py` | Pure DeepSpeed training (programmatic config) |

**DeepSpeed-specific features:**
- **ZeRO stage configuration** — ZeRO-1 (optimizer sharding), ZeRO-2 (optimizer + gradient sharding), ZeRO-3 (full parameter sharding), ZeRO-3-Offload (CPU offload).
- **Model pre-staging** — Optional pre-staging of model weights to node-local `$TMPDIR` to avoid GPFS I/O contention (commented out by default).
- **DeepSpeed JSON configs** — Located in `configs/`, define ZeRO stage, micro batch size, gradient accumulation, etc.
- **Metrics collection** — `metrics.py` collects DeepSpeed-specific metrics (throughput, memory, ZeRO stats).

**DeepSpeed JSON Configs (`deepspeed-common/configs/`):**

| File | Purpose |
|------|---------|
| `zero1.json` | ZeRO-1 config (optimizer state sharding across GPUs) |
| `zero2.json` | ZeRO-2 config (optimizer + gradient sharding) |
| `zero3.json` | ZeRO-3 config (full parameter sharding) |
| `accelerate_config.yaml` | Accelerate config used with DeepSpeed for FSDP+ZeRO hybrid |

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

---

## Cross-Cutting Utilities

### `activate-env-variables-per-supercomputer.sh`

Sets machine-specific environment variables for NCCL, CUDA, and PyTorch. Sourced before training:

```bash
source activate-env-variables-per-supercomputer.sh
```

**Supported machines:**

| Machine | NCCL Settings | CUDA Devices |
|---------|---------------|--------------|
| `bsc-mn5-acc` | IB network, `ib0-ib3`, `mlx5_0-mlx5_5`, NVLS disabled | `0,1,2,3` |
| `leonardo` | `COMPILER=nvhpc`, `CUDA_HOME=/cineca/prod/CUDA/12.1` | — |

**Key variables set for `bsc-mn5-acc`:**
```bash
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=0
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
```

### `utils.sh` — Shell Utilities

| Function | Description |
|----------|-------------|
| `get_dataset_path(dataset, json_file)` | Extracts dataset path from JSON config using `jq` |
| `get_dataset_info(dataset, yaml_file)` | Extracts dataset path and handler from YAML using `yq` |
| `get_model_type(model, json_file)` | Resolves model identifier to model type |
| `get_model_directory(model_type, json_file)` | Resolves model type to filesystem path |
| `get_model_parallelism_config(model, parallelism, config_file)` | Gets parallelism config for a model |

### `gpu_plots.py` — GPU Utilization Plotting

Collects GPU metrics and generates matplotlib/seaborn visualizations:

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
2. Activate environment (source activate-env-variables-per-supercomputer.sh)
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
    bash -c "cd $LAUNCH_FOLDER && source activate-env-per-supercomputer.sh && python $TRAIN_SCRIPT ..."
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
    --dataloader_num_workers 32 \
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

1. **Create framework directory:** `scripts/<framework-name>-common/`

2. **Create launcher scripts:** `run-none.sh`, `run-ddp.sh`, `run-fsdp.sh` (or equivalent parallelism strategies)

3. **Create training entry points:** `finetune-none.py`, `finetune-ddp.py`, `finetune-fsdp.py`

4. **Create GPU monitor:** `gpu_monitor.py` (adapt from existing frameworks)

5. **Create utils:** `utils.py` with framework-specific utilities

6. **Register in config:** Add `configs/framework/<framework-name>.yaml` with:
   ```yaml
   name: <framework-name>
   parallelism:
     none: { min_gpus: 1, max_gpus: 1 }
     ddp: { min_gpus: 2 }
     fsdp: { min_gpus: 2 }
   scripts:
     run: scripts/<framework-name>-common/run-${parallelism}.sh
     finetune: scripts/<framework-name>-common/finetune-${parallelism}.py
     shared: scripts/shared
     copy_files:
       - scripts/<framework-name>-common/gpu_monitor.py
       - scripts/<framework-name>-common/utils.py
       - scripts/gpu_plots.py
       - ${scripts.run}
       - ${scripts.finetune}
   ```

7. **Register in Hydra:** Add to `dataclasses_hydra/__init__.py`:
   ```python
   cs.store(group="framework", name="<framework-name>", node=FrameworkConfig)
   ```

8. **Add to generator:** Add `<framework-name>` to `FRAMEWORKS` list in `hydra_app.py`.

---

## See Also

- [configs_hydra/README.md](../configs_hydra/README.md) — Configuration system
- [scripts/slurm/README.md](slurm/README.md) — SLURM job submission CLI
- [envs/README.md](../envs/README.md) — Environment management
- [training_MN5/README.md](../../README.md) — Root project overview
