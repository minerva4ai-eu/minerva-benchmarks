# HuggingFace Accelerate — Distributed Training Scripts

This directory contains training scripts and launchers for **HuggingFace Accelerate**, a library that simplifies multi-GPU and multi-node distributed training without requiring changes to the core training logic.

## Overview

Accelerate abstracts away the complexity of distributed training by providing a simple `accelerate launch` command that handles process initialization, device placement, and communication backend setup. These scripts are built on top of HuggingFace's `SFTTrainer` (from TRL) and `Trainer` APIs, with custom performance tracking callbacks.

## Directory Structure

```
accelerate-common/
├── run-none.sh              # Single-GPU launcher
├── run-ddp.sh               # DDP (Distributed Data Parallel) launcher
├── run-fsdp.sh              # FSDP (Fully Sharded Data Parallel) launcher
├── finetune-ddp.py          # DDP training entry point
├── finetune-fsdp.py         # FSDP training entry point (native PyTorch FSDP)
├── gpu_monitor.py           # GPU metrics collection (memory, utilization, power)
└── utils.py                 # Shared utilities (arg parsing, parameter counting, etc.)
```

## Launchers

### `run-none.sh` — Single-GPU Training

Runs training on a single GPU without any distributed parallelism. Useful for debugging, prototyping, or small models.

```bash
bash run-none.sh
```

### `run-ddp.sh` — Distributed Data Parallel

Launches training across multiple GPUs on a single node (or multi-node) using DDP. Each GPU holds a full copy of the model and gradients are synchronized via all-reduce.

```bash
bash run-ddp.sh
```

**Configuration:**
- Uses `accelerate launch` with `--multi-gpu` flag
- Communication backend: `c10d`
- Master port: `29500`

### `run-fsdp.sh` — Fully Sharded Data Parallel

Launches training using PyTorch's FSDP, which shards model parameters, gradients, and optimizer states across GPUs. Supports both minimum and maximum communication-computation overlap modes.

```bash
bash run-fsdp.sh
```

**Configuration:**
- Uses `accelerate launch` with `--multi-gpu` flag
- Communication backend: `c10d`
- Sharding strategy: `FULL_SHARD`
- Activation checkpointing: enabled

## Training Entry Points

### `finetune-ddp.py`

Single-process fine-tuning script designed to be launched via `accelerate launch` with DDP parallelism. Uses `PerformanceTrackingSFTTrainer` (subclass of TRL's `SFTTrainer`) with:

- Raw dataset loading via `load_and_prepare_raw_dataset`
- Automatic tokenizer setup (pad token defaults to eos token)
- Precision support: `fp32`, `fp16`, `bf16`
- GPU monitoring via `GPUMonitorCallback`
- MFU (Model FLOPs Utilization) tracking via `mfu_callback_from_hf_config`

### `finetune-fsdp.py`

FSDP training entry point using native PyTorch FSDP with `SFTTrainer`. Key features:

- **Layer wrapping**: Automatically determines transformer layers to wrap via `get_fsdp_layer_to_wrap(model_name)`
- **FSDP config**:
  - `transformer_layer_cls_to_wrap`: Auto-detected from model architecture
  - `use_orig_params`: `True` (required for `torch.compile` + param groups)
  - `sharding_strategy`: `FULL_SHARD`
  - `activation_checkpointing`: `True` (with `use_reentrant=False`)
  - `cpu_ram_efficient_loading`: `True`
- Supports `--max_comm_comp_overlap` flag for maximum comm-compute overlap (sets `forward_prefetch=True`, `limit_all_gathers=False`)

### `finetune-fsdp-Trainer.py`

FSDP training entry point using Accelerate's `Trainer` API (not `SFTTrainer`). Uses the older `load_dataset` function that returns pre-tokenized datasets with a custom `collate_fn`.

**Key differences from `finetune-fsdp.py`:**
- Uses `PerformanceTrackingTrainer` instead of `PerformanceTrackingSFTTrainer`
- Pre-tokenized datasets with explicit `collate_fn`
- Uses `init_on_device` context for faster model initialization

## GPU Monitoring

### `gpu_monitor.py`

Provides GPU metrics collection via NVIDIA's NVML library (`pynvml`). Two main interfaces:

**`GPUMonitorCallback`** — Trainer callback that logs metrics at each training step:
- Average/peak GPU memory (GB)
- Average/peak GPU utilization (%)
- Average/peak GPU power (W)

**`start_gpu_monitor()`** — Background thread that samples GPU stats at a configurable interval (default: 5 seconds):
- Returns `(stats_dict, stop_flag)` for manual control
- Stats include: `mem`, `util`, `power`, `timestamps`

**`get_gpu_stats(n_gpus)`** — Returns current GPU statistics:
- `mem_used`: Per-GPU memory usage in GB
- `util`: Per-GPU utilization percentage
- `power`: Per-GPU power consumption in watts

## Utilities

### `utils.py`

Shared argument parsing and utility functions:

**`parse_args()`** — Defines CLI arguments for all training scripts:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | *required* | Path to pretrained model |
| `--data` | str | *required* | Path to JSON dataset |
| `--dataset` | str | *required* | Dataset name |
| `--output_dir` | str | `./output` | Output directory |
| `--epochs` | int | `None` | Number of training epochs |
| `--batch_size` | int | `1` | Per-device batch size |
| `--lr` | float | `0.01` | Learning rate |
| `--weight_decay` | float | `2e-5` | Weight decay |
| `--max_length` | int | `1024` | Max token length |
| `--precision` | str | `fp32` | Precision: `fp32`, `fp16`, `bf16` |
| `--gradient_accumulation_steps` | int | `16` | Gradient accumulation steps |
| `--dataloader_num_workers` | int | `4` | Dataloader workers |
| `--max_steps` | float | `None` | Maximum training steps |
| `--enable_compile` | bool | `False` | Enable `torch.compile()` |
| `--max_comm_comp_overlap` | bool | `False` | Max FSDP comm-compute overlap |

**Helper functions:**
- `count_parameters(model)` — Returns `(trainable, total, trainable_pct)`
- `save_summary_stats_json(summary, output_file)` — Saves training summary as JSON
- `print_rank(rank_or_msg, msg)` — Rank-aware printing utility

## Execution Flow

```
SLURM job submission
    → run-{none|ddp|fsdp}.sh
        → training_activate_runtime_environment
        → training_build_runtime_prefix (optional Singularity)
        → accelerate launch --multi-gpu ...
            → finetune-{ddp|fsdp}.py
                → load_and_prepare_raw_dataset
                → AutoModelForCausalLM.from_pretrained
                → SFTConfig + SFTTrainer
                → trainer.train()
                    → GPUMonitorCallback (per-step)
                    → mfu_callback_from_hf_config (MFU tracking)
```

## Dependencies

- `accelerate` — HuggingFace distributed training launcher
- `transformers` — HuggingFace model hub and tokenizers
- `trl` — Transformer Reinforcement Learning (SFTTrainer)
- `torch` — PyTorch with FSDP support
- `pynvml` — NVIDIA GPU monitoring
- `psutil` — System process utilities

## Notes

- All scripts expect SLURM environment variables (`SLURM_NNODES`, `SLURM_NTASKS_PER_NODE`, etc.)
- The `shared/` directory (parent) provides common utilities: `custom_train.py`, `data.py`, `flops.py`, `utils.py`
- GPU monitoring runs as a background thread during training
- Output is saved to `${LAUNCH_FOLDER}/output/`
