# PyTorch TorchRun — Distributed Training Scripts

This directory contains training scripts and launchers for **PyTorch's native distributed training** via `torchrun` (formerly `torch.distributed.launch`). No external training frameworks — just pure PyTorch.

## Overview

TorchRun is PyTorch's built-in launcher for distributed training. It handles process initialization, environment variable setup, and rendezvous (communication backend) configuration. These scripts use HuggingFace's `SFTTrainer` from TRL with native PyTorch distributed primitives.

## Directory Structure

```
torchrun-common/
├── run-none.sh              # Single-GPU launcher
├── run-ddp.sh               # DDP (Distributed Data Parallel) launcher
├── run-fsdp.sh              # FSDP (Fully Sharded Data Parallel) launcher
├── finetune-none.py         # Single-GPU training entry point
├── finetune-ddp.py          # DDP training entry point
├── finetune-fsdp.py         # FSDP training entry point
├── gpu_monitor.py           # GPU metrics collection
└── utils.py                 # Shared utilities (rank printing, arg parsing, etc.)
```

## Launchers

### `run-none.sh` — Single-GPU Training

Runs training on a single GPU without any distributed parallelism. Sets `RANK=0`, `LOCAL_RANK=0`, `WORLD_SIZE=1` manually.

```bash
bash run-none.sh
```

**Configuration:**
- Direct Python execution: `python finetune-none.py`
- No distributed initialization
- Dataloader workers: 2 (optimized for single-GPU)

### `run-ddp.sh` — Distributed Data Parallel

Launches training across multiple GPUs using PyTorch's native DDP via `torchrun`. Each GPU holds a full model copy; gradients are synchronized via all-reduce.

```bash
bash run-ddp.sh
```

**Configuration:**
- Uses `torchrun` launcher with `--nnodes`, `--nproc_per_node`
- Rendezvous: `c10d` backend
- Master port: `29500`
- Job ID used as rendezvous ID (`--rdzv_id`)

### `run-fsdp.sh` — Fully Sharded Data Parallel

Launches training using PyTorch's FSDP with `torchrun`. Supports both minimum and maximum communication-computation overlap modes, writing outputs to separate directories (`$SLURM_JOB_ID-min-overlap` and `$SLURM_JOB_ID-max-overlap`).

```bash
bash run-fsdp.sh
```

**Configuration:**
- Uses `torchrun` launcher with `--nnodes`, `--nproc_per_node`
- Rendezvous: `c10d` backend
- Master port: `29500`
- Sharding strategy: `FULL_SHARD`
- Activation checkpointing: enabled

## Training Entry Points

### `finetune-none.py`

Single-GPU fine-tuning script. No distributed initialization required.

**Key features:**
- Uses `PerformanceTrackingSFTTrainer` (subclass of TRL's `SFTTrainer`)
- Raw dataset loading via `load_and_prepare_raw_dataset`
- Automatic tokenizer setup (pad token defaults to eos token)
- Precision support: `fp32`, `fp16`, `bf16`
- GPU monitoring via `GPUMonitorCallback`
- MFU tracking via `mfu_callback_from_hf_config`

### `finetune-ddp.py`

DDP training entry point launched via `torchrun`. Uses distributed initialization from environment variables set by `torchrun`.

**Key features:**
- Distributed rank/world_size detection from `dist.is_initialized()` or `RANK`/`WORLD_SIZE` env vars
- Uses `PerformanceTrackingSFTTrainer` (subclass of TRL's `SFTTrainer`)
- Raw dataset loading via `load_and_prepare_raw_dataset`
- Automatic tokenizer setup (pad token defaults to eos token)
- Precision support: `fp32`, `fp16`, `bf16`
- GPU monitoring via `GPUMonitorCallback`
- MFU tracking via `mfu_callback_from_hf_config`

### `finetune-fsdp.py`

FSDP training entry point using native PyTorch FSDP with `SFTTrainer`.

**Key features:**
- **Layer wrapping**: Auto-detects transformer layers via `get_fsdp_layer_to_wrap(model_name)`
- **FSDP config**:
  - `transformer_layer_cls_to_wrap`: Auto-detected from model architecture
  - `use_orig_params`: `True` (required for `torch.compile` + param groups)
  - `sharding_strategy`: `FULL_SHARD`
  - `activation_checkpointing`: `True` (with `use_reentrant=False`)
  - `cpu_ram_efficient_loading`: `True`
- Raw dataset loading via `load_and_prepare_raw_dataset`
- Supports `--max_comm_comp_overlap` flag for maximum comm-compute overlap

## GPU Monitoring

### `gpu_monitor.py`

Provides GPU metrics collection via NVIDIA's NVML library (`pynvml`). Identical to the other framework directories.

**`GPUMonitorCallback`** — Trainer callback logging per-step metrics:
- Average/peak GPU memory (GB)
- Average/peak GPU utilization (%)
- Average/peak GPU power (W)

**`start_gpu_monitor(interval_sec, n_gpus)`** — Background thread sampler:
- Returns `(stats_dict, stop_flag)`
- Stats: `mem`, `util`, `power`, `timestamps`

**`get_gpu_stats(n_gpus)`** — Current GPU statistics:
- `mem_used`: Per-GPU memory in GB
- `util`: Per-GPU utilization %
- `power`: Per-GPU power in watts

## Utilities

### `utils.py`

Shared argument parsing and utility functions:

**`print_rank(rank_or_msg, msg)`** — Rank-aware printing:
```python
print_rank("msg")       # All ranks
print_rank(0, "msg")    # Rank 0 only
```

**`count_parameters(model)`** — Returns `(trainable, total, trainable_pct)`

**`save_summary_stats_json(summary, output_file)`** — Saves training summary as JSON

**`parse_args()`** — CLI arguments:

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
| `--gradient_accumulation_steps` | int | `16` | Gradient accumulation |
| `--dataloader_num_workers` | int | `4` | Dataloader workers |
| `--max_steps` | float | `None` | Maximum training steps |
| `--enable_compile` | bool | `False` | Enable `torch.compile()` |
| `--max_comm_comp_overlap` | bool | `False` | Max FSDP comm-compute overlap |

## Execution Flow

```
SLURM job submission
    → run-{none|ddp|fsdp}.sh
        → training_activate_runtime_environment
        → training_build_runtime_prefix (optional Singularity)
        → torchrun --nnodes N --nproc_per_node G \
             --rdzv_id $JOB_ID --rdzv_backend c10d \
             --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
             finetune-{ddp|fsdp}.py
            → load_and_prepare_raw_dataset
            → AutoModelForCausalLM.from_pretrained
            → SFTConfig + SFTTrainer
            → trainer.train()
                → GPUMonitorCallback (per-step)
                → mfu_callback_from_hf_config (MFU tracking)
```

## Dependencies

- `torch` — PyTorch with `torchrun` and FSDP support
- `transformers` — HuggingFace model hub and tokenizers
- `trl` — Transformer Reinforcement Learning (SFTTrainer)
- `pynvml` — NVIDIA GPU monitoring
- `psutil` — System process utilities

## Notes

- All scripts expect SLURM environment variables (`SLURM_NNODES`, `SLURM_NTASKS_PER_NODE`, etc.)
- The `shared/` directory (parent) provides common utilities: `custom_train.py`, `data.py`, `flops.py`, `utils.py`
- GPU monitoring runs as a background thread during training
- Output is saved to `${LAUNCH_FOLDER}/output/`
- FSDP launcher writes to separate directories for min/max overlap modes
- No external training framework required — pure PyTorch distributed
