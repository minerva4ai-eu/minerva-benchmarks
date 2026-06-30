# DeepSpeed — Distributed Training Scripts

This directory contains training scripts and launchers for **Microsoft DeepSpeed**, a deep learning optimization library that provides ZeRO (Zero Redundancy Optimizer) for memory-efficient distributed training.

## Overview

DeepSpeed enables training large language models at scale through its ZeRO optimization stages (ZeRO-1, ZeRO-2, ZeRO-3, ZeRO-3-Offload). These scripts support both **pure DeepSpeed** (custom training loop) and **DeepSpeed + Accelerate** (HuggingFace Trainer integration) modes.

## Directory Structure

```
deepspeed-common/
├── configs/
│   ├── zero1.json               # ZeRO-1 configuration
│   ├── zero2.json               # ZeRO-2 configuration
│   ├── zero3.json               # ZeRO-3 configuration
│   └── accelerate_config.yaml   # Accelerate config for DeepSpeed integration
├── run-deepspeed.sh             # Pure DeepSpeed launcher
├── run-deepspeed-accelerate.sh  # DeepSpeed + Accelerate launcher
├── finetune-deepspeed-pure.py   # Pure DeepSpeed training entry point
├── finetune-deepspeed-accelerate.py  # DeepSpeed + Accelerate training entry point
├── gpu_monitor.py               # GPU metrics collection
├── metrics.py                   # Custom metrics callback
└── utils.py                     # Shared utilities and argument parsing
```

## Launchers

### `run-deepspeed.sh` — Pure DeepSpeed

Launches training using DeepSpeed's native `deepspeed` launcher with a custom training loop. This provides full control over the training process and DeepSpeed initialization.

```bash
bash run-deepspeed.sh
```

**Configuration:**
- Uses `deepspeed` command directly
- Config file: `configs/${ZERO_STAGE}.json` (e.g., `zero1.json`, `zero2.json`, `zero3.json`)
- Generates DeepSpeed hostfile automatically from SLURM node list for multi-node training
- Dynamic `zero_hpZ_partition_size` based on total GPU count

### `run-deepspeed-accelerate.sh` — DeepSpeed + Accelerate

Launches training using HuggingFace's `accelerate launch` with DeepSpeed config injected via `--config_file`. Combines Accelerate's process management with DeepSpeed's optimization.

```bash
bash run-deepspeed-accelerate.sh
```

**Configuration:**
- Uses `accelerate launch --config_file configs/accelerate_config.yaml`
- Accelerate config placeholders are dynamically replaced:
  - `{{MASTER_IP}}` → head node IP
  - `{{NUM_NODES}}` → number of SLURM nodes
  - `{{NUM_GPUS}}` → total GPU count
  - `{{path to ds_config.json}}` → DeepSpeed config path
  - `{{HPZ_PARTITION_SIZE}}` → total GPU count (for ZeRO-2/3)

## DeepSpeed Configurations

### `configs/zero1.json` — ZeRO-1

Partition optimizer states only across GPUs. Each GPU holds full model parameters and gradients.

**Use case:** Moderate model sizes where optimizer states are the primary memory bottleneck.

### `configs/zero2.json` — ZeRO-2

Partitions both gradients and optimizer states across GPUs. Each GPU holds full model parameters.

**Use case:** Larger models where gradient memory is significant.

### `configs/zero3.json` — ZeRO-3

Partitions parameters, gradients, and optimizer states across all GPUs. Maximum memory efficiency.

**Use case:** Very large models that don't fit on a single GPU's memory.

### `configs/accelerate_config.yaml` — Accelerate + DeepSpeed

HuggingFace Accelerate configuration that delegates training to DeepSpeed. Contains placeholders for dynamic SLURM environment values.

## Training Entry Points

### `finetune-deepspeed-pure.py`

Pure DeepSpeed training script with a custom training loop. Uses DeepSpeed's native API for model initialization and training.

**Key features:**
- Manual `deepspeed.initialize()` call with model, optimizer, and scheduler
- Custom training loop with step timing and MFU calculation
- Distributed info handling via `get_dist_info()`
- FLOPs estimation: `6 × num_params` FLOPs per token
- MFU computation: `achieved_tflops / peak_tflops`
- GPU monitoring via background thread (`start_gpu_monitor`)
- Precision support: `fp32`, `fp16`, `bf16`

**MFU Helpers:**
```python
estimate_flops_per_token(model)  # 6 × num_params
compute_mfu(flops_per_token, tokens_per_step, step_time, peak_tflops)
```

### `finetune-deepspeed-accelerate.py`

DeepSpeed + Accelerate training using HuggingFace's `SFTTrainer`. Leverages Accelerate's process management and TRL's supervised fine-tuning capabilities.

**Key features:**
- Uses `PerformanceTrackingSFTTrainer` (subclass of TRL's `SFTTrainer`)
- Raw dataset loading via `load_and_prepare_raw_dataset`
- Automatic tokenizer setup (pad token defaults to eos token)
- Precision support: `fp32`, `fp16`, `bf16`
- GPU monitoring via `GPUMonitorCallback`
- MFU tracking via `mfu_callback_from_hf_config`
- Gradient checkpointing support via `--gradient_checkpointing`

## GPU Monitoring

### `gpu_monitor.py`

Identical to the accelerate-common version. Provides GPU metrics via NVIDIA NVML:

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

## Metrics

### `metrics.py`

**`CustomMetricsCallback`** — Trainer callback for additional logging:
- GPU memory allocation tracking (`gpu_mem_gb`)
- Cumulative tokens processed (`cumulative_tokens`)
- Per-step timing (`step_time`)
- Formatted step metrics output

## Utilities

### `utils.py`

Shared argument parsing and utilities:

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
| `--warmup_ratio` | float | `0` | LR warmup ratio |
| `--weight_decay` | float | `2e-5` | Weight decay |
| `--max_length` | int | `1024` | Max token length |
| `--precision` | str | `fp32` | Precision: `fp32`, `fp16`, `bf16` |
| `--gradient_accumulation_steps` | int | `16` | Gradient accumulation |
| `--dataloader_num_workers` | int | `4` | Dataloader workers |
| `--deepspeed_config_file` | str | `None` | Path to DeepSpeed config |
| `--gradient_checkpointing` | bool | `False` | Enable gradient checkpointing |
| `--enable_compile` | bool | `False` | Enable `torch.compile()` |
| `--max_steps` | float | `None` | Maximum training steps |

## Execution Flow

### Pure DeepSpeed
```
SLURM job submission
    → run-deepspeed.sh
        → training_activate_runtime_environment
        → Generate hostfile from SLURM nodes
        → Update DeepSpeed config (hpZ partition size)
        → deepspeed finetune-deepspeed-pure.py
            → deepspeed.initialize(model, optimizer, scheduler)
            → Custom training loop
                → MFU calculation per step
                → GPU monitoring (background thread)
```

### DeepSpeed + Accelerate
```
SLURM job submission
    → run-deepspeed-accelerate.sh
        → training_activate_runtime_environment
        → Generate hostfile from SLURM nodes
        → Update configs (hpZ, master IP, node count)
        → accelerate launch --config_file accelerate_config.yaml
            → finetune-deepspeed-accelerate.py
                → SFTConfig + SFTTrainer
                → trainer.train()
                    → GPUMonitorCallback (per-step)
                    → mfu_callback_from_hf_config (MFU)
```

## Dependencies

- `deepspeed` — Microsoft DeepSpeed optimization library
- `transformers` — HuggingFace model hub and tokenizers
- `trl` — Transformer Reinforcement Learning (SFTTrainer)
- `accelerate` — HuggingFace distributed training launcher
- `torch` — PyTorch
- `pynvml` — NVIDIA GPU monitoring
- `psutil` — System process utilities

## Notes

- All scripts expect SLURM environment variables (`SLURM_NNODES`, `SLURM_NTASKS_PER_NODE`, etc.)
- The `shared/` directory (parent) provides common utilities: `custom_train.py`, `data.py`, `flops.py`, `utils.py`
- DeepSpeed hostfile is auto-generated for multi-node training
- `zero_hpZ_partition_size` is dynamically set to total GPU count for ZeRO-2/3
- Output is saved to `${LAUNCH_FOLDER}/output/`
- GPU monitoring runs as a background thread during training
