# Hydra Configuration System

This directory contains the Hydra-based configuration system for generating, validating, and composing LLM training benchmark experiments. It replaces the older static JSON/YAML config approach with a type-safe, composable, and constraint-driven architecture.

## Overview

The configuration system works in four stages:

1. **Schema Definition** — Python dataclasses define the contract for every config field (types, defaults, validation).
2. **YAML Registration** — Hydra's `ConfigStore` registers dataclass schemas and YAML files under named groups.
3. **Config Composition** — At runtime, Hydra composes a `BenchmarkConfig` by merging a base config with overrides for model, framework, dataset, and parallelism.
4. **Constraint Validation** — Rule-based checks filter out invalid combinations (GPU counts, framework support, memory limits) before any jobs are submitted.

### How It All Fits Together

The system generates **all valid training configurations** by computing the Cartesian product of:

```
MODELS × FRAMEWORKS × DATASETS × PARALLELISM_STRATEGIES × TRAINING_HYPERPARAMETERS
```

For example, with 5 models (`llama3_8b`, `gemma3_1b`, `gemma3_12b`, `mistral_7b`, `llama3_70b`), 3 frameworks (`accelerate`, `torchrun`, `deepspeed-accelerate`), and 2 datasets (`alpaca`, `squadv2`), the generator produces **30 base configs** (before constraint filtering). Each base config is then expanded by the training hyperparameters (batch sizes × precisions × gradient accumulations × learning rates × optimizers × gradient checkpointing × enable_compile), potentially yielding **hundreds of individual benchmark jobs**.

All invalid combinations are filtered out by constraint rules before any jobs are submitted, ensuring that only feasible configurations reach the SLURM scheduler.

## Architecture

The system is organized into four layers:

```
configs_hydra/
├── dataclasses_hydra/     # Python dataclass schemas (the "contract")
├── constraints/           # Rule-based validation
├── configs/               # YAML config files organized by Hydra group
└── hydra_app.py           # Orchestrator: composition + validation + combo generation
```

---

## 1. Dataclass Schemas (`dataclasses_hydra/`)

Each dataclass defines the structure and validation rules for a config section. They are registered with Hydra's `ConfigStore` in `dataclasses_hydra/__init__.py`.

### `BenchmarkConfig` (`benchmark.py`)

The root config that ties everything together. Contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique benchmark identifier |
| `trainings` | `TrainArgsConfig` | Training hyperparameters (batch sizes, precisions, LR, etc.) |
| `arch` | `HPCArchitecture` | HPC architecture specs (GPU VRAM, TFLOPs, node config) |
| `model` | `ModelConfig` | Model architecture dimensions, GPU requirements |
| `dataset` | `DatasetConfig` | Dataset name, path, task type, sequence length |
| `framework` | `FrameworkConfig` | Framework name, parallelism specs, script paths |
| `machine` | `MachineConfig` | Supercomputer identity, modules, Singularity container |
| `experiment` | `ExperimentConfig` | Experiment name, output dir, repeat count |
| `slurm` | `SlurmConfig` | SLURM account, QoS, partition, sbatch/srun params |

### `ModelConfig` (`model.py`)

Defines model-specific parameters:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Model identifier (e.g., `llama3_8b`) |
| `path` | `str` | Path to model weights |
| `params_billions` | `float` | Total parameters in billions |
| `architecture_type` | `ArchitectureType` | `dense`, `moe`, or `ssm` |
| `num_layers` | `int` | Number of transformer layers |
| `hidden_dim` | `int` | Hidden dimension size |
| `ffn_intermediate_dim` | `int` | FFN intermediate dimension |
| `num_attention_heads` | `int` | Number of attention heads |
| `num_kv_heads` | `int` | Number of KV heads (GQA) |
| `vocab_size` | `int` | Vocabulary size |
| `head_dim` | `int` | Attention head dimension |
| `attention_type` | `str` | Attention variant (`mha`, `gqa`, etc.) |
| `tokenizer_type` | `str` | Tokenizer type (e.g., `tiktoken`) |
| `max_gpus_scale` | `int` | Maximum GPUs the model can scale to |
| `frameworks_supported` | `List[str]` | Supported frameworks (`torchrun`, `accelerate`, `deepspeed`) |
| `parallelism_supported` | `List[str]` | Supported parallelism strategies |
| `active_params_billions` | `Optional[float]` | MoE only: active parameters |
| `num_experts` | `Optional[int]` | MoE only: number of experts |
| `top_k_experts` | `Optional[int]` | MoE only: top-k experts per token |

**Validation**: MoE models require `active_params_billions`, `num_experts`, and `top_k_experts`. Architecture type is validated against the `ArchitectureType` enum.

**Currently registered models**: `llama3_8b`, `mistral_7b`, `llama3_70b`. Each has a corresponding YAML file in `configs/model/` with architecture dimensions and GPU requirements. Machine-specific path overrides are stored in `*-MN5.yaml` files. Note: `gemma3_1b` and `gemma3_12b` have YAML files but are not yet registered in `dataclasses_hydra/__init__.py`.

### `FrameworkConfig` (`framework.py`)

Defines framework-specific settings:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Framework name (`torchrun`, `accelerate`, `deepspeed`, `deepspeed-accelerate`) |
| `parallelism_name` | `str` | Current parallelism strategy (set at composition time) |
| `parallelism` | `Dict[str, ParallelismSpec]` | Map of parallelism strategy → GPU constraints |
| `scripts` | `ScriptsConfig` | Paths to run script, finetune script, shared code, and files to copy |

**`ParallelismSpec`**:

| Field | Type | Description |
|-------|------|-------------|
| `min_gpus` | `int` | Minimum GPUs required |
| `max_gpus` | `int` | Maximum GPUs allowed (default: 999) |

**`ScriptsConfig`**:

| Field | Type | Description |
|-------|------|-------------|
| `run` | `str` | Path to SLURM launcher script (supports `${framework.parallelism_name}` interpolation) |
| `finetune` | `str` | Path to training entry point script |
| `shared` | `str` | Path to shared code directory |
| `copy_files` | `List[str]` | Additional files to copy to launch folder |

### `DatasetConfig` (`dataset.py`)

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Dataset identifier (e.g., `alpaca`) |
| `path` | `str` | Path to dataset file(s) |
| `task` | `str` | Training task (`Instruction-Finetuning` or `Pretraining`) |
| `train` | `str` | Train split identifier |
| `validation` | `Optional[str]` | Validation split identifier |
| `max_seq_len` | `int` | Maximum sequence length |

### `SlurmConfig` (`slurm.py`)

| Field | Type | Description |
|-------|------|-------------|
| `account` | `str` | SLURM account |
| `qos` | `str` | SLURM QoS |
| `partition` | `str` | SLURM partition |
| `sbatch` | `SbatchConfig` | sbatch directives (nodes, gpus_per_node, time, gres, etc.) |
| `srun` | `SrunConfig` | srun directives |

### `HPCArchitecture` (`arch.py`)

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Architecture name (e.g., `MN5`) |
| `gpus` | `Dict[str, GpuConfig]` | GPU types with VRAM, TFLOPs per precision |
| `node` | `NodeConfig` | GPUs per node |
| `comm` | `CommConfig` | Inter-node and intra-node communication specs |

**`GpuConfig`** includes peak TFLOPs for each precision type (`fp32`, `bf16`, `fp16`, `fp8`) and a helper function `get_peak_flops(cfg, precision)`.

### `TrainArgsConfig` (`benchmark.py`)

Defines training hyperparameter combinations that the generator expands into individual experiments:

| Field | Type | Description |
|-------|------|-------------|
| `batch_sizes` | `List[int]` | Batch sizes (required, ≥ 1) |
| `precisions` | `List[str]` | Precision types (`fp32`, `bf16`, `fp16`, `fp8`) |
| `grad_accums` | `List[int]` | Gradient accumulation steps |
| `lr` | `List[float]` | Learning rates (> 0) |
| `optimizer` | `List[str]` | Optimizers (`adam`, `adamw`, `sgd`, `adafactor`) |
| `gradient_checkpointing` | `List[bool]` | `[True]`, `[False]`, or `[True, False]` |
| `steps` | `Optional[List[int]]` | Max training steps (mutually exclusive with epochs) |
| `epochs` | `Optional[List[int]]` | Training epochs (mutually exclusive with steps) |
| `enable_compile` | `List[bool]` | Enable PyTorch compile (`True`/`False`) |

---

## 2. Constraints (`constraints/`)

Rule-based validation filters out invalid config combinations before job submission.

### `ConstraintRule` (`base.py`)

Abstract base class:

```python
class ConstraintRule(ABC):
    @abstractmethod
    def check(self, c: BenchmarkConfig) -> RuleResult:
        ...
```

### `RuleResult` (`base.py`)

```python
@dataclass
class RuleResult:
    passed: bool
    rule_name: str
    reason: str = ""
```

### Built-in Rules (`rules.py`)

| Rule | Purpose |
|------|---------|
| `ParallelismGPUFloor` | Checks that GPU count (nodes × gpus_per_node) is within `[min_gpus, max_gpus]` for the selected parallelism strategy |
| `FrameworkParallelismValidityRule` | Verifies that the selected framework supports the chosen parallelism strategy |
| Memory/parameter rules | Estimate VRAM requirements based on `BYTES_PER_PARAM` and `OPTIMIZER_BYTES` tables, scale by `MAX_GPUS_SCALE`, and check against available GPU VRAM |

**VRAM Estimation Math**: The memory rules use two lookup tables to estimate VRAM requirements:

- **`BYTES_PER_PARAM`** — Maps precision types to bytes per parameter (e.g., `fp32=4`, `bf16=2`, `fp16=2`).
- **`OPTIMIZER_BYTES`** — Maps optimizer types to bytes per parameter for optimizer states (e.g., `adamw=8` for Adam's 2 moment estimates in fp32, `sgd=4` for momentum in fp32).

The total VRAM estimate per GPU is computed as:

```
total_vram = (params_billions × 1e9 × bytes_per_param) / total_gpus
           + (params_billions × 1e9 × optimizer_bytes) / total_gpus
           + activation_overhead + headroom
```

This estimate is then compared against the available GPU VRAM from the architecture config. If the estimate exceeds available VRAM, the config is skipped with a reason explaining the memory shortfall.

**Adding a custom rule**: Create a new class inheriting from `ConstraintRule`, implement `check()`, and add it to the validation pipeline in `hydra_app.py`.

---

## 3. YAML Configs (`configs/`)

YAML files are organized by Hydra group and registered in `dataclasses_hydra/__init__.py`. Each group maps to a dataclass schema.

### Directory Structure

```
configs/
├── base.yaml              # Root config — defines defaults, machine, experiment settings
├── MN5.yaml               # MN5-specific overrides (extends base, adds arch + slurm)
├── MN5-singularity.yaml   # MN5 with Singularity container config
├── MN5-uv-venv.yaml       # MN5 with uv venv Python environment
├── arch/
│   └── MN5.yaml           # MareNostrum5 GPU specs (H100-SXM, 64GB VRAM, TFLOPs)
├── dataset/
│   ├── base.yaml          # Base dataset template
│   ├── alpaca.yaml        # Alpaca instruction-tuning dataset
│   ├── alpaca-MN5.yaml    # MN5-specific path override for Alpaca
│   ├── squadv2.yaml       # SQuAD v2 question-answering dataset
│   └── squadv2-MN5.yaml   # MN5-specific path override for SQuAD v2
├── framework/
│   ├── base.yaml          # Base framework template (empty)
│   ├── accelerate.yaml    # HuggingFace Accelerate (ddp, fsdp)
│   ├── deepspeed.yaml     # Microsoft DeepSpeed (zero1, zero2, zero3, zero3-offload)
│   ├── deepspeed-accelerate.yaml  # DeepSpeed with Accelerate integration (zero1, zero2, zero3, zero3-offload)
│   └── torchrun.yaml      # PyTorch native (none, ddp, fsdp)
├── model/
│   ├── base_training.yaml # Base model template + training hyperparameter combinations
│   ├── llama3_8b.yaml     # LLaMA 3 8B (dense)
│   ├── llama3_8b-MN5.yaml # MN5-specific path override
│   ├── llama3_70b.yaml    # LLaMA 3 70B (dense)
│   ├── llama3_70b-MN5.yaml
│   ├── mistral_7b.yaml    # Mistral 7B (dense)
│   ├── mistral_7b-MN5.yaml
│   ├── gemma3_1b.yaml     # Gemma 3 1B (dense)
│   ├── gemma3_1b-MN5.yaml
│   ├── gemma3_12b.yaml    # Gemma 3 12B (dense)
│   └── gemma3_12b-MN5.yaml
├── slurm/
│   ├── base.yaml          # Base SLURM template (empty — values filled by overrides)
│   └── MN5.yaml           # MareNostrum5 SLURM settings (account, QoS, partition, 4 GPUs/node)
└── trainings/             # (not used — training combos are in base_training.yaml)
```

### Config Composition

Hydra resolves configs through a **defaults list** in `base.yaml`. The defaults list defines the order in which configs are merged:

```yaml
defaults:
  - model: base_training       # Load base model template
  - framework: base            # Load base framework template
  - dataset: base              # Load base dataset template
  - _self_                     # Merge base.yaml's own fields last
```

Note: Training hyperparameters are loaded via `base_training.yaml` (not a separate `trainings` group). The `_self_` key ensures that `base.yaml`'s own fields take precedence over any conflicting fields from the included configs.

The `MN5.yaml` extends `base` and adds architecture/SLURM overrides:

```yaml
defaults:
  - base                       # Start with base.yaml
  - arch: MN5                  # Override with MN5 GPU specs
  - slurm: MN5                 # Override with MN5 SLURM settings
  - _self_                     # Merge MN5.yaml's own fields last
```

Machine-specific environment configurations are available as additional layers:

- `MN5-singularity.yaml` — extends `MN5` with Singularity container path and args
- `MN5-uv-venv.yaml` — extends `MN5` with uv virtual environment path

These can be used as the `config_name` when composing configs to select the desired runtime environment.

**Variable interpolation** is supported throughout the config system using Hydra's interpolation syntax `${group.field}`. This allows configs to reference values from other groups without duplication:

```yaml
# In framework/accelerate.yaml
run: scripts/accelerate-common/run-${framework.parallelism_name}.sh
finetune: scripts/accelerate-common/finetune-${framework.parallelism_name}.py

# In slurm/MN5.yaml
gres: gpu:${arch.node.gpus_per_node}

# In framework/deepspeed.yaml
scripts:
  run: scripts/deepspeed-common/run-deepspeed.sh
  finetune: scripts/deepspeed-common/finetune-deepspeed-pure.py
  copy_files:
    - scripts/deepspeed-common/gpu_monitor.py
    - scripts/deepspeed-common/utils.py
    - scripts/gpu_plots.py
    - scripts/deepspeed-common/configs
    - ${framework.scripts.run}
    - ${framework.scripts.finetune}

# In framework/deepspeed-accelerate.yaml
scripts:
  run: scripts/deepspeed-common/run-deepspeed-accelerate.sh
  finetune: scripts/deepspeed-common/finetune-deepspeed-accelerate.py
```

When composing a config, Hydra resolves all interpolations to produce a fully concrete `BenchmarkConfig` object.

### Training Combinations Expansion

Training hyperparameter combinations are defined in `configs/model/base_training.yaml` under the `combinations` key. The `TrainArgsConfig` defines hyperparameter lists that are expanded into individual experiments via **Cartesian product**. For example, if `base_training.yaml` specifies:

```yaml
combinations:
  batch_sizes: [1, 4, 8]
  precisions: ["bf16"]
  grad_accums: [8, 16]
  lr: [2e-5]
  optimizer: ["adamw"]
  gradient_checkpointing: [False]
  steps: [50]
  enable_compile: [False, True]
```

The generator produces `3 × 1 × 2 × 1 × 1 × 1 × 1 × 2 = 24` individual configs, each with a unique combination of hyperparameters.

**Mutual exclusivity**: `steps` and `epochs` are mutually exclusive — you cannot specify both in the same config. If both are provided, the config is skipped.

### `single_gpu_also_valid` Logic

Some architectures support both single-GPU and multi-node configurations. The `single_gpu_also_valid` flag in the `machine` config (not architecture) controls whether 1-GPU configs are generated in addition to full-node configs:

```yaml
# In base.yaml or MN5.yaml
machine:
  single_gpu_also_valid: True
```

When `single_gpu_also_valid` is `true` AND the parallelism strategy has `min_gpus: 1` AND running on 1 node, the generator produces configs for both 1 GPU and 4 GPUs (nodes × gpus_per_node). When `false`, only full-node configs are generated. This is useful for architectures where single-GPU testing is meaningful (e.g., for quick validation) but not for production-scale training.

### Adding a New Model

1. **Create the model YAML file** at `configs/model/<name>.yaml` with all required fields from `ModelConfig`. Example for a new model:

   ```yaml
   # configs/model/phi3_3b.yaml
   name: phi3_3b
   path: /path/to/phi3-3b
   params_billions: 3.8
   architecture_type: dense
   num_layers: 32
   hidden_dim: 3072
   ffn_intermediate_dim: 8192
   num_attention_heads: 32
   num_kv_heads: 32
   vocab_size: 32064
   head_dim: 96
   attention_type: mha
   tokenizer_type: tiktoken
   max_gpus_scale: 8
   frameworks_supported:
     - accelerate
     - torchrun
     - deepspeed
   parallelism_supported:
     - none
     - ddp
     - fsdp
     - zero1
     - zero2
     - zero3
   ```

2. **Register the model** in `dataclasses_hydra/__init__.py` by adding it to the `MODELS` list:

   ```python
   MODELS = ["llama3_8b", "gemma3_1b", "gemma3_12b", "mistral_7b", "llama3_70b", "phi3_3b"]
   ```

   Note: The dataclass schema (`ModelConfig`) is already registered in `__init__.py`. You need to add both a `cs.store()` call for the model name AND add it to the `MODELS` list in `hydra_app.py`.

3. **(Optional) Add machine-specific overrides** at `configs/model/<name>-MN5.yaml` to override the model path for a specific machine:

   ```yaml
   # configs/model/phi3_3b-MN5.yaml
   path: /gpfs/scratch/phi3-3b-mn5
   ```

### Adding a New Dataset

1. **Create the dataset YAML file** at `configs/dataset/<name>.yaml` with `name`, `path`, `task`, `train`, `max_seq_len`. Example:

   ```yaml
   # configs/dataset/fineweb.yaml
   name: fineweb
   path: /path/to/fineweb/train.jsonl
   task: Pretraining
   train: train
   max_seq_len: 4096
   ```

2. **Register the dataset** in `dataclasses_hydra/__init__.py` by adding it to the `DATASETS` list:

   ```python
   DATASETS = ["alpaca", "squadv2", "fineweb"]
   ```

3. **(Optional) Create a dataset handler** in `scripts/shared/datasets/` if the dataset requires custom preprocessing. Register the handler in `DATASET_HANDLER_MAP`.

### Adding a New Framework

1. **Create the framework YAML file** at `configs/framework/<name>.yaml` with `name`, `parallelism` specs, and `scripts` paths. Example:

   ```yaml
   # configs/framework/vllm.yaml
   name: vllm
   parallelism:
     none: { min_gpus: 1, max_gpus: 1 }
     tensor_parallel: { min_gpus: 2, max_gpus: 8 }
   scripts:
     run: scripts/vllm-common/run.sh
     finetune: scripts/vllm-common/serve.py
     shared: scripts/shared
     copy_files:
       - scripts/vllm-common/gpu_monitor.py
       - scripts/vllm-common/utils.py
   ```

2. **Register the framework** in `dataclasses_hydra/__init__.py` by adding it to the `FRAMEWORKS` list:

   ```python
   FRAMEWORKS = ["accelerate", "torchrun", "deepspeed", "vllm"]
   ```

3. **Create the corresponding script directories** under `scripts/<name>-common/` with launcher scripts, training entry points, and utilities.

---

## 4. Orchestrator (`hydra_app.py`)

The main entry point for config generation and validation.

### Key Functions

#### `generate_valid_combos(config_path, config_name, outpath)`

The core function that orchestrates the entire config generation pipeline:

```python
def generate_valid_combos(config_path, config_name, outpath):
    """
    Generates all valid benchmark configurations and saves them to outpath.
    
    Returns:
        tuple: (valid_configs, skipped_reasons)
            - valid_configs: List[BenchmarkConfig] - all valid configs
            - skipped_reasons: List[str] - reasons why configs were skipped
    """
```

**Detailed execution flow:**

1. **Iterate over combinations**: For each `(model, framework, dataset)` triple from the `MODELS`, `FRAMEWORKS`, `DATASETS` lists:
   - Compose a base config via Hydra: `compose(config_name, overrides=[f"model={model}", f"framework={framework}", f"dataset={dataset}"])`
   - For each parallelism strategy in `cfg.framework.parallelism_supported`:
     - Create a deep copy of the config
     - Set `cfg.framework.parallelism_name` to the current strategy
     - Run all constraint rules against the combo
     - If all rules pass, add to valid list; otherwise, record the skip reason

2. **Expand training hyperparameters**: For each valid base config:
   - Compute the Cartesian product of all training hyperparameter lists
   - For each combination, create a new `BenchmarkConfig` with the specific hyperparameters
   - Each expanded config gets a unique `id` based on model, framework, dataset, parallelism, and hyperparameters

3. **Save and return**: Save all valid configs as YAML files to `outpath` and return `(valid, skipped)` tuples.

#### `expand_arch_gpu_configs(cfg)`

Replicates the GPU configuration logic: if `single_gpu_also_valid` is True and the architecture supports it, returns both `[1, gpus_per_node]`; otherwise returns `[gpus_per_node]`.

This function is called during the combo generation to determine which GPU counts to generate configs for. For example, on MN5 with 4 GPUs per node and `single_gpu_also_valid=true`, it returns `[1, 4]`, meaning configs for both 1-GPU and 4-GPU (1-node) runs are generated.

### Usage

```python
from configs_hydra.hydra_app import generate_valid_combos

valid, skipped = generate_valid_combos(
    config_path="./configs_hydra/configs",
    config_name="base",
    outpath="benchmark-runs/"
)

print(f"Valid configs: {len(valid)}")
print(f"Skipped configs: {len(skipped)}")
for reason in skipped:
    print(f"  Skipped: {reason}")
```

### Typical Output

When run on MN5 with the default MODELS, FRAMEWORKS, and DATASETS lists, `generate_valid_combos()` typically produces:

- **~50-100 valid base configs** (after constraint validation)
- **~500-2000 expanded configs** (after training hyperparameter expansion)
- **~20-50 skipped configs** with reasons like "GPU count exceeds max for parallelism" or "VRAM estimate exceeds available"

The exact numbers depend on the models, frameworks, and datasets configured in `hydra_app.py`.

---

## Quick Reference: Config Flow

```
MODELS × FRAMEWORKS × DATASETS
        ↓
   Hydra compose()  →  BenchmarkConfig
        ↓
   Constraint rules  →  valid / skipped
        ↓
   Expand trainings  →  individual combos
        ↓
   Save YAML + submit to SLURM
```

## See Also

- [training_MN5/README.md](../README.md) — Root project overview
- [scripts/README.md](../scripts/README.md) — Training scripts and launchers
- [scripts/slurm/README.md](../scripts/slurm/README.md) — SLURM job submission CLI
- [envs/README.md](../envs/README.md) — Environment management
