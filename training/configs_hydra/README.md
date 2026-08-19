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
| `total_params_billions` | `float` | Total parameters in billions |
| `architecture_type` | `ArchitectureType` | `dense`, `moe`, or `ssm` |
| `num_layers` | `int` | Number of transformer layers |
| `hidden_dim` | `int` | Hidden dimension size |
| `ffn_intermediate_dim` | `int` | FFN intermediate dimension |
| `num_attention_heads` | `int` | Number of attention heads |
| `num_kv_heads` | `int` | Number of KV heads (GQA/MQA) |
| `vocab_size` | `int` | Vocabulary size |
| `head_dim` | `int` | Attention head dimension |
| `attention_type` | `str` | Attention variant (`mha`, `mqa`, `gqa`, etc.) |
| `tokenizer_type` | `str` | Tokenizer type (e.g., `tiktoken`, `sentencepiece`) |
| `max_gpus_scale` | `int` | Maximum GPUs the model can scale to |
| `frameworks_supported` | `List[str]` | Supported frameworks (`torchrun`, `accelerate`, `deepspeed`, etc.) |
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

### `TrainArgsConfig` and Training Hyperparameter Combinations

Individual training defaults are defined in the `training` block of `base_training.yaml`:

```yaml
training:
  batch_size: 1
  grad_accum: 1
  precision: "bf16"
  steps: 50
  lr: 2e-5
  optimizer: "adamw"
  gradient_checkpointing: False
  enable_compile: True
  epochs: null
```

Training hyperparameter combinations that the generator expands are defined in the `combinations` block of `base_training.yaml` (or inherited by specific model configs):

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

The generator computes the Cartesian product of these lists: `3 × 1 × 2 × 1 × 1 × 1 × 1 × 2 = 24` individual configs.

**Field Descriptions:**

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

**Mutual exclusivity**: `steps` and `epochs` are mutually exclusive — you cannot specify both in the same config. If both are provided, the config is skipped.

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

Machine-specific configs are organized in subdirectories named after the HPC machine's `name_pattern` (defined in `base.yaml` under `machine.name_pattern`). For example, all MareNostrum5 (MN5) machine-specific overrides are in the `MN5/` subdirectory. When adding a new HPC machine, create a new subdirectory with the same name as the `machine.name_pattern` value.

```
configs/
├── base.yaml                    # Root config — defines defaults, machine, experiment settings
├── MN5.yaml                     # MN5-specific root config (extends base, adds arch + slurm)
├── MN5-singularity.yaml         # MN5 with Singularity container config (extends MN5)
├── MN5-uv-venv.yaml             # MN5 with uv venv Python environment (extends MN5)
├── arch/
│   └── MN5.yaml                 # MareNostrum5 GPU specs (H100-SXM, 64GB VRAM, TFLOPs)
├── dataset/
│   ├── base.yaml                # Base dataset template
│   ├── alpaca.yaml              # Alpaca instruction-tuning dataset (portable across machines)
│   ├── squadv2.yaml             # SQuAD v2 question-answering dataset (portable)
│   └── MN5/
│       ├── alpaca.yaml          # MN5-specific path override for Alpaca
│       └── squadv2.yaml         # MN5-specific path override for SQuAD v2
├── framework/
│   ├── base.yaml                # Base framework template (empty)
│   ├── accelerate.yaml          # HuggingFace Accelerate (portable, defines parallelism + scripts)
│   ├── deepspeed.yaml           # Microsoft DeepSpeed (portable)
│   ├── deepspeed-accelerate.yaml  # DeepSpeed with Accelerate integration (portable)
│   ├── torchrun.yaml            # PyTorch native (portable)
│   └── MN5/
│       ├── accelerate.yaml      # MN5-specific overrides (Singularity container path, etc.)
│       ├── deepspeed.yaml
│       ├── deepspeed-accelerate.yaml
│       └── torchrun.yaml
├── model/
│   ├── base_training.yaml       # Base model template + training hyperparameter combinations
│   ├── llama3_8b.yaml           # LLaMA 3 8B (portable, dense architecture)
│   ├── llama3_70b.yaml          # LLaMA 3 70B (portable, dense)
│   ├── mistral_7b.yaml          # Mistral 7B (portable, dense)
│   ├── gemma3_1b.yaml           # Gemma 3 1B (portable, dense)
│   ├── gemma3_12b.yaml          # Gemma 3 12B (portable, dense)
│   └── MN5/
│       ├── llama3_8b.yaml       # MN5-specific path override (e.g., GPFS scratch location)
│       ├── llama3_70b.yaml
│       ├── mistral_7b.yaml
│       ├── gemma3_1b.yaml
│       └── gemma3_12b.yaml
├── slurm/
│   ├── base.yaml                # Base SLURM template (empty — values filled by overrides)
│   └── MN5.yaml                 # MareNostrum5 SLURM settings (account, QoS, partition, 4 GPUs/node)
```

**Naming Convention**: Portable (machine-agnostic) configs are stored at the top level of their group. Machine-specific overrides are stored in `{machine.name_pattern}/` subdirectories.

### Config Composition

Hydra resolves configs through a **defaults list** in `base.yaml` and machine-specific root configs (e.g., `MN5.yaml`). The defaults list defines the order in which configs are merged.

**In `base.yaml`:**

```yaml
defaults:
  - model: base_training       # Load base model template
  - framework: base            # Load base framework template
  - dataset: base              # Load base dataset template
  - _self_                     # Merge base.yaml's own fields last
```

**In machine-specific config (e.g., `MN5.yaml`):**

```yaml
defaults:
  - arch: MN5                  # Load MN5 GPU architecture specs
  - slurm: MN5                 # Load MN5 SLURM settings
  - base                       # Load base.yaml and its defaults
  - _self_                     # Merge MN5.yaml's own fields last
```

**At runtime**, `hydra_app.py` composes configs using Hydra's `compose()` function with machine-specific overrides:

```python
cfg = compose(
    config_name,  # e.g., "MN5"
    overrides=[
        f"model={machine.name_pattern}/{model}-{machine.name_pattern}",  # e.g., model=MN5/llama3_8b-MN5
        f"framework={machine.name_pattern}/{framework}-{machine.name_pattern}",  # e.g., framework=MN5/accelerate-MN5
        f"dataset={machine.name_pattern}/{dataset}-{machine.name_pattern}",  # e.g., dataset=MN5/alpaca-MN5
    ]
)
```

This pattern allows:
- **Portable base configs** (e.g., `llama3_8b.yaml`, `accelerate.yaml`) to be shared across HPC machines
- **Machine-specific overrides** (e.g., `MN5/llama3_8b.yaml`) to customize paths, Singularity containers, and environment settings per machine
- **Easy extensibility**: Adding a new HPC machine (e.g., `LUMI`) simply requires creating a new `LUMI/` subdirectory with overrides and a corresponding `LUMI.yaml` root config

**Variable interpolation** is supported throughout the config system using Hydra's interpolation syntax `${group.field}`. This allows configs to reference values from other groups without duplication:

```yaml
# In framework/accelerate.yaml
run: scripts/accelerate-common/run-${framework.parallelism_name}.sh
finetune: scripts/accelerate-common/finetune-${framework.parallelism_name}.py

# In slurm/MN5.yaml
gres: gpu:${arch.node.gpus_per_node}

# In framework configs
scripts:
  run: scripts/deepspeed-common/run-deepspeed.sh
  finetune: scripts/deepspeed-common/finetune-deepspeed-pure.py
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

1. **Create the portable model YAML file** at `configs/model/<name>.yaml` with all required fields from `ModelConfig`. This config should work across all HPC machines. Example for a new model:

   ```yaml
   # configs/model/phi3_3b.yaml
   defaults:
     - base_training
     - _self_
   
   name: phi3_3b
   path: /path/to/phi3-3b  # Generic/default path
   total_params_billions: 3.8
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

2. **Register the model** in `dataclasses_hydra/__init__.py` by adding it to the `MODELS` list in `hydra_app.py`:

   ```python
   MODELS = ["llama3_8b", "gemma3_1b", "gemma3_12b", "mistral_7b", "llama3_70b", "phi3_3b"]
   ```

3. **(Per HPC machine) Create machine-specific overrides** at `configs/model/{machine.name_pattern}/<name>-{machine.name_pattern}.yaml` to override the model path and other machine-specific settings:

   ```yaml
   # configs/model/MN5/phi3_3b-MN5.yaml
   path: /gpfs/scratch/bsc99/phi3-3b-mn5  # MN5-specific GPFS path
   ```

   This file is composed when the generator uses `model=MN5/phi3_3b-MN5`, allowing each HPC machine to have its own model paths while sharing the same portable config structure.

### Adding a New Dataset

1. **Create the portable dataset YAML file** at `configs/dataset/<name>.yaml` with portable defaults:

   ```yaml
   # configs/dataset/fineweb.yaml
   name: fineweb
   path: /path/to/fineweb/train.jsonl  # Generic/default path
   task: Pretraining
   train: train
   max_seq_len: 4096
   ```

2. **Register the dataset** in `hydra_app.py` by adding it to the `DATASETS` list:

   ```python
   DATASETS = ["alpaca", "squadv2", "fineweb"]
   ```

3. **(Per HPC machine) Create machine-specific overrides** at `configs/dataset/{machine.name_pattern}/<name>-{machine.name_pattern}.yaml` to override paths for that machine:

   ```yaml
   # configs/dataset/MN5/fineweb-MN5.yaml
   path: /gpfs/scratch/bsc99/fineweb/train.jsonl  # MN5-specific path
   ```

4. **(Optional) Create a dataset handler** in `scripts/shared/datasets/` if the dataset requires custom preprocessing. Register the handler in `DATASET_HANDLER_MAP`.

### Adding a New Framework

1. **Create the portable framework YAML file** at `configs/framework/<name>.yaml` with base parallelism specs and scripts. This should be machine-agnostic:

   ```yaml
   # configs/framework/vllm.yaml
   defaults:
     - base
     - _self_
   
   name: vllm
   python_environment:
   singularity_container: # Leave empty if overriding per machine
   
   parallelism_name: ""
   parallelism:
     none: { min_gpus: 1 }
     tensor_parallel: { min_gpus: 2 }
   
   scripts:
     run: scripts/vllm-common/run.sh
     finetune: scripts/vllm-common/serve.py
     shared: scripts/shared
     copy_files:
       - scripts/vllm-common/gpu_monitor.py
       - scripts/vllm-common/utils.py
   ```

2. **Register the framework** in `hydra_app.py` by adding it to the `FRAMEWORKS` list:

   ```python
   FRAMEWORKS = ["accelerate", "torchrun", "deepspeed", "vllm"]
   ```

3. **(Per HPC machine) Create machine-specific overrides** at `configs/framework/{machine.name_pattern}/<name>-{machine.name_pattern}.yaml` to set Singularity containers or other machine-specific settings:

   ```yaml
   # configs/framework/MN5/vllm-MN5.yaml
   singularity_container: /gpfs/scratch/bsc99/vllm-latest.sif
   ```

4. **Create the corresponding script directories** under `scripts/<name>-common/` with launcher scripts, training entry points, and utilities.

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
   - Get the machine name pattern from the initial config (e.g., `_init_cfg.machine.name_pattern = "MN5"`)
   - Compose a base config via Hydra with machine-specific overrides:
     ```python
     cfg = compose(
         config_name,  # e.g., "MN5"
         overrides=[
             f"model={name_pattern}/{model}-{name_pattern}",
             f"framework={name_pattern}/{framework}-{name_pattern}",
             f"dataset={name_pattern}/{dataset}-{name_pattern}",
         ]
     )
     ```
   - For each parallelism strategy in `cfg.model.parallelism_supported`:
     - Create a deep copy of the config
     - Set `cfg.framework.parallelism_name` to the current strategy
     - Set `cfg.framework.parallelism` to only contain the selected strategy's specs
     - Run all constraint rules against the combo
     - If all rules pass, add to valid list; otherwise, record the skip reason

2. **Determine GPU node counts**: For each valid parallelism combo:
   - Calculate minimum nodes required based on memory constraints
   - Generate candidate node counts up to the model's `max_gpus_scale`
   - For each valid node count:

3. **Expand training hyperparameters**: For each valid node configuration:
   - Compute the Cartesian product of all training hyperparameter lists from `cfg.model.combinations`
   - For each combination, create a new `BenchmarkConfig` with specific hyperparameters
   - Handle single-GPU case: if `machine.single_gpu_also_valid=True` and min parallelism is 1 GPU, also generate 1-GPU configs
   - Each expanded config gets a unique `id` based on machine, model, framework, parallelism, dataset, nodes, and hyperparameters

4. **Validate and save**: For each config:
   - Run constraint rules (VRAM, framework support, parallelism GPU requirements)
   - Save valid configs as YAML files to `outpath`
   - Log skipped configs with reasons

The function returns `(valid_configs, skipped_configs)` tuples.

#### `expand_arch_gpu_configs(cfg)`

Replicates the GPU configuration logic: if `single_gpu_also_valid` is True and the architecture supports it, returns both `[1, gpus_per_node]`; otherwise returns `[gpus_per_node]`.

This function is called during the combo generation to determine which GPU counts to generate configs for. For example, on MN5 with 4 GPUs per node and `single_gpu_also_valid=true`, it returns `[1, 4]`, meaning configs for both 1-GPU and 4-GPU (1-node) runs are generated.

### Usage

```python
from configs_hydra.hydra_app import generate_valid_combos

# Generate configs for MN5
valid, skipped = generate_valid_combos(
    config_path="./configs_hydra/configs",
    config_name="MN5",  # Use MN5 machine config
    outpath="benchmark-runs/"
)

print(f"Valid configs: {len(valid)}")
print(f"Skipped configs: {len(skipped)}")
for cfg, reasons in skipped:
    print(f"  Skipped {cfg.id}:")
    for result in reasons:
        print(f"    - {result.rule_name}: {result.reason}")

# To use a different machine, simply change config_name:
valid, skipped = generate_valid_combos(
    config_path="./configs_hydra/configs",
    config_name="LUMI",  # Use LUMI machine config instead
    outpath="benchmark-runs/"
)
```

### Typical Output

When run on MN5 with the default MODELS, FRAMEWORKS, and DATASETS lists, `generate_valid_combos()` typically produces:

- **~50-100 valid base configs** (after constraint validation)
- **~500-2000 expanded configs** (after training hyperparameter expansion)
- **~20-50 skipped configs** with reasons like "GPU count exceeds max for parallelism" or "VRAM estimate exceeds available"

The exact numbers depend on the models, frameworks, and datasets configured in `hydra_app.py`.

---

## Adding a New HPC Machine

The configuration system is designed to support multiple HPC machines through the `machine.name_pattern` mechanism. Each HPC machine has its own subdirectories in `configs/` for machine-specific overrides.

### Step-by-Step Guide

1. **Create the machine root config** at `configs/<MACHINE>.yaml` (e.g., `configs/LUMI.yaml`):

   ```yaml
   # configs/LUMI.yaml
   defaults:
     - arch: LUMI           # Create this in configs/arch/LUMI.yaml
     - slurm: LUMI          # Create this in configs/slurm/LUMI.yaml
     - base
     - _self_
   
   machine:
     name: csc-lumi         # Actual supercomputer name
     name_pattern: LUMI     # Pattern used for subdirectory names
     modules: "PrgEnv-cray craype-x86-rome rocm/6.1.0"
     runtime_env_mode: singularity
     single_gpu_also_valid: True
     env:
       NCCL_DEBUG: INFO
       PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
   ```

   The `name_pattern` must match the subdirectory names you create in the next steps.

2. **Create machine-specific architecture config** at `configs/arch/LUMI.yaml`:

   ```yaml
   # configs/arch/LUMI.yaml
   name: LUMI
   gpus:
     MI300X:
       vram_gb: 192
       peak_flops:
         fp32: 1456.0e12
         fp16: 1456.0e12
         bf16: 1456.0e12
   node:
     gpus_per_node: 8
   comm:
     intra_node: NVLink  # or your machine's interconnect
     inter_node: IB
   ```

3. **Create machine-specific SLURM config** at `configs/slurm/LUMI.yaml`:

   ```yaml
   # configs/slurm/LUMI.yaml
   account: project_abc123
   qos: standard
   partition: gpu
   sbatch:
     nodes: 1
     gpus_per_node: 8
     time: "01:00:00"
     gres: gpu:mi300x:8
   ```

4. **Create machine-specific override subdirectories** and populate them:

   - `configs/model/LUMI/` — Create one override file per model:
     ```yaml
     # configs/model/LUMI/llama3_8b-LUMI.yaml
     path: /scratch/llama3-8b  # LUMI-specific model path
     ```

   - `configs/framework/LUMI/` — Create one override file per framework:
     ```yaml
     # configs/framework/LUMI/accelerate-LUMI.yaml
     singularity_container: /scratch/containers/accelerate-latest.sif
     ```

   - `configs/dataset/LUMI/` — Create one override file per dataset:
     ```yaml
     # configs/dataset/LUMI/alpaca-LUMI.yaml
     path: /scratch/datasets/alpaca/train.jsonl
     ```

   **Naming convention**: `<config_name>-<machine.name_pattern>.yaml`

5. **Update `hydra_app.py`** to use the new machine when running:

   ```python
   # In hydra_app.py, when calling generate_valid_combos:
   valid, skipped = generate_valid_combos(
       config_path="./configs_hydra/configs",
       config_name="LUMI",  # Switch to LUMI config
       outpath="benchmark-runs/"
   )
   ```

### How It Works

When the generator composes configs, it uses the machine root config name (e.g., `"LUMI"`) and the machine-specific override pattern:

```python
cfg = compose(
    "LUMI",  # Load LUMI.yaml and its defaults (arch/LUMI, slurm/LUMI, base, etc.)
    overrides=[
        "model=LUMI/llama3_8b-LUMI",     # Load base llama3_8b.yaml, then override with LUMI/llama3_8b-LUMI.yaml
        "framework=LUMI/accelerate-LUMI",
        "dataset=LUMI/alpaca-LUMI",
    ]
)
```

This two-layer approach (portable base + machine-specific overrides) ensures:
- **Code reuse**: Base configs are written once and reused across machines
- **Maintainability**: Machine-specific paths and settings are isolated
- **Scalability**: Adding a new machine requires only creating one new subdirectory per config group, not duplicating entire configs

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
