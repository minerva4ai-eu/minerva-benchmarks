# 🧠 MINERVA Training & Fine-Tuning Benchmarks

LLM training and fine-tuning benchmarks for HPC supercomputers, part of the [MINERVA](https://minerva-project.eu/) project. Designed to evaluate large language model training performance on BSC's MareNostrum 5 and other HPC systems.

These benchmarks measure:
- **Total Time to Train / Fine-tune** — End-to-end wall-clock time
- **Training throughput** — Tokens per second, TFLOPs per GPU
- **Memory consumption** — GPU VRAM usage per parallelism strategy
- **GPU utilization** — Power draw, memory bandwidth, compute utilization
- **Scaling behavior** — Single-GPU → DDP → FSDP → ZeRO comparison

## 📚 Documentation

| Topic | README |
|-------|--------|
| **Configuration System** (Hydra schemas, YAML configs, constraint validation) | [configs_hydra/README.md](configs_hydra/README.md) |
| **Training Scripts** (launchers, entry points, shared code, utilities) | [scripts/README.md](scripts/README.md) |
| **SLURM Job Submission** (CLI, job lifecycle, monitoring) | [scripts/slurm/README.md](scripts/slurm/README.md) |
| **Environment Management** (Singularity containers, uv, CUDA versions) | [envs/README.md](envs/README.md) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  User invokes: ./minerva-cli.sh run --config-name base-MN5      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  configs_hydra/hydra_app.py                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 1. Compose BenchmarkConfig from YAML + overrides        │    │
│  │    (model, framework, dataset, parallelism)             │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ 2. Validate with constraint rules                       │    │
│  │    (GPU floor/ceiling, framework support, memory)       │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ 3. Expand training combos (batch_size × precision × LR) │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  scripts/slurm/submitter.py                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 4. Build launch folder: benchmark-runs/{machine}/{date}/ │    │
│  │                        {model}/{framework}/.../           │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ 5. Copy scripts (framework + shared) to launch folder   │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ 6. Build env dict (MODEL, DATASET, PARALLELISM, etc.)   │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Singularity Container (envs/uv/cuda128-flash-attn/)            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 7. sbatch → SLURM schedules job on compute nodes        │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ 8. Launcher script runs inside container:               │    │
│  │    accelerate/torchrun/deepspeed launch → training      │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ 9. GPU monitoring (gpu_plots.py) runs in background     │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Results                                                        │
│  ├── benchmark-runs/{machine}/{date}/.../launch-{R}/output/    │
│  │   ├── checkpoints/                                          │    │
│  │   ├── metrics.json                                          │    │
│  │   └── profiler/{job_id}/gpu_plots.png                       │    │
│  └── generateSummaryTable.py → full_benchmark_summary_*.csv   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Supported Frameworks & Parallelism

| Framework | Single-GPU | DDP | FSDP | ZeRO-1 | ZeRO-2 | ZeRO-3 | ZeRO-3-Offload |
|-----------|:----------:|:---:|:----:|:------:|:------:|:------:|:--------------:|
| **HuggingFace Accelerate** | ✅ | ✅ | ✅ | — | — | — | — |
| **PyTorch TorchRun** | ✅ | ✅ | ✅ | — | — | — | — |
| **Microsoft DeepSpeed** | — | — | — | ✅ | ✅ | ✅ | ✅ |

---

## 📁 Project Structure

```
training_MN5/
├── configs_hydra/                    # Hydra-based configuration system
│   ├── README.md                     # ← See configs_hydra/README.md
│   ├── hydra_app.py                  # Orchestrator: composition + validation
│   ├── dataclasses_hydra/            # Python dataclass schemas
│   ├── constraints/                  # Rule-based validation
│   └── configs/                      # YAML config files
│       ├── base.yaml / base-MN5.yaml # Root configs
│       ├── model/                    # Model definitions
│       ├── framework/                # Framework specs
│       ├── dataset/                  # Dataset definitions
│       ├── slurm/                    # SLURM batch config
│       ├── trainings/                # Training hyperparameter combos
│       └── arch/                     # HPC architecture specs
│
├── scripts/                          # Training scripts & utilities
│   ├── README.md                     # ← See scripts/README.md
│   ├── shared/                       # Framework-agnostic code
│   │   ├── custom_train.py           # CustomTrainer + MegatronFlopsCallback
│   │   ├── data.py                   # Dataset loading & preprocessing
│   │   └── utils.py                  # print_rank, count_parameters, timed
│   ├── accelerate-common/            # HuggingFace Accelerate
│   ├── torchrun-common/              # PyTorch native distributed
│   ├── deepspeed-common/             # Microsoft DeepSpeed
│   ├── slurm/                        # SLURM CLI (see scripts/slurm/README.md)
│   ├── activate-env-variables-per-supercomputer.sh
│   ├── utils.sh                      # Shell utilities
│   └── gpu_plots.py                  # GPU utilization plotting
│
├── envs/                             # Environment management
│   ├── README.md                     # ← See envs/README.md
│   ├── uv/                           # uv + Singularity containers
│   │   ├── cuda124-flash-attn/       # CUDA 12.4
│   │   └── cuda128-flash-attn/       # CUDA 12.8
│   └── cli/                          # CLI environment
│
├── minerva-cli.sh                    # CLI entry point (wraps Singularity)
├── generateSummaryTable.py           # Aggregate benchmark results
├── generateSummaryTablev2.py         # Alternative results aggregator
└── README.md                         # ← This file
```

---

## 🚀 Quick Start

### 1. Build the Singularity Container

```bash
cd envs/uv/cuda128-flash-attn/
singularity build singularity_uv-runtime.sif singularity_uv-runtime.def
```

See [envs/README.md](envs/README.md) for full environment management details.

### 2. Configure for Your Machine

Edit `configs_hydra/configs/base.yaml` (or create a machine-specific variant):

```yaml
machine:
  name: bsc-mn5-acc
  modules: singularity jq cuda/12.1 miniforge
  singularity_container: /path/to/singularity_uv-runtime.sif
  singularity_binds:
    - "--bind /gpfs/scratch"
  singularity_args:
    - "--nv"
```

Ensure `scripts/activate-env-variables-per-supercomputer.sh` has correct NCCL/CUDA settings for your cluster.

### 3. Run Benchmarks

```bash
# Dry run — generate configs without submitting
./minerva-cli.sh run --dry-run --config-name MN5

# Submit all valid jobs
./minerva-cli.sh run --config-name MN5

# Monitor running jobs
./minerva-cli.sh monitor
```

### 4. Generate Summary

```bash
python generateSummaryTable.py
```

---

## ⚙️ Configuration

The benchmark system uses [Hydra](https://hydra.cc/) for config composition. Key configuration files:

| File | Purpose |
|------|---------|
| `configs_hydra/configs/base.yaml` | Base config (container path, SLURM settings) |
| `configs_hydra/configs/MN5.yaml` | Machine-specific config (container path, SLURM settings) |
| `configs_hydra/configs/model/*.yaml` | Model definitions (architecture dims, GPU requirements) |
| `configs_hydra/configs/framework/*.yaml` | Framework specs (parallelism, script paths) |
| `configs_hydra/configs/dataset/*.yaml` | Dataset definitions (path, task type, seq length) |
| `configs_hydra/configs/trainings/combinations.yaml` | Training hyperparameters (batch size, precision, LR, etc.) |
| `configs_hydra/configs/slurm/MN5.yaml` | SLURM batch directives (account, QoS, partition, GPUs) |

See [configs_hydra/README.md](configs_hydra/README.md) for full configuration reference.

### Modifying the Generator

Edit the `MODELS`, `FRAMEWORKS`, `DATASETS` lists in `configs_hydra/hydra_app.py` to control which combinations are generated:

```python
MODELS = ["llama3_8b", "mistral_7b"]
FRAMEWORKS = ["accelerate", "deepspeed"]
DATASETS = ["alpaca", "squadv2"]
```

---

## 📊 Results

### Output Structure

```
benchmark-runs/
└── {machine}/                    # e.g., bsc-mn5-acc
    └── {date}/                   # e.g., 10-06-2026
        └── {model}/{framework}/{dataset}/
            └── nodes-{N}/
                └── run_id-{N}/
                    └── launch-{R}/
                        └── output/
                            ├── checkpoints/
                            ├── metrics.json
                            └── profiler/
                                └── gpu_plots.png
```

### Aggregation

```bash
python generateSummaryTable.py    # Standard aggregation
python generateSummaryTablev2.py  # Alternative aggregation
```

Produces CSV files like `full_benchmark_summary_{machine}.csv`.

---

## 🖥️ Supported Machines

| Machine | Partition | GPU | NCCL Settings |
|---------|-----------|-----|---------------|
| **MareNostrum 5 (BSC)** | `acc` | AMD MI300A | IB, `ib0-ib3`, `mlx5_0-mlx5_5` |
| **Leonardo (CINECA)** | — | NVIDIA | `COMPILER=nvhpc`, CUDA 12.1 |

Machine-specific settings are defined in `scripts/activate-env-variables-per-supercomputer.sh`.

---

## ✅ Requirements

- **HPC Cluster** with SLURM job scheduler
- **GPU** compatible with CUDA 12.x or ROCm (AMD MI300A)
- **Singularity/Apptainer** for container execution
- **Python 3.11** (provided inside containers)
- **InfiniBand** networking (for multi-node NCCL communication)

---

## 📖 Detailed Documentation

| Topic | Where to Look |
|-------|---------------|
| Config schemas & YAML files | [configs_hydra/README.md](configs_hydra/README.md) |
| Constraint validation rules | [configs_hydra/README.md](configs_hydra/README.md#2-constraints-) |
| Adding new models/datasets/frameworks | [configs_hydra/README.md](configs_hydra/README.md#adding-a-new-model) |
| Training scripts & launchers | [scripts/README.md](scripts/README.md) |
| SLURM CLI & job submission | [scripts/slurm/README.md](scripts/slurm/README.md) |
| Singularity containers & uv | [envs/README.md](envs/README.md) |
| GPU monitoring & plotting | [scripts/README.md](scripts/README.md#gpu_plotspy---gpu-utilization-plotting) |
| Shared training code | [scripts/README.md](scripts/README.md#shared-code-shared) |

---

## 📄 License

See [LICENSE](LICENSE) for project licensing information.
* Accelerate, Torchrun, Deepspeed, Transformers

---

## ⚠️ Notes & Limitations
* FSDP behavior depends heavily on model architecture and shard configuration.
* Performance may vary significantly across GPU architectures.
* Ensure models are downloaded and placed correctly to their paths.





## 📁 Folder Overview
### benchmarks/
Contains the benchmark scripts, configurations, and inputs used to measure model performance or throughput across different setups.
* How are we running the benchmarks: https://github.com/vllm-project/vllm/tree/main/benchmarks

### configs/
Holds JSON configuration files for mapping and organizing datasets, model types, and other runtime behavior:

* **config.json:** Main configuration file for controlling benchmarking logic.
* **config_datasets_paths_map.json:** Maps dataset names to their file paths.
* **model_type_map.json:** Maps model identifiers to their types or categories.
* **model_type_directories_map.json:** Maps model types to their corresponding directory paths.

### envs-yaml/
Contains YAML environment specifications for different tools or runtime contexts:

* **vllm-0.9.1-env.yaml:** Environment spec for using vLLM 0.9.1.
* **deepspeed-MII-env.yaml:** Environment for running DeepSpeed with Microsoft’s MII.

### results/
Stores the outputs from benchmark runs, such as logs, metrics, summaries, or result tables. This is typically auto-generated.

### scripts/
* **utils.sh:** Common bash functions or helper routines used across scripts.
* **activate-env-per-supercomputer.sh:** Bash script for activating each environment (conda/miniforge/python/etc) depending on each machine.
* **activate-env-variables-per-supercomputer.sh:** Bash script for initializing needed variables for running the benchmarks in each machine.

#### deepspeed/:
Scripts for serving and benchmarking models using DeepSpeed with Microsoft’s MII (Model Inference Interface):

* **deepspeed-mii_configurable_benchmarking_serve.sh:** Shell script to launch benchmarking for DeepSpeed MII with configurable model and dataset parameters.
* **serve_deepspeed_mii.py:** Python script to start a model serving instance using DeepSpeed-MII, handling model loading, inference, and server endpoints.
* **gpu_summary_monitor-cuda.py** Python script to monitor NVIDIA GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
* **gpu_summary_monitor-rocm.py** Python script to monitor AMD GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.

#### sglang/:
Scripts focused on benchmarking and serving with SGLang:

* **serve.sh:** Starts a local or remote SGLang model server. It also sends requests to the inference server.
* **sglang_configurable_benchmarking_serve.sh:** Script for running configurable SGLang benchmarks through the serve endpoint.
* **gpu_summary_monitor-cuda.py** Python script to monitor NVIDIA GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
* **gpu_summary_monitor-rocm.py** Python script to monitor AMD GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
* **wrapper_singularity.sh** Bash script for initializing some variables inside the Singularity container.

#### vllm/:
Scripts focused on benchmarking and serving with vLLM:

* **run_cluster.sh:** Launches the ray cluster fr multinode setups.
* **serve.sh:** Starts a local or remote vLLM model server. It also sends requests to the inference server.
* **vllm_configurable_benchmarking_serve.sh:** Script for running configurable vLLM benchmarks through the serve endpoint.
* **gpu_summary_monitor-cuda.py** Python script to monitor NVIDIA GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
* **gpu_summary_monitor-rocm.py** Python script to monitor AMD GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.


---

## 📄 Key Files
### .env-$MACHINE
Environment variable definitions for configuring paths and environment variables for each Machine.

### generateSummaryTable.py
A Python script that likely compiles benchmark results into a summary table for reporting or comparison.

### run_all_benchmarks.sh
A shell script to execute all benchmarks in a batch. Likely the main entry point for running tests.

### README.md
Documentation file (this one) explaining the structure, usage, and purpose of the repository.

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).  
You are free to use, modify, and distribute this code, provided that any derivative works are also released under the same license. Commercial and non-commercial use is allowed under the GPL-3.0 terms.

---

### 💬 Suggestions and Feedback

If you have any suggestions or would like to contribute improvements to this repository, please contact us at **minerva_support@bsc.es**.


---

### 📚 References
[1] **accelerate:** @Misc{accelerate,
  title =        {Accelerate: Training and inference at scale made simple, efficient and adaptable.},
  author =       {Sylvain Gugger and Lysandre Debut and Thomas Wolf and Philipp Schmid and Zachary Mueller and Sourab Mangrulkar and Marc Sun and Benjamin Bossan},
  howpublished = {\url{https://github.com/huggingface/accelerate}},
  year =         {2022}
}

[accelerate GitHub](https://github.com/huggingface/accelerate) and [accelerate HuggingFace](https://huggingface.co/docs/accelerate/index)

[2] **DeepSpeed:** [DeepSpeed GitHub](https://github.com/deepspeedai/DeepSpeed) and [deepspeed.ai](https://www.deepspeed.ai/)

[3] **PyTorch:** [PyTorch GitHub](https://github.com/pytorch/pytorch)

[4] **Transformers:** [Transformers HuggingFace](https://huggingface.co/docs/transformers/index)

[5] **Torchrun:** [Torchrun GitHub](https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py) and [Torchrun Docs](https://docs.pytorch.org/docs/stable/elastic/run.html)

[6] **LLama Models:** 
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.
- [HuggingFace Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [HuggingFace Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)

[7] **Gemma Models:** Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., ... & Iqbal, S. (2025). Gemma 3 technical report. arXiv preprint arXiv:2503.19786. [HuggingFace Gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) and [HuggingFace Gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)

[8] **Mistral Models:** [HuggingFace Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)


---