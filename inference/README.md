# 🧠 Inference Benchmarks

Reproducible **LLM serving benchmarks** for EuroHPC supercomputers, part of the MINERVA project.
The suite launches an inference server (vLLM or SGLang) through Slurm, stresses it with a fixed set of concurrency levels, records latency / throughput / GPU memory / power, and aggregates everything into comparable CSV tables and scores.

| | |
|---|---|
| **Frameworks** | vLLM · SGLang |
| **Datasets** | ShareGPT · Sonnet |
| **Models** | Llama-3.1-8B-Instruct · Llama-3.3-70B-Instruct · Llama-3.1-405B-Instruct · gemma-3-12b-it · Mistral-7B-Instruct-v0.3 · ALIA-40b-instruct-2605 |
| **Parallelism** | Single GPU → full node (TP) → multi-node (TP × PP) |
| **Hardware** | NVIDIA (CUDA) and AMD (ROCm) GPUs |
| **Metrics** | TTFT, ITL, TPOT, output throughput, request throughput, GPU memory (avg/peak), GPU power (avg/peak) |

---

## 📑 Table of Contents

1. [Supported systems](#️-supported-systems)
2. [Project structure](#-project-structure)
3. [Quick start](#-quick-start)
4. [Setup](#️-setup)
5. [Running benchmarks](#️-running-benchmarks)
6. [Monitoring and resubmitting jobs](#-monitoring-and-resubmitting-jobs)
7. [Results layout](#-results-layout)
8. [Result summary and scores](#-result-summary-and-scores)
9. [Adding a new supercomputer](#-adding-a-new-supercomputer)
10. [Framework notes and known limitations](#-framework-notes-and-known-limitations)
11. [Key files](#-key-files)
12. [License, support and references](#-license)

---

## 🖥️ Supported systems

Each system is described by **one `.env-<machine>` file** and **one `configs-<machine>/` folder**. The `<machine>` identifier is what you put in the `MACHINE` variable of the run scripts.

| `MACHINE` | System / partition | GPUs per node | `MACHINE_TYPE` | Runtime used |
|---|---|---|---|---|
| `bsc-mn5-acc` | MareNostrum 5 – ACC (BSC) | 4 × NVIDIA (64 GB) | `cuda` | Singularity |
| `leonardo` | Leonardo – `boost_usr_prod` (CINECA) | 4 × NVIDIA (64 GB) | `cuda` | Singularity |
| `idris-jeanzay-h100` | Jean Zay – `gpu_p6` H100 (IDRIS) | 4 × NVIDIA H100 (80 GB) | `cuda` | Singularity |
| `cines-adastra-mi250` | Adastra – MI250 (CINES) | 8 × AMD MI250 (64 GB) | `rocm` | Singularity |
| `cines-adastra-mi300` | Adastra – MI300 (CINES) | 4 × AMD MI300 (128 GB) | `rocm` | Singularity |

Values come from the `.env-<machine>` files (`GPUS_PER_NODE`, `VRAM_PER_GPU`). Result CSVs for these systems are stored in [`../analytics/data`](../analytics/data).

---

## 📁 Project structure

```text
inference/
├── benchmarks/                         # Benchmark client code (adapted from vLLM's benchmarks)
│   ├── benchmark_serving.py            #   ← the client used by all run scripts
│   ├── backend_request_func.py         #   request functions per backend (vllm, sglang, ...)
│   ├── benchmark_latency.py            #   other vLLM benchmarks (not used by the run scripts)
│   ├── benchmark_throughput.py
│   ├── benchmark_prefix_caching.py
│   ├── benchmark_prioritization.py
│   ├── benchmark_guided.py
│   ├── benchmark_serving_guided.py
│   └── structured_schemas/
├── configs-<machine>/                  # One folder per supercomputer (bsc-mn5-acc, leonardo, ...)
│   ├── config.json                     #   environment variables recorded for a run
│   ├── config_datasets_paths_map.json  #   dataset name  → file path
│   ├── model_type_map.json             #   model name    → model type
│   └── model_type_directories_map.json #   model type    → directory containing the models
├── envs-sing-imgs/                     # Singularity/Apptainer definition files (.def)
│   ├── vllm-0.21.0-cuda-compat.def     #   vLLM 0.21.0, CUDA (with CUDA-13 compat libs)
│   ├── vllm-0.21.0-rocm.def            #   vLLM 0.21.0, ROCm
│   ├── sglang-0.5.6.post1.def          #   SGLang 0.5.6.post1, CUDA
│   └── sglang-0.5.6.post1-rocm7.def    #   SGLang 0.5.6.post1, ROCm 7
├── envs-yaml/                          # Conda environment specs
│   ├── vllm-0.9.1-env.yaml
│   └── sglang-0.5.6.post1.yaml
├── scripts/
│   ├── utils.sh                        # JSON lookup helpers (model type, model dir, dataset path)
│   ├── activate-env-per-supercomputer.sh            # How to activate Python/Conda per machine
│   ├── activate-env-variables-per-supercomputer.sh  # NCCL, Singularity bindings, caches per machine
│   ├── vllm/
│   │   ├── run_mp_vllm.sh              # Slurm job: multi-node vLLM server + benchmark loop
│   │   └── gpu_summary_monitor-{cuda,rocm}.py
│   └── sglang/
│       ├── sglang_configurable_benchmarking_serve.sh  # Slurm job: launches serve.sh on every node
│       ├── serve.sh                    # Starts SGLang server, then runs the benchmark loop
│       ├── wrapper_singularity.sh      # Passes variables into the container
│       └── gpu_summary_monitor-{cuda,rocm}.py
├── .env-<machine>                      # Per-machine paths, modules, account, partition, hardware
├── run_1_benchmark.sh                  # Smoke test (one configuration)
├── run_all_benchmarks.sh               # Full benchmark matrix
├── benchmark_status.sh                 # Read-only status table of the whole matrix
├── run_failed_benchmaks.sh             # Resubmit only the launches that failed
├── generateSummaryTable.py             # results/ → consolidated CSV
├── generateScores.py                   # consolidated CSV → CSV with latency/throughput/energy scores
├── requirements.txt                    # Dependencies of the two post-processing scripts
└── results/                            # Auto-generated (git-ignored)
```

> The GPU monitors are provided per vendor: `gpu_summary_monitor-cuda.py` (NVML) and `gpu_summary_monitor-rocm.py` (AMD SMI). The run scripts pick the right one from `MACHINE_TYPE`.

---

## 🚀 Quick start

All scripts must be launched **from this `inference/` directory** (they use relative paths).

```bash
git clone https://github.com/minerva4ai-eu/minerva-benchmarks.git
cd minerva-benchmarks/inference

# 1. Edit the machine file: paths, modules, account, partition, images (see "Setup")
$EDITOR .env-<machine>
$EDITOR configs-<machine>/*.json

# 2. Smoke test: one framework, one model, one dataset
#    (edit MACHINE / MACHINE_TYPE at the top of the script first)
bash run_1_benchmark.sh

# 3. Full benchmark matrix
bash run_all_benchmarks.sh

# 4. Follow progress, resubmit failures
bash benchmark_status.sh
DRY_RUN=true bash run_failed_benchmaks.sh   # preview
bash run_failed_benchmaks.sh                # resubmit

# 5. Aggregate
pip install -r requirements.txt
python generateSummaryTable.py
python generateScores.py
```

---

## ⚙️ Setup

### 1. 🔧 Clone the repository

```bash
git clone https://github.com/minerva4ai-eu/minerva-benchmarks.git
cd minerva-benchmarks/inference
```

### 2. 🧩 Configure your machine

Two things describe a machine: the `.env-<machine>` file and the `configs-<machine>/` folder.

#### `.env-<machine>`

The run scripts load it with `source .env-$MACHINE` (all variables are exported).

| Variable | Meaning |
|---|---|
| `VLLM_IMAGE`, `SGLANG_IMAGE` | Absolute path to the `.sif` images used for vLLM / SGLang |
| `METRICS_IMAGE` | Image used by SGLang to run the benchmark client and GPU monitor (must contain the benchmark dependencies, including `datasets`) |
| `ENVIRONMENT_VLLM`, `ENVIRONMENT_SGLANG` | Path to a Conda/venv environment, activated on the host by `activate-env-per-supercomputer.sh`. Both the vLLM and the SGLang job scripts activate `ENVIRONMENT_VLLM`, so set it to a valid environment (the servers themselves run inside the containers) |
| `BENCHMARK_FILE` | Absolute path to `benchmarks/benchmark_serving.py` |
| `PORT` | Port of the inference server (default `8000`) |
| `MODULES`, `SINGULARITY_MODULE` | Modules loaded before activating environments / running containers |
| `SUPCOMPUTER_NAME`, `PARTITION_NAME` | Used to name the result CSVs (`full_benchmark_summary_<name>_<partition>.csv`) |
| `ACCOUNT`, `QOS`, `CONSTRAINT`, `TIME_LIMIT` | Slurm options (`CONSTRAINT` is used on Jean Zay) |
| `GPUS_PER_NODE`, `CPUS_PER_GPU`, `VRAM_PER_GPU` | Hardware description; drives `--gres`, `--cpus-per-task` and the size of the benchmark matrix |
| `MACHINE`, `MACHINE_TYPE` | Machine identifier (must match the `.env-` / `configs-` suffix) and `cuda` or `rocm` |
| `NET_IFACE` *(optional)* | Network interface used to compute the vLLM host IP (falls back to the first address of `hostname -I`) |
| `IB_IFACE` *(optional)* | Interface used for `NCCL_SOCKET_IFNAME` on Jean Zay |

> ⚠️ Paths in the committed `.env-*` files point to the maintainers' project directories. **Replace every path, module, account and QOS with your own.**

#### `configs-<machine>/`

| File | Purpose |
|---|---|
| `config_datasets_paths_map.json` | Maps `sharegpt` / `sonnet` to the dataset files on your file system |
| `model_type_map.json` | Maps each model name to a type (e.g. `Text Generation`) |
| `model_type_directories_map.json` | Maps each model type to the directory that contains the model folders |
| `config.json` | List of environment variables recorded for a run (normally left untouched) |

A model is resolved as `<directory of its type>/<model name>`, so **the folder name must match the model name exactly**:

```json
// model_type_map.json
{ "Llama-3.1-8B-Instruct": "Text Generation" }

// model_type_directories_map.json
{ "Text Generation": "/path/to/models" }

// → /path/to/models/Llama-3.1-8B-Instruct
```

> ⚠️ The lookups in `scripts/utils.sh` are `grep`-based. Keep these JSON files **flat**, with one `"key": "value"` pair per entry and quoted strings.

### 3. 📦 Runtime environments

#### Singularity / Apptainer images (vLLM and SGLang)

Build the images from the provided definition files (on a machine where you have the needed privileges, or with `--fakeroot`), then copy the `.sif` files to the cluster and set `VLLM_IMAGE` / `SGLANG_IMAGE`. `*.sif` files are git-ignored.

```bash
singularity build vllm-0.21.0-rocm.sif envs-sing-imgs/vllm-0.21.0-rocm.def

# Or pull directly from Docker Hub
singularity pull sglang-0.5.6.post1.sif docker://lmsysorg/sglang:v0.5.6.post1
```

| Definition file | Base image | Typical use |
|---|---|---|
| `vllm-0.21.0-cuda-compat.def` | `vllm/vllm-openai:v0.21.0` + CUDA 13 compat libs | NVIDIA systems with older drivers (e.g. Leonardo) |
| `vllm-0.21.0-rocm.def` | `vllm/vllm-openai-rocm:v0.21.0` | Adastra (AMD) |
| `sglang-0.5.6.post1.def` | `lmsysorg/sglang:v0.5.6.post1-runtime` | NVIDIA systems |
| `sglang-0.5.6.post1-rocm7.def` | `lmsysorg/sglang:v0.5.6.post1-rocm700-mi30x` | Adastra (AMD) |

**Notes**

* The image that runs `benchmark_serving.py` needs the Python package `datasets` (imported unconditionally). The ROCm and vLLM definitions install it; if you use an image without it, add `pip install datasets` to its `%post`.
* Framework versions are **compared across clusters**, so keep them identical to the ones above (vLLM `0.21.0`, SGLang `0.5.6.post1`).
* Bind mounts, extra Singularity flags (`--nv` / `--rocm`), NCCL variables and cache folders are defined **per machine** in `scripts/activate-env-variables-per-supercomputer.sh`. Review that file before your first run.

#### Conda environments

The job scripts activate a Conda environment on the host (`ENVIRONMENT_VLLM`) before launching the containers. The specs below are provided for that purpose, and also allow running vLLM or SGLang without containers.

```bash
ENV_DIR="/your/envs/path"      # e.g. ~/model-benchmark-envs

# module load miniforge        # adapt if Conda is not available on your system
conda env create --prefix ${ENV_DIR}/vllm-0.9.1-env -f envs-yaml/vllm-0.9.1-env.yaml
conda env create --prefix ${ENV_DIR}/sglang-env     -f envs-yaml/sglang-0.5.6.post1.yaml    # optional
```

Then set `ENVIRONMENT_VLLM` (and `ENVIRONMENT_SGLANG`, if used) in `.env-<machine>`.

> `envs-yaml/sglang-0.5.6.post1.yaml` contains a hard-coded `name:` path under the BSC file system. Use `--prefix` as above, or edit the `name:` line.

**How environments get activated.** `scripts/activate-env-per-supercomputer.sh <env>` contains one `case "$MACHINE"` branch per cluster (module loads, `conda activate` vs `source .../bin/activate`, NCCL variables). If your cluster activates environments differently, change it there, not in the framework scripts.

### 4. 📥 Download models and datasets

**Models.** Download every model you plan to benchmark into the directory configured in `model_type_directories_map.json`:

```bash
pip install huggingface-hub
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --repo-type model \
    --local-dir /path/to/models/Llama-3.1-8B-Instruct \
    --token hf_TOKEN
```

> Llama models are gated: you must accept Meta's license on Hugging Face first. ALIA models are published by [BSC-LT](https://huggingface.co/BSC-LT).

> ⚠️ A mismatch between a model name, its entry in `model_type_map.json`, and its folder name makes the run script stop with *"Unknown model type … or missing directory mapping"*.

**Datasets.** The two supported datasets are ShareGPT (`ShareGPT_V3_unfiltered_cleaned_split.json`) and Sonnet (`sonnet.txt`). Download them to a shared location and point `config_datasets_paths_map.json` at them. Compute nodes usually have no internet access, so download from a login node.

---

## ▶️ Running benchmarks

### How the benchmark matrix is built

Both run scripts loop over the following variables (defined at the top of each script):

| Variable | Meaning |
|---|---|
| `FRAMEWORKS` | `vllm`, `sglang` |
| `DATASETS` | `sharegpt`, `sonnet` |
| `MODELS` | Model names (must exist in `model_type_map.json`) |
| `NUMBER_OF_NODES` | e.g. `(1 2 4)` |
| `MAX_MODEL_LENGTHS` | Context lengths, e.g. `(4096 16384 32768)` |
| `REPEATS` | Independent launches per configuration (results are averaged later) |
| `MACHINE`, `MACHINE_TYPE` | Which `.env-<machine>` / `configs-<machine>/` to use, and `cuda` / `rocm` |

For every combination the script submits **one Slurm job** (`sbatch`). The parallel layout is derived automatically:

| Nodes | GPUs used per node | Tensor parallel (TP) | Pipeline parallel (PP) |
|---|---|---|---|
| 1 | `1` **and** `GPUS_PER_NODE` (two runs) | GPUs per node used | 1 |
| > 1 | `GPUS_PER_NODE` | `GPUS_PER_NODE` | number of nodes |

Inside each job, the server is started once and then benchmarked at **five concurrency levels (150, 250, 300, 500, 1000)** with **1000 prompts** each. A GPU monitor samples memory and power (every 0.1 s) during every level.

**Automatic exclusions** (the combination is skipped):

* Llama-3.1-405B(-Instruct) on vLLM or SGLang with fewer than 4 nodes.
* gemma-3-12b-it on SGLang with more than 1 node.

### 1. 🏁 Run a simple benchmark first

> ⚠️ **Always run the smoke test on a new machine before the full matrix.**
> The committed defaults of every script point to the maintainers' current machine. Check `MACHINE` and `MACHINE_TYPE` at the top of the script.

`run_1_benchmark.sh` runs one configuration (by default vLLM, ShareGPT, `Llama-3.1-8B-Instruct`, one `MAX_MODEL_LENGTH`) and writes to `results/`.

```bash
bash run_1_benchmark.sh
```

Checklist if it fails:

* `.env-<machine>` paths, modules, `ACCOUNT`, `QOS`, `PARTITION_NAME` are yours.
* Model, dataset and image paths exist and are readable from compute nodes.
* `activate-env-per-supercomputer.sh` and `activate-env-variables-per-supercomputer.sh` have a branch for your machine.
* Singularity bindings in `activate-env-variables-per-supercomputer.sh` cover your file system (models, datasets, project directory).
* Slurm output of the job: `results/<framework>/<dataset>/<model>/<run folder>/launch-1/run-<jobid>.{out,err}`.

### 2. 🏁 Run all benchmarks

Once the smoke test succeeds:

```bash
bash run_all_benchmarks.sh
```

The default matrix is 2 frameworks × 2 datasets × 6 models × nodes `(1 2 4)` × context lengths `(4096 16384 32768)` × 3 repeats, plus a 1-GPU variant for every single-node run, before exclusions. That is a **large** number of GPU jobs, so reduce the arrays at the top of the script if your allocation is limited.

Which script each framework uses:

| Framework | Slurm job script | Server | Isolation |
|---|---|---|---|
| vLLM | `scripts/vllm/run_mp_vllm.sh` | `vllm serve` (multi-node via vLLM's multiprocessing launcher: `--nnodes`, `--node-rank`, `--headless` on workers) | Singularity |
| SGLang | `scripts/sglang/sglang_configurable_benchmarking_serve.sh` → `serve.sh` | `python -m sglang.launch_server` on every node | Singularity |

For reproducibility, each launch folder receives a **copy of the exact scripts** that ran in it.

---

## 🔭 Monitoring and resubmitting jobs

### `benchmark_status.sh` - status table (read-only)

Classifies every launch of the matrix by combining result files with `squeue` / `sacct`, and prints one row per framework × dataset × model:

| Status | Meaning |
|---|---|
| `completed` | All five `Concurrency_*.json` files exist |
| `running` / `pending` | A job for this launch folder is in the queue |
| `failed` | Latest real attempt ended in `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, ... |
| `cancelled` | Latest real attempt was cancelled after it started |
| `no_history` | Folder exists, but no job ever ran (or it finished without result files) |
| `not_attempted` | No launch folder |

Jobs cancelled before they ever started are ignored. Set `MACHINE`, the matrix arrays, and optionally `SACCT_WINDOW` (default `now-60days`) at the top of the script.

```bash
bash benchmark_status.sh
```

### `run_failed_benchmaks.sh` — resubmit only what failed

Resubmits a launch **only if** its folder exists, it is incomplete, nothing is pending/running for it, and its latest real job ended in a failure state. If `squeue` or `sacct` is unreachable the script aborts instead of guessing.

```bash
DRY_RUN=true bash run_failed_benchmaks.sh   # show what would be resubmitted
bash run_failed_benchmaks.sh                # resubmit
```

Switches at the top of the script: `ONLY_EXISTING_LAUNCHES` (default `true`), `RETRY_CANCELLED` (default `false`), `SACCT_WINDOW`.

> Keep the `FRAMEWORKS`, `DATASETS`, `MODELS`, `NUMBER_OF_NODES`, `MAX_MODEL_LENGTHS`, `REPEATS` and `MACHINE` arrays identical in `run_all_benchmarks.sh`, `benchmark_status.sh` and `run_failed_benchmaks.sh`; otherwise they will look at different folders.

---

## 🗂️ Results layout

```text
results/
└── <framework>/<dataset>/<model>/
    └── Nodes_<N>-GPUs_<total>-TP_<tp>-PP_<pp>-MaxModelLength_<len>/
        └── launch-<k>/
            ├── Concurrency_{150,250,300,500,1000}.json   # benchmark_serving.py output
            ├── gpu_summary_{150,...,1000}.txt            # GPU memory/power summary (JSON content)
            ├── logs_benchmark_<conc>.log                 # client logs
            ├── run-<jobid>.out / run-<jobid>.err         # Slurm output
            └── copies of the scripts used for this launch
```

`GPUs_<total>` is the **total** number of GPUs (nodes × GPUs per node used). A launch is considered complete when all five `Concurrency_*.json` files exist.

---

## 📊 Result summary and scores

After the benchmarks finish, aggregate the results (a lightweight environment is enough: `pip install -r requirements.txt` installs `pandas` and `python-dotenv`).

```bash
python generateSummaryTable.py   # results/ → results/full_benchmark_summary_<SUPCOMPUTER_NAME>_<PARTITION_NAME>.csv
python generateScores.py         # adds scores → results/full_benchmark_summary_<SUPCOMPUTER_NAME>_<PARTITION_NAME>_score.csv
```

Example on MareNostrum 5: `full_benchmark_summary_MareNostrum5_ACC.csv` and `full_benchmark_summary_MareNostrum5_ACC_score.csv`.

> **Which `.env` is used?** Both scripts pick the env file from the host name: `leonardo` → `.env-leonardo`, `jean-zay` → `.env-idris-jeanzay-h100`, and `BSC_MACHINE=mn5` → `.env-bsc-mn5-acc` (so `export BSC_MACHINE=mn5` on MareNostrum 5). On any other system (e.g. Adastra) add your branch to the `if/elif` block at the top of both scripts.

### What the summary contains

One row per *(framework, dataset, model, nodes, TP, PP, max model length, concurrency)*, with metrics **averaged over the repeats** (`launch-1…launch-N`):

| Column(s) | Source |
|---|---|
| TTFT, ITL, TPOT (ms) | Mean values from `Concurrency_<c>.json` |
| Output throughput (tokens/s), Request throughput (requests/s) | `Concurrency_<c>.json` |
| GPU Memory Usage Avg / Peak (GB) | `gpu_summary_<c>.txt` (average across GPUs / maximum over GPUs) |
| Power Usage Avg / Peak (W) | `gpu_summary_<c>.txt` (average across GPUs / maximum over GPUs) |

### Scores added by `generateScores.py`

| Score | Definition | Better |
|---|---|---|
| **Latency Score** | `1 / (TTFT + ITL + TPOT)`; the scaled version is multiplied by 10⁴ | higher |
| **Throughput Score** | Harmonic mean of output throughput and request throughput | higher |
| **Energy Score (Tokens/Watt)** | Output throughput ÷ average GPU power | higher |
| **Global Score** | `0.34 · Latency Score Scaled + 0.34 · Throughput Score + 0.32 · Energy Score` | higher |

Rows are sorted by Global Score. Launches in which the server returned no successful requests produce zero metrics and therefore `inf` scores, so filter them out before ranking.

### Visualising results

Copy the `*_score.csv` files into [`../analytics/data`](../analytics/data) and run the Plotly dashboard from the `analytics` folder:

```bash
cd ../analytics
python dashboard_app_plotly.py
```

Once everything is done, **commit and push** your machine's files (`.env-<machine>`, `configs-<machine>/`, and the summary CSVs you want to share).

---

## ➕ Adding a new supercomputer

1. **Create `.env-<machine>`** by copying the closest existing one, and adjust paths, modules, account, partition, QOS, `GPUS_PER_NODE`, `CPUS_PER_GPU`, `VRAM_PER_GPU`, `MACHINE`, `MACHINE_TYPE`.
2. **Create `configs-<machine>/`** by copying an existing folder and updating the model and dataset paths.
3. **Build or copy the container images** (`.def` files in `envs-sing-imgs/`) and set `VLLM_IMAGE`, `SGLANG_IMAGE`, `METRICS_IMAGE`.
4. **Add a `case` branch** for `<machine>` in:
   * `scripts/activate-env-per-supercomputer.sh` (modules / environment activation)
   * `scripts/activate-env-variables-per-supercomputer.sh` (NCCL / RCCL variables, Singularity bindings and extra flags, cache folders)
5. **Slurm flags** (only if the default is not enough). `run_1_benchmark.sh` and `run_all_benchmarks.sh` already handle `*jeanzay*` (`--constraint`, `-q`, `--partition`), `*adastra*` (`--constraint=<partition>`) and a default `-q $QOS --partition=$PARTITION_NAME`. Edit the `SITE_SPECIFIC_ARGS` block in those scripts for any other case.
6. **Host-name detection** for the summary scripts (see [Result summary](#-result-summary-and-scores)).
7. Run `run_1_benchmark.sh`, then `run_all_benchmarks.sh`.

---

## 📄 Key files

| File | Purpose |
|---|---|
| `.env-<machine>` | Paths, modules, Slurm account/partition and hardware description of one machine |
| `run_1_benchmark.sh` | Smoke test: one configuration |
| `run_all_benchmarks.sh` | Entry point for the full benchmark matrix |
| `benchmark_status.sh` | Status table for the matrix (read-only) |
| `run_failed_benchmaks.sh` | Resubmit failed launches (supports `DRY_RUN=true`) |
| `generateSummaryTable.py` | Builds the consolidated CSV from `results/` |
| `generateScores.py` | Adds latency, throughput, energy and global scores |
| `requirements.txt` | Dependencies of the post-processing scripts (`pandas`, `python-dotenv`) |
| `scripts/utils.sh` | JSON lookup helpers used by the run scripts |
| `README.md` | This document |

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).
You are free to use, modify, and distribute this code, provided that derivative works are released under the same license.

---

## 💬 Suggestions and feedback

For questions, suggestions or contributions, contact **[minerva_support@bsc.es](mailto:minerva_support@bsc.es)**.

---

## 📚 References

[1] **vLLM:** Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention. In *Proceedings of the 29th Symposium on Operating Systems Principles* (pp. 611-626). [vLLM GitHub](https://github.com/vllm-project/vllm) · [vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks) (the client in `benchmarks/` is adapted from these)

[2] **SGLang:** Zheng, L., Yin, L., Xie, Z., Sun, C. L., Huang, J., Yu, C. H., ... & Sheng, Y. (2024). SGLang: Efficient execution of structured language model programs. *Advances in Neural Information Processing Systems, 37*, 62557-62583. [SGLang GitHub](https://github.com/sgl-project/sglang) · [SGLang documentation](https://docs.sglang.io/index.html)

[3] **Llama models:** Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). LLaMA: Open and efficient foundation language models. *arXiv:2302.13971*.
[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) · [Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)

[4] **Gemma models:** Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., ... & Iqbal, S. (2025). Gemma 3 technical report. *arXiv:2503.19786*. [gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)

[5] **Mistral models:** [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

[6] **ALIA models:** [BSC-LT on Hugging Face](https://huggingface.co/BSC-LT)
