# 🧠 Training & Fine-Tuning Benchmarks

In addition to inference benchmarks, this repository also supports training and fine-tuning benchmarks for large language models using PyTorch Distributed, FSDP, and HuggingFace Accelerate.

These benchmarks are designed to evaluate:
* Total Time to Train or Fine-tune
* Training throughput
* Memory consumption
* GPU utilization
* Scaling behavior (single-GPU, DDP, FSDP)

## 📁 Training Benchmark Project Structure

```text
.
├── configs/
│   ├── config.json                          # Global benchmark configuration
│   ├── config_datasets_handlers_map.py      # Maps datasets to Python handler classes
│   ├── config_datasets_paths_map.json       # Dataset name → path mapping
│   ├── model_parallelism_config.json        # DDP/FSDP/single-GPU configs
│   ├── model_type_directories_map.json      # Model type → filesystem path
│   └── model_type_map.json                  # Model identifier → model type
│
├── datasets/
│   ├── alpaca-cleaned/
│   │   └── alpaca_data_cleaned.json
│   ├── squad_v2/
│   │   ├── train-00000-of-00001.parquet
│   │   └── validation-00000-of-00001.parquet
│   └── handlers/
│       ├── AlpacaHandler.py                 # Dataset preprocessing logic
│       └── SquadV2Handler.py
│
├── envs-yaml/
│   └── fine-tuning-env.yaml                 # Environment for training benchmarks
│
├── scripts/
│   ├── accelerate-common/
│   │   ├── custom_train.py                  # Shared CustomTrainer class
│   │   ├── finetune-none.py                 # Single-GPU training
│   │   ├── finetune-ddp.py                  # DDP training
│   │   ├── finetune-fsdp.py                 # FSDP training
│   │   ├── run-none.sh                      # Launcher (single-GPU)
│   │   ├── run-ddp.sh                       # Launcher (DDP)
│   │   ├── run-fsdp.sh                      # Launcher (FSDP)
│   │   ├── gpu_monitor.py                   # GPU usage monitoring
│   │   └── utils.py
│   │
│   ├── torchrun-common/
│   │   ├── custom_train.py                  # Shared CustomTrainer class
│   │   ├── finetune-none.py
│   │   ├── finetune-ddp.py
│   │   ├── finetune-fsdp.py
│   │   ├── run-none.sh
│   │   ├── run-ddp.sh
│   │   ├── run-fsdp.sh
│   │   ├── gpu_monitor.py
│   │   └── utils.py
│   │
│   ├── gpu_plots.py                         # GPU utilization plotting
│   └── utils.sh
│
├── .env                                     # Environment variables
├── .env-bsc-mn5-acc                         # Environment variables BSC MN5 ACC partition
├── generateSummaryTable.py                  # Aggregates training benchmark results
├── run_1_benchmark.sh                       # 1 benhcmark test for running the training benchmarks
├── run_all_benchmarks.sh                    # Entry point for training benchmarks
└── README.md                                # Project overview

```

## 📦 Supported Training Frameworks

The training benchmarks support two frameworks:

🔹 HuggingFace Accelerate

Easier multi-GPU and multi-node setup

Supports:
* Single-GPU
* Distributed Data Parallel (DDP)
* Fully Sharded Data Parallel (FSDP)

Scripts are located in:
```bash
./scripts/accelerate-common/
```

🔹 TorchRun Distributed
* Lower-level control over DDP and FSDP
* Useful for detailed performance analysis

Scripts are located in:
```bash
./scripts/torchrun-common/
```


### 📊 Datasets & Handlers

Datasets are decoupled from training logic using dataset handlers.

#### Dataset Handlers

Defined in:
```bash
./configs/config_datasets_handlers_map.py
```
Each dataset handler:
* Loads raw data
* Applies preprocessing
* Formats samples for training

Examples:
* AlpacaHandler.py → Instruction fine-tuning
* SquadV2Handler.py → Question answering

#### Dataset Paths

Defined in:
```bash
./configs/config_datasets_paths_map.json
```

## ⚙️ Setup Instructions

### 1. 🔧 Clone the repository

```bash
# Clone git repository
git clone https://

# Copy inference data from MN5 to your Machine Name.
cp -R ./training-minerva-benchmarks/training-MN5/ ./training-minerva-benchmarks/training-LEONARDO/

# Move inside your machine folder.
cd ./training-minerva-benchmarks/training-LEONARDO/

# You can push code and results to the git repository inside your Machine folder.
# git push
```

### 2. 🐍 Create Training Environments
You will need 1 Conda/Miniforge/Python environments for all frameworks. You can optionally install all to a specific path.

| Replace /your/envs/path/ with your desired directory for isolated envs (e.g., ~/model-benchmark-envs/).

```bash
# Set a path to store all environments
ENV_DIR="/your/envs/path/"

# Create fine-tuning environment # Adapt that command if conda is not supported in your supercomputer
# module load miniforge
conda env create --prefix ${ENV_DIR}/fine-tune-dev -f ./envs-yaml/fine-tune-dev-env.yaml

```

You will need to update your `.env-$MACHINE` file.

You may activate an env with:
```bash
# Export environment variables
source .env-$MACHINE

conda activate ${ENVIRONMENT_FINETUNING}         # or the corresponding one
   # or
source activate ${ENVIRONMENT_FINETUNING}   # MN5
# Check the file `activate-env-per-supercomputer.sh`
# and change how you are activating the environment in your cluster depending on the $MACHINE variable.
```
* **IMPORTANT:** Check the file `activate-env-per-supercomputer.sh` and change how you are activating the environment in your cluster depending on the `MACHINE` variable.





## ▶️ Running Training Benchmarks

### 1. Adjust Configurations
Before running, check or modify:

* **configs/config.json:** Global settings (not update).
* **configs/config_datasets_paths_map.json:** Path to each dataset file.
* **configs/model_parallelism_config.json:** Setups and configurations to run for each model and parallelism type (distinct `batch sizes`, `max model length`, `steps`, `epochs` or `precion type` to iterate).
* **configs/model_type_directories_map.json:** Model type directories mapping.
* **configs/model_type_map.json:** Model identifier-to-type mapping (not update). 
   ⚠️ WARNING: Make sure you have the models downloaded in each corresponding model type. Example contents:
      - **Text Generation:** "Llama-3.1-8B-Instruct", "Llama-3.1-405B", "gemma-3-12b-it", "Mistral-7B-Instruct-v0.3".
      - **Vision:** None
* **scripts/activate-env-per-supercomputer.sh:** Define how to activate your environment in your cluster.
* **scripts/activate-env-variables-per-supercomputer.sh:** Define your needed variables for running the benchmarks in your cluster.


#### 1.1. 📥 Downloading Models

Before running benchmarks, ensure all models listed in your `configs/model_type_map.json` are downloaded and available in the correct directories, as specified in `configs/model_type_directories_map.json`.

Each model should be placed in a folder that matches its expected type. For example:

- **Text Generation Models** → LLaMA, Gemma, Mistral.
- **Vision Models** → Currently not required (none defined in this setup).

> ⚠️ **WARNING:** Mismatches between model identifiers and their expected directories will cause benchmark failures.

You can download HuggingFace models manually or using the CLI:

```bash
# Example: Download a model manually into the vllm directory
# You need to create and activate an environment that has huggingface installed.
#      pip install huggingface-hub
mkdir -p /path/to/text-generation/Llama-3.1-8B-Instruct

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --repo-type model --local-dir /path/to/text-generation/Llama-3.1-8B-Instruct --token hf_TOKEN

```

**Note:** You need to accept the llama agreement to share your contact information to access llama models. The information you provide will be collected, stored, processed and shared in accordance with the Meta Privacy Policy. 



### 2. 🏁 Run a Simple Benchmark

⚠️ **WARNING:** Ensure `.env-$MACHINE` is set properly if using any custom environment variables. You will need to change PATHS, MODULES, ACCOUNT, etc... Inside that `.env-$MACHINE` specific file.

⚠️ **WARNING:** Ensure changing the ENVIRONMENT VARIABLES Section inside the `run_all_benchmarks.sh` script (change variables such as `MACHINE`, `MACHINE_TYPE` and the test that you want to run). First of all, try the `run_1_benchmark.sh` in your supercomputer. Once it's running OK, proceed with the `run_all_benchmarks.sh`.

⚠️ **WARNING:** Check how we are activating environments in MN5 (e.g. source activate $ENVIRONMENT_FINETUNING) inside the scripts of **scripts/accelerate-common/** , **scripts/torchrun-common/** or **scripts/deepspeed-common** and change it accordingly to your cluster.

The `run_1_benchmark.sh` script will:
* Set up the required environment.
* Launch `accelerate` benchmark and `Llama-3.1-8B-Instruct`.
* Save results to the `results/` folder.

```bash
# Run a simple benchmark for testing
bash run_1_benchmark.sh
```

### 3. 🏁 Run All Benchmarks

Once the Simple Benchmark is running OK for each parallelism type (`none`,`ddp`,`fsdp`), each model (`gemma-3-1b-it`,`Llama-3.1-8B-Instruct`,`Mistral-7B-Instruct-v0.3`,`Llama-3.3-70B-Instruct`) and each dataset (`alpaca` and `squadv2`). We can try to run all benchmarks following those instructions:

⚠️ **WARNING:** Ensure `.env-$MACHINE` is set properly if using any custom environment variables. You will need to change PATHS, MODULES, ACCOUNT, etc... Inside that `.env-$MACHINE` specific file.

⚠️ **WARNING:** Ensure `scripts/activate-env-per-supercomputer.sh` and `scripts/activate-env-variables-per-supercomputer.sh` are set properly if using any custom environment variables or conda/python/miniforge environments. 

⚠️ **WARNING:** Ensure changing the ENVIRONMENT VARIABLES Section inside the `run_all_benchmarks.sh` script. First of all, try the `run_1_benchmark.sh` in your supercomputer. Once it's running OK, proceed with the `run_all_benchmarks.sh`.

⚠️ **WARNING:** Check how we are activating environments in MN5 (e.g. source activate $ENVIRONMENT_VLLM) inside the scripts of **scripts/accelerate-common/**, **scripts/torchrun-common/**or **scripts/deepspeed-common** and change it accordingly to your cluster.


The `run_all_benchmarks.sh` script will:

* Set up the required environment.
* Launch benchmarks for all supported backends.
* Save results to the `results/` folder.

```bash
# Run the complete benchmark suite
bash run_all_benchmarks.sh
```

This script internally calls:
* scripts/accelerate-common/run-ddp.sh
* scripts/accelerate-common/run-fsdp.sh
* scripts/accelerate-common/run-none.sh
* scripts/torchrun-common/run-*.sh
* scripts/deepspeed-common/run-*.sh


### 📊 Training Benchmark Results Summary
After the benchmarks are done, generate a summary table:

```bash
# Export environment variables
source .env-$MACHINE

# Activate environment.
# IMPORTANT! Change it accordingly in your cluster.
module load miniforge
source activate ${ENVIRONMENT_FINETUNING}

# Run the generation of the Table.
python generateSummaryTable.py

```

This will output a consolidated report from the files in `results/`.
In MN5:
   * `full_benchmark_training_summary_MareNostrum5_ACC.csv`

⚠️ IMPORTANT! Once you have run all benchmarks, make sure you `push` all your changes in your machine folder inside git repository.

---

## ✅ Requirements
* Miniconda/Anaconda/Miniforge
* GPU and drivers compatible with Accelerate/vLLM
* Python 3.10+ recommended
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