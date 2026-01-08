
## 📁 Project Structure

```text
.
├── benchmarks/                       # Benchmark definitions, inputs, and templates
├── configs/                          # JSON config files for models and datasets
<<<<<<< HEAD
=======
├── containers/                       # Singularity images (.sif files)
>>>>>>> c9f2946 (Initial commit from GitLab)
├── envs-yaml/                        # Conda environments for each backend
├── results/                          # Auto-generated benchmark results and logs
├── scripts/                          # Automation scripts for all supported backends
│   ├── utils.sh                      # Common shell functions
<<<<<<< HEAD
=======
│   ├── activate-env-per-supercomputer.sh           # How to Activate python/conda/miniforge environment for each cluster
│   ├── activate-env-variables-per-supercomputer.sh # Variables needed for running the benchmarks in each cluster
>>>>>>> c9f2946 (Initial commit from GitLab)
│   ├── deepspeed/                    # DeepSpeed-MII benchmark scripts
│   │   ├── deepspeed-mii_configurable_benchmarking_serve.sh
│   │   ├── gpu_summary_monitor.py
│   │   └── serve_deepspeed_mii.py
<<<<<<< HEAD
=======
│   ├── sglang/                       # SGLang benchmarking scripts
│   │   ├── sglang_configurable_benchmarking_serve
│   │   ├── gpu_summary_monitor.py
|       ├── wrapper_singularity.sh
│   │   └── serve.sh
>>>>>>> c9f2946 (Initial commit from GitLab)
│   └── vllm/                         # vLLM benchmarking scripts
│       ├── run_cluster.sh
│       ├── serve.sh
│       ├── gpu_summary_monitor.py
│       └── vllm_configurable_benchmarking_serve.sh
<<<<<<< HEAD
├── .env                              # Environment variable overrides
=======
├── .env-$MACHINE                     # Environment variable overrides
>>>>>>> c9f2946 (Initial commit from GitLab)
├── generateSummaryTable.py           # Compiles benchmark results into a summary table
├── generateScores.py                 # Adds Specific Metric Scores into the Summary table
├── run_1_benchmark.sh                # Run a single benchmark (for testing)
├── run_all_benchmarks.sh             # Run all benchmarks
└── README.md                         # Project overview and documentation
```


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
<<<<<<< HEAD
=======
* **activate-env-per-supercomputer.sh:** Bash script for activating each environment (conda/miniforge/python/etc) depending on each machine.
* **activate-env-variables-per-supercomputer.sh:** Bash script for initializing needed variables for running the benchmarks in each machine.
>>>>>>> c9f2946 (Initial commit from GitLab)

#### deepspeed/:
Scripts for serving and benchmarking models using DeepSpeed with Microsoft’s MII (Model Inference Interface):

* **deepspeed-mii_configurable_benchmarking_serve.sh:** Shell script to launch benchmarking for DeepSpeed MII with configurable model and dataset parameters.
* **serve_deepspeed_mii.py:** Python script to start a model serving instance using DeepSpeed-MII, handling model loading, inference, and server endpoints.
<<<<<<< HEAD
* **gpu_summary_monitor.py** Python script to monitor NVIDIA GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
=======
* **gpu_summary_monitor-cuda.py** Python script to monitor NVIDIA GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
* **gpu_summary_monitor-rocm.py** Python script to monitor AMD GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.

#### sglang/:
Scripts focused on benchmarking and serving with SGLang:

* **serve.sh:** Starts a local or remote SGLang model server. It also sends requests to the inference server.
* **sglang_configurable_benchmarking_serve.sh:** Script for running configurable SGLang benchmarks through the serve endpoint.
* **gpu_summary_monitor-cuda.py** Python script to monitor NVIDIA GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
* **gpu_summary_monitor-rocm.py** Python script to monitor AMD GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
* **wrapper_singularity.sh** Bash script for initializing some variables inside the Singularity container.
>>>>>>> c9f2946 (Initial commit from GitLab)

#### vllm/:
Scripts focused on benchmarking and serving with vLLM:

* **run_cluster.sh:** Launches the ray cluster fr multinode setups.
* **serve.sh:** Starts a local or remote vLLM model server. It also sends requests to the inference server.
* **vllm_configurable_benchmarking_serve.sh:** Script for running configurable vLLM benchmarks through the serve endpoint.
<<<<<<< HEAD
* **gpu_summary_monitor.py** Python script to monitor NVIDIA GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
=======
* **gpu_summary_monitor-cuda.py** Python script to monitor NVIDIA GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
* **gpu_summary_monitor-rocm.py** Python script to monitor AMD GPU memory and power usage in real-time, computing average and peak values per GPU and saving a JSON summary.
>>>>>>> c9f2946 (Initial commit from GitLab)


---

## ⚙️ Setup Instructions

### 1. 🔧 Clone the repository

```bash
# Clone git repository
git clone https://gitlab.bsc.es/minerva/benchmarks.git

# Copy inference data from MN5 to your Machine Name.
cp -R ./benchmarks/inference-MN5/ ./benchmarks-inference/inference-LEONARDO/

# Move inside your machine folder.
cd ./benchmarks-inference/inference-LEONARDO/

# You can push code and results to the git repository inside your Machine folder.
# git push
```

### 2. 🐍 Create Conda Environments
You will need 2 Conda environments for different backends. You can optionally install all to a specific path.

| Replace /your/envs/path/ with your desired directory for isolated envs (e.g., ~/model-benchmark-envs/).

```bash
# Set a path to store all environments
ENV_DIR="/your/envs/path/"

# Create vLLM environment # Adapt that command if conda is not supported in your supercomputer
# module load miniforge
conda env create --prefix ${ENV_DIR}/vllm-0.9.1-env -f envs-yaml/vllm-0.9.1-env.yaml

# Create DeepSpeed-MII environment # Adapt that command if conda is not supported in your supercomputer
# module load miniforge
conda env create --prefix ${ENV_DIR}/deepspeed-MII-env -f envs-yaml/deepspeed-MII-env.yaml

```

<<<<<<< HEAD
You will need to update your `.env` file.
=======
You will need to update your `.env-$MACHINE` file.
>>>>>>> c9f2946 (Initial commit from GitLab)

You may activate an env with:
```bash
# Export environment variables
<<<<<<< HEAD
source .env
=======
source .env-$MACHINE
>>>>>>> c9f2946 (Initial commit from GitLab)

conda activate ${ENVIRONMENT_VLLM}         # or the corresponding one
   # or
source activate ${ENVIRONMENT_DEEPSPEED}   # MN5
<<<<<<< HEAD
```
=======
# Check the file `activate-env-per-supercomputer.sh`
# and change how you are activating the environment in your cluster depending on the $MACHINE variable.
```
* **IMPORTANT:** Check the file `activate-env-per-supercomputer.sh` and change how you are activating the environment in your cluster depending on the `MACHINE` variable.

#### 2.1. Singularity Images

If you prefer, you can switch to singularity images instead of conda environments. We provide one example using SGLang Framework.

```bash
singularity pull ./containers/sglang-dev-0.5.6.post1.sif docker://lmsysorg/sglang:v0.5.6.post1
# Version of sglang used:
# sglang                    0.5.6.post1     /sgl-workspace/sglang/python
```

**Note:** Specific versions are compared across clusters. So, check that your `SGLang version` matches with this one. Same applies for other frameworks.

Use scripts and instead of activating the `ENVIRONMENT_VLLM`, use singularity commands. Example:
```bash
# '.env-$MACHINE' file
# Define 'SINGULARITY_MODULE'
# Define 'SGLANG_IMAGE'
module load $MODULE_SINGULARITY
singularity exec --nv $SGLANG_IMAGE python3 -m sglang.launch_server  --model-path /gpfs/scratch/models_registry/gemma-3-12b-it" \
   --context-length "4096" \
   --port "8000" \
   --grammar-backend "xgrammar" \
   --dist-init-addr "$NCCL_INIT_ADDR" \
   --model-impl transformers \
   --nnodes 1 \
   --node-rank "0"

```

* **IMPORTANT:** Before running the SGLang framework with a Singularity image, you need to check the `scripts/activate-env-per-supercomputer.sh` and `scripts/activate-env-variables-per-supercomputer.sh`. And change accordingly your SINGULARITY variables to set bindings, extra arguments, NCCL variables, etc.

>>>>>>> c9f2946 (Initial commit from GitLab)

## ▶️ Running Benchmarks
### 1. Adjust Configurations
Before running, check or modify:

* **configs/config.json:** Global settings (not update).
* **configs/model_type_map.json:** Model identifier-to-type mapping (not update). 
   ⚠️ WARNING: Make sure you have the models downloaded in each corresponding model type. Example contents:
      - **Text Generation:** "Llama-3.1-8B-Instruct", "Llama-3.1-405B", "gemma-3-12b-it", "Mistral-7B-Instruct-v0.3".
      - **Vision:** None
* **configs/config_datasets_paths_map.json:** Dataset path configuration.
* **configs/model_type_directories_map.json:** Model type directories mapping.
<<<<<<< HEAD
=======
* **scripts/activate-env-per-supercomputer.sh:** Define how to activate your environment in your cluster.
* **scripts/activate-env-variables-per-supercomputer.sh:** Define your needed variables for running the benchmarks in your cluster.

>>>>>>> c9f2946 (Initial commit from GitLab)

#### 1.1. 📥 Downloading Models

Before running benchmarks, ensure all models listed in your `configs/model_type_map.json` are downloaded and available in the correct directories, as specified in `configs/model_type_directories_map.json`.

Each model should be placed in a folder that matches its expected type. For example:

- **Text Generation Models** (e.g. LLaMA, Gemma, Mistral) → `vllm` or `deepspeed` directories.
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

<<<<<<< HEAD
⚠️ **WARNING:** Ensure `.env` is set properly if using any custom environment variables. You will need to change PATHS, MODULES, ACCOUNT, etc... Inside that `.env` specific file.

⚠️ **WARNING:** Ensure changing the ENVIRONMENT VARIABLES Section inside the `run_all_benchmarks.sh` script. First of all, try the `run_1_benchmark.sh` in your supercomputer. Once it's running OK, proceed with the `run_all_benchmarks.sh`.

⚠️ **WARNING:** Check how we are activating environments in MN5 (e.g. source activate $ENVIRONMENT_VLLM) inside the scripts of **scripts/vllm/** or **scripts/deepspeed** and change it accordingly to your cluster.
=======
⚠️ **WARNING:** Ensure `.env-$MACHINE` is set properly if using any custom environment variables. You will need to change PATHS, MODULES, ACCOUNT, etc... Inside that `.env-$MACHINE` specific file.

⚠️ **WARNING:** Ensure changing the ENVIRONMENT VARIABLES Section inside the `run_all_benchmarks.sh` script. First of all, try the `run_1_benchmark.sh` in your supercomputer. Once it's running OK, proceed with the `run_all_benchmarks.sh`.

⚠️ **WARNING:** Check how we are activating environments in MN5 (e.g. source activate $ENVIRONMENT_VLLM) inside the scripts of **scripts/vllm/** , **scripts/sglang/** or **scripts/deepspeed** and change it accordingly to your cluster.
>>>>>>> c9f2946 (Initial commit from GitLab)

The `run_1_benchmark.sh` script will:

* Set up the required environment.
* Launch `vllm` benchmark and `Llama-3.1-8B-Instruct`.
* Save results to the `results/` folder.

```bash
# Run a simple benchmark for testing
bash run_1_benchmark.sh
```


### 3. 🏁 Run All Benchmarks

Once the Simple Benchmark is running OK. We can try to run all benchmarks following those instructions:

<<<<<<< HEAD
⚠️ **WARNING:** Ensure `.env` is set properly if using any custom environment variables. You will need to change PATHS, MODULES, ACCOUNT, etc... Inside that `.env` specific file.

⚠️ **WARNING:** Ensure changing the ENVIRONMENT VARIABLES Section inside the `run_all_benchmarks.sh` script. First of all, try the `run_1_benchmark.sh` in your supercomputer. Once it's running OK, proceed with the `run_all_benchmarks.sh`.

⚠️ **WARNING:** Check how we are activating environments in MN5 (e.g. source activate $ENVIRONMENT_VLLM) inside the scripts of **scripts/vllm/** or **scripts/deepspeed** and change it accordingly to your cluster.
=======
⚠️ **WARNING:** Ensure `.env-$MACHINE` is set properly if using any custom environment variables. You will need to change PATHS, MODULES, ACCOUNT, etc... Inside that `.env-$MACHINE` specific file.

⚠️ **WARNING:** Ensure `scripts/activate-env-per-supercomputer.sh` and `scripts/activate-env-variables-per-supercomputer.sh` are set properly if using any custom environment variables or conda/python/miniforge environments. 

⚠️ **WARNING:** Ensure changing the ENVIRONMENT VARIABLES Section inside the `run_all_benchmarks.sh` script. First of all, try the `run_1_benchmark.sh` in your supercomputer. Once it's running OK, proceed with the `run_all_benchmarks.sh`.

⚠️ **WARNING:** Check how we are activating environments in MN5 (e.g. source activate $ENVIRONMENT_VLLM) inside the scripts of **scripts/vllm/**, **scripts/sglang/**or **scripts/deepspeed** and change it accordingly to your cluster.
>>>>>>> c9f2946 (Initial commit from GitLab)


The `run_all_benchmarks.sh` script will:

* Set up the required environment.
* Launch benchmarks for all supported backends.
* Save results to the `results/` folder.

```bash
# Run the complete benchmark suite
bash run_all_benchmarks.sh
```

This script internally calls:
* scripts/vllm/vllm_configurable_benchmarking_serve.sh
<<<<<<< HEAD
* scripts/deepspeed/deepspeed-mii_configurable_benchmarking_serve.sh


=======
* scripts/sglang/sglang_configurable_benchmarking_serve
* scripts/deepspeed/deepspeed-mii_configurable_benchmarking_serve.sh


⚠️ **WARNING:** DeepSpeed-MII is not working in AMD GPUs.

>>>>>>> c9f2946 (Initial commit from GitLab)

## 📊 Result Summary
After the benchmarks are done, generate a summary table:

```bash
# Export environment variables
<<<<<<< HEAD
source .env
=======
source .env-$MACHINE
>>>>>>> c9f2946 (Initial commit from GitLab)

# Activate environment.
# IMPORTANT! Change it accordingly in your cluster.
module load miniforge
source activate ${ENVIRONMENT_VLLM}

# Run the generation of the Table.
python generateSummaryTable.py

# Run the generation of the Metrics scores.
python generateScores.py
```

This will output a consolidated report from the files in `results/`.
In MN5:
   * `full_benchmark_summary_MareNostrum5_ACC.csv`
   * `full_benchmark_summary_MareNostrum5_ACC_score.csv`

IMPORTANT! Once you have run all benchmarks, make sure you `push` all your changes in your machine folder inside git repository.

---

## ✅ Requirements
* Miniconda/Anaconda
* GPU and drivers compatible with DeepSpeed/vLLM
* Python 3.10+ recommended
<<<<<<< HEAD
* vLLM, DeepSpeed-MII
=======
* vLLM, DeepSpeed-MII, SGLang
>>>>>>> c9f2946 (Initial commit from GitLab)


---

## 📄 Key Files
<<<<<<< HEAD
### .env
Environment variable definitions for configuring paths and environment variables.
=======
### .env-$MACHINE
Environment variable definitions for configuring paths and environment variables for each Machine.
>>>>>>> c9f2946 (Initial commit from GitLab)

### generateSummaryTable.py
A Python script that likely compiles benchmark results into a summary table for reporting or comparison.

### generateScores.py
A Python script that adds into a summary table the scores for each type of metrics for comparison.

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
[1] **vLLM:** Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I. (2023, October). Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles (pp. 611-626).

[vLLM GitHub](https://github.com/vllm-project/vllm) and  [vLLM Benchmarks GitHub](https://github.com/vllm-project/vllm/tree/main/benchmarks)

[2] **DeepSpeed-MII:** [DeepSpeed-MII GitHub](https://github.com/deepspeedai/DeepSpeed-MII)

[3] **LLama Models:** 
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.
- [HuggingFace Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [HuggingFace Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)

[4] **Gemma Models:** Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., ... & Iqbal, S. (2025). Gemma 3 technical report. arXiv preprint arXiv:2503.19786. [HuggingFace Gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)

[5] **Mistral Models:** [HuggingFace Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

<<<<<<< HEAD
=======
[6] **SGLang:** Zheng, L., Yin, L., Xie, Z., Sun, C. L., Huang, J., Yu, C. H., ... & Sheng, Y. (2024). Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37, 62557-62583.

[sglang GitHub](https://github.com/sgl-project/sglang) and [sglang Documentation](https://docs.sglang.io/index.html)

>>>>>>> c9f2946 (Initial commit from GitLab)

---