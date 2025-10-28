
## 📁 Project Structure

```text
.
├── benchmarks/                       # Benchmark definitions, inputs, and templates
├── configs/                          # JSON config files for models and datasets
├── envs-yaml/                        # Conda environments for each backend
├── results/                          # Auto-generated benchmark results and logs
├── scripts/                          # Automation scripts for all supported backends
│   ├── utils.sh                      # Common shell functions
│   ├── deepspeed/                    # DeepSpeed-MII benchmark scripts
│   │   ├── deepspeed-mii_configurable_benchmarking_serve.sh
│   │   └── serve_deepspeed_mii.py
│   └── vllm/                         # vLLM benchmarking scripts
│       ├── run_cluster.sh
│       ├── serve.sh
│       └── vllm_configurable_benchmarking_serve.sh
├── .env                              # Environment variable overrides
├── generateSummaryTable.py           # Compiles benchmark results into a summary table
├── run_1_benchmark.sh                # Run a single benchmark (for testing)
├── run_all_benchmarks.sh             # Run all benchmarks
└── README.md                         # Project overview and documentation
```


## 📁 Folder Overview
### benchmarks/
Contains the benchmark scripts, configurations, and inputs used to measure model performance or throughput across different setups.

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

#### deepspeed/:
Scripts for serving and benchmarking models using DeepSpeed with Microsoft’s MII (Model Inference Interface):

* **deepspeed-mii_configurable_benchmarking_serve.sh:** Shell script to launch benchmarking for DeepSpeed MII with configurable model and dataset parameters.
* **serve_deepspeed_mii.py:** Python script to start a model serving instance using DeepSpeed-MII, handling model loading, inference, and server endpoints.

#### vllm/:
Scripts focused on benchmarking and serving with vLLM:

* **run_cluster.sh:** Launches the ray cluster fr multinode setups.
* **serve.sh:** Starts a local or remote vLLM model server. It also sends requests to the inference server.
* **vllm_configurable_benchmarking_serve.sh:** Script for running configurable vLLM benchmarks through the serve endpoint.



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

You will need to update your `.env` file.

You may activate an env with:
```bash
# Export environment variables
source .env

conda activate ${ENVIRONMENT_VLLM}         # or the corresponding one
   # or
source activate ${ENVIRONMENT_DEEPSPEED}   # MN5
```

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

⚠️ **WARNING:** Ensure `.env` is set properly if using any custom environment variables. You will need to change PATHS, MODULES, ACCOUNT, etc... Inside that `.env` specific file.

⚠️ **WARNING:** Ensure changing the ENVIRONMENT VARIABLES Section inside the `run_all_benchmarks.sh` script. First of all, try the `run_1_benchmark.sh` in your supercomputer. Once it's running OK, proceed with the `run_all_benchmarks.sh`.

⚠️ **WARNING:** Check how we are activating environments in MN5 (e.g. source activate $ENVIRONMENT_VLLM) inside the scripts of **scripts/vllm/** or **scripts/deepspeed** and change it accordingly to your cluster.

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

⚠️ **WARNING:** Ensure `.env` is set properly if using any custom environment variables. You will need to change PATHS, MODULES, ACCOUNT, etc... Inside that `.env` specific file.

⚠️ **WARNING:** Ensure changing the ENVIRONMENT VARIABLES Section inside the `run_all_benchmarks.sh` script. First of all, try the `run_1_benchmark.sh` in your supercomputer. Once it's running OK, proceed with the `run_all_benchmarks.sh`.

⚠️ **WARNING:** Check how we are activating environments in MN5 (e.g. source activate $ENVIRONMENT_VLLM) inside the scripts of **scripts/vllm/** or **scripts/deepspeed** and change it accordingly to your cluster.


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
* scripts/deepspeed/deepspeed-mii_configurable_benchmarking_serve.sh



## 📊 Result Summary
After the benchmarks are done, generate a summary table:

```bash
# Export environment variables
source .env

# Activate environment.
# IMPORTANT! Change it accordingly in your cluster.
module load miniforge
source activate ${ENVIRONMENT_VLLM}

# Run the generation of the Table.
python generateSummaryTable.py
```

This will output a consolidated report from the files in `results/`.
In MN5:
   * `full_benchmark_summary_MareNostrum 5_ACC.csv`

IMPORTANT! Once you have run all benchmarks, make sure you `push` all your changes in your machine folder inside git repository.

---

## ✅ Requirements
* Miniconda/Anaconda
* GPU and drivers compatible with DeepSpeed/vLLM
* Python 3.10+ recommended
* vLLM, DeepSpeed-MII


---

## 📄 Key Files
### .env
Environment variable definitions for configuring paths and environment variables.

### generateSummaryTable.py
A Python script that likely compiles benchmark results into a summary table for reporting or comparison.

### run_all_benchmarks.sh
A shell script to execute all benchmarks in a batch. Likely the main entry point for running tests.

### README.md
Documentation file (this one) explaining the structure, usage, and purpose of the repository.

---

## 📄 License

This project is licensed under the [Creative Commons BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).  
You are free to use and modify the code **for non-commercial benchmarking purposes**, provided you give appropriate credit.  
Commercial use is **strictly prohibited**.

---

### 💬 Suggestions and Feedback

If you have any suggestions or would like to contribute improvements to this repository, please contact us at **minerva_support@bsc.es**.

