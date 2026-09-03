##############################################################
# All configuration modules to be included are written below #
# Keep as is, to have a reference of complete foncigurations #
# to be used for all benchmarks generated.                   #
#                                                            #
# If needed to generate only specific or customized new      #
# benchmarks, copy and override values.                      #
#
# Example:
#
#   MODELS = ["llama3_8b", "mistral_7b", "llama3_70b"]
#   FRAMEWORKS = ["torchrun", "deepspeed"]
#   DATASETS = ["alpaca", "squadv2"]
#   ...
#   MODELS = ["new_model_config"]
#   FRAMEWORKS = ["torchrun", "deepspeed"]
#   DATASETS = ["alpaca", "squadv2", "new_dataset_config"]
MODELS = [
    "gemma3_1b",
    "qwen2.5_7B_Instruct",
    "qwen2.5_72B_Instruct",
    "mistral_7b",
    "llama3_8b",
    "gemma3_12b",
    "llama3_70b",
]
FRAMEWORKS = [
    "accelerate-cuda130",
    "torchrun-cuda130",
    "deepspeed-accelerate-cuda130",
    "megatron-nemo-2509",
    "deepspeed-cuda130",
]
DATASETS = ["alpaca", "squadv2", "tulu-3-sft-mixture"]

################################################################]
AVAILABLE_FRAMEWORKS = [
    "accelerate-cuda121",
    "accelerate-cuda128",
    "accelerate-cuda130",
    "torchrun-cuda121",
    "torchrun-cuda128",
    "torchrun-cuda130",
    "deepspeed-accelerate-cuda121",
    "deepspeed-accelerate-cuda128",
    "deepspeed-accelerate-cuda130",
    "deepspeed-cuda121",
    "deepspeed-cuda128",
    "deepspeed-cuda130",
    "megatron-nemo-2509",
]

MEGATRON_PARALLELISM_AVAILABLE = ["tp", "pp", "cp", "dp", "ep"]
