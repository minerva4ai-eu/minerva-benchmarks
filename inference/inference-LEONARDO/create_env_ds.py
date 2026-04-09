import os
import yaml

os.system("module load cuda")
os.system("module load gcc")

# torch from PyPI defaults to CPU-only; force a CUDA 12.1 wheel so that
# NCCL and the distributed back-end are available on Leonardo (CUDA 12.2).
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}

with open("envs-yaml/deepspeed-MII-env.yaml") as file_handle:
    environment_data = yaml.safe_load(file_handle)

for dependency in environment_data["dependencies"]:
    if isinstance(dependency, dict):
        for lib in dependency['pip']:
            pkg_name = lib.split("==")[0].split("+")[0]
            if pkg_name in TORCH_PACKAGES:
                os.system(f"pip install {lib} --index-url {TORCH_INDEX_URL}")
            else:
                os.system(f"pip install {lib}")
