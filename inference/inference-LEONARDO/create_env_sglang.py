import os
import yaml

os.system("module load cuda")
os.system("module load gcc")

with open("envs-yaml/sglang-0.5.6.post1.yaml") as file_handle:
    environment_data = yaml.safe_load(file_handle)

for dependency in environment_data["dependencies"]:
    if isinstance(dependency, dict):
      for lib in dependency['pip']:
        os.system(f"pip install {lib}")
