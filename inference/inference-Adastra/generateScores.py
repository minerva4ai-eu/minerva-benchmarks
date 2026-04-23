import json
import os

import pandas as pd

# You need to have activated your virtual enviornment.
from dotenv import load_dotenv

load_dotenv(".env")


BASE_DIR = os.getcwd()  # Adjust if needed
BASE_DIR_RESULTS = os.path.join(BASE_DIR, "results")
SUPCOMPUTER_NAME = os.getenv("SUPCOMPUTER_NAME", "Add to .env file")
GPUS_PER_NODE = os.getenv("GPUS_PER_NODE", "Add to .env file")
PARTITION_NAME = os.getenv("PARTITION_NAME", "Add to .env file")
MODEL_TYPE_MAP = json.load(
    open(os.path.join(BASE_DIR, "configs", "model_type_map.json"))
)

INPUT_FILE = f"results/full_benchmark_summary_{SUPCOMPUTER_NAME}_{PARTITION_NAME}.csv"
OUTPUT_FILE = (
    f"results/full_benchmark_summary_{SUPCOMPUTER_NAME}_{PARTITION_NAME}_score.csv"
)

CSV_HEADERS = [
    "Supercomputer",
    "Partition",
    "Model",
    "Dataset/Model Type",
    "Dataset",
    "Framework",
    "Benchmark Type",
    "Concurrency Level",
    "Number of Nodes",
    "GPUs per Node",
    "Total Used GPUs",
    "Tensor",
    "Pipeline",
    "Max Model Length",
    "Additional Arguments",
    "GPU Memory Usage Avg (GB)",
    "GPU Memory Usage Peak (GB)",
    "Power Usage Avg (W)",
    "Power Usage Peak (W)",
    "TTFT (ms)",
    "ITL (ms)",
    "TPOT (ms)",
    "Output Throughput (tokens/s)",
    "Request Throughput (requests/s)",
]

# Load CSV
df = pd.read_csv(os.path.join(BASE_DIR, INPUT_FILE))

# Latency Score: You want lower latency → higher score. The simplest way is:
latency_scale = 1e4
df["Latency Score"] = 1 / (df["TTFT (ms)"] + df["ITL (ms)"] + df["TPOT (ms)"])
df["Latency Score Scaled"] = df["Latency Score"] * latency_scale

# Throughput Score: higher is better → sum
# df["Throughput Score"] = df["Output Throughput (tokens/s)"] + df["Request Throughput (requests/s)"]

# Throughput Score: Harmonic Mean (penalizes imbalance)
# Higher = Better (fast in both dimensions).
# Throughput Score = 2/(1/Output Throughput + 1/Request Throughput)
df["Throughput Score"] = 2 / (
    (1 / df["Output Throughput (tokens/s)"])
    + (1 / df["Request Throughput (requests/s)"])
)

# Energy score: lower memory/power is better → inverse
# Directly measures efficiency (higher = better (more tokens per Joule).
# df["Energy Score"] = 1 / (df["GPU Memory Usage (GB)"] + df["Power Usage Avg (W)"])
# Energy Score: How many Tokens per Watt
df["Energy Score (Tokens/Watt)"] = (
    (df["Output Throughput (tokens/s)"]) / (df["Power Usage Avg (W)"])
)


# Global score (weights can be tuned): higher = better overall trade-off (balanced best config).
w_latency, w_throughput, w_energy = 0.34, 0.34, 0.32  # 0.4, 0.4, 0.2
df["Global Score"] = (
    w_latency * df["Latency Score Scaled"]
    + w_throughput * df["Throughput Score"]
    + w_energy * df["Energy Score (Tokens/Watt)"]
)

# Sort by best global score
df_sorted = df.sort_values(by="Global Score", ascending=False)

# Save
df_sorted.to_csv(os.path.join(BASE_DIR, OUTPUT_FILE), index=False)
print("Scored results saved to scored_results.csv")
