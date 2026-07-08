from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load and normalize CSVs
# -----------------------------
DATA_DIR = Path("data")


def find_col(df, possibilities):
    cols = {c.lower(): c for c in df.columns}

    for p in possibilities:
        if p.lower() in cols:
            return cols[p.lower()]

    for p in possibilities:
        for c_lower, c in cols.items():
            if p.lower() in c_lower:
                return c

    return None


def load_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    canonical = {
        "supercomputer": ["supercomputer", "system", "site"],
        "partition": ["partition"],
        "model": ["model"],
        "dataset": ["dataset"],
        "framework": ["framework"],
        "concurrency_level": [
            "concurrency level",
            "concurrency_level",
            "concurrency",
        ],
        "number_of_nodes": ["number of nodes", "nodes"],
        "total_used_gpus": [
            "Number of GPUs",
            "number of gpus",
            "number_of_gpus",
            "total used gpus",
            "total_used_gpus",
            "total_gpus",
        ],
        "power_avg_w": [
            "power usage avg (w)",
            "power_avg_w",
            "power avg (w)",
        ],
        "output_throughput_toks_s": [
            "output throughput (tokens/s)",
            "output throughput",
            "tokens/s",
        ],
        "request_throughput_reqs_s": [
            "request throughput (requests/s)",
            "request throughput",
            "requests/s",
        ],
        "itl_ms": ["itl (ms)", "itl_ms", "itl"],
        "tpot_ms": ["tpot (ms)", "tpot_ms", "tpot"],
        "ttft_ms": ["ttft (ms)", "ttft_ms", "ttft"],
        "tensor_parallelism": [
            "Tensor",
            "Tensor Parallelism",
            "Tensor Parallel",
            "tensor_parallelism",
            "tensor parallel",
        ],
        "pipeline_parallelism": [
            "Pipeline",
            "Pipeline Parallelism",
            "Pipeline Parallel",
            "pipeline_parallelism",
            "pipeline parallel",
        ],
        "max_model_length": [
            "Max Model Length",
            "Max Length",
            "Max_Model_Length",
            "max_model_length",
            "max length",
        ],
    }

    rename_map = {}
    for canon, variants in canonical.items():
        found = find_col(df, variants)
        if found:
            rename_map[found] = canon

    df = df.rename(columns=rename_map)

    if "supercomputer" not in df.columns:
        df["supercomputer"] = path.stem

    return df


# -----------------------------
# Load all CSVs
# -----------------------------
files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise RuntimeError("No CSV files found in data/")

dfs = [load_and_normalize(f) for f in files]
df_full = pd.concat(dfs, ignore_index=True)

# -----------------------------
# Convert numeric columns
# -----------------------------
numeric_cols = [
    "output_throughput_toks_s",
    "request_throughput_reqs_s",
    "power_avg_w",
    "concurrency_level",
    "number_of_nodes",
    "total_used_gpus",
    "tensor_parallelism",
    "pipeline_parallelism",
    "max_model_length",
]

for c in numeric_cols:
    if c in df_full.columns:
        df_full[c] = pd.to_numeric(df_full[c], errors="coerce")

# -----------------------------
# Filter for Llama models
# -----------------------------
llama_models = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.3-70B-Instruct",
    "Llama-3.1-405B-Instruct",
]

df_filtered = df_full[
    df_full["model"].str.lower().isin(
        [m.lower() for m in llama_models]
    )
].copy()

# Keep only runs with exactly 8 total GPUs
df_filtered = df_filtered[
    df_filtered["total_used_gpus"] == 8
].copy()

# Extract model size (8B, 70B, 405B)
df_filtered["model_size"] = df_filtered["model"].str.extract(
    r"(\d+B)"
)[0]

# -----------------------------
# Debug info
# -----------------------------
print("\n=== FILTERED DATA ===")
print("Frameworks:", sorted(df_filtered["framework"].dropna().unique()))
print("Model sizes:", sorted(df_filtered["model_size"].dropna().unique()))
print("Supercomputers:", sorted(df_filtered["supercomputer"].dropna().unique()))
print("Partitions:", sorted(df_filtered["partition"].dropna().unique()))
print("Node counts:", sorted(df_filtered["number_of_nodes"].dropna().unique()))
print("GPU counts:", sorted(df_filtered["total_used_gpus"].dropna().unique()))

print("\nSample rows:")
print(
    df_filtered[
        [
            "supercomputer",
            "partition",
            "framework",
            "model_size",
            "number_of_nodes",
            "total_used_gpus",
            "output_throughput_toks_s",
        ]
    ].head()
)

# -----------------------------
# Heatmap Function
# -----------------------------
def plot_framework_comparison_heatmap(
    df,
    metric="output_throughput_toks_s",
    filename="framework_comparison_8gpus.png",
):
    required_cols = {
        metric,
        "framework",
        "supercomputer",
        "partition",
        "model_size",
        "number_of_nodes",
    }

    missing = required_cols - set(df.columns)

    if missing:
        print(f"Missing columns: {missing}")
        return

    df_clean = df.dropna(subset=required_cols)

    if df_clean.empty:
        print("No valid data after filtering.")
        return

    # Average metric for duplicate runs
    agg = (
        df_clean.groupby(
            [
                "supercomputer",
                "partition",
                "number_of_nodes",
                "model_size",
                "framework",
            ]
        )[metric]
        .mean()
        .reset_index()
    )

    # Build readable row label
    agg["row_label"] = (
        agg["supercomputer"].astype(str)
        + " | "
        + agg["partition"].astype(str)
        + " | "
        + agg["number_of_nodes"].astype(int).astype(str)
        + " nodes"
        + " | "
        + agg["model_size"].astype(str)
    )

    heatmap_data = agg.pivot(
        index="row_label",
        columns="framework",
        values=metric,
    )

    # Preferred framework ordering
    preferred_order = ["vllm", "sglang"]

    existing_cols = [
        c
        for c in preferred_order
        if c in [x.lower() for x in heatmap_data.columns]
    ]

    if existing_cols:
        col_map = {
            c.lower(): c for c in heatmap_data.columns
        }

        heatmap_data = heatmap_data[
            [col_map[c] for c in existing_cols]
        ]

    if heatmap_data.empty:
        print("Heatmap data is empty.")
        return

    plt.figure(figsize=(10, max(6, len(heatmap_data) * 0.45)))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": metric},
    )

    plt.title(
        "Framework Comparison Heatmap\n"
        "Output Throughput (Tokens/s) — 8 Total GPUs"
    )

    plt.xlabel("Framework")
    plt.ylabel(
        "Supercomputer | Partition | Nodes | Model Size"
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap: {filename}")


# -----------------------------
# Generate Heatmap
# -----------------------------
plot_framework_comparison_heatmap(
    df_filtered,
    metric="output_throughput_toks_s",
    filename="framework_comparison_8gpus.png",
)

print("\nHeatmap generation complete.")