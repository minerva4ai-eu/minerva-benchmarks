from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("plots_llama8b_framework_compare")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_MODEL = "8B"

# -----------------------------
# Helper
# -----------------------------
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


# -----------------------------
# Load + normalize
# -----------------------------
def load_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    canonical = {
        "supercomputer": ["supercomputer", "system", "site"],
        "model": ["model"],
        "framework": ["framework"],
        "total_used_gpus": [
            "Number of GPUs",
            "number of gpus",
            "number_of_gpus",
            "total used gpus",
            "total_used_gpus",
        ],
        "power_avg_w": ["power usage avg (w)", "power_avg_w", "power avg (w)"],
        "output_throughput_toks_s": [
            "output throughput (tokens/s)",
            "output throughput",
            "tokens/s",
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
# Load data
# -----------------------------
files = sorted(DATA_DIR.glob("*.csv"))
if not files:
    raise RuntimeError("No CSV files found in data/")

df = pd.concat([load_and_normalize(f) for f in files], ignore_index=True)

# -----------------------------
# Numeric cleanup
# -----------------------------
for c in ["power_avg_w", "total_used_gpus", "output_throughput_toks_s"]:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")

# -----------------------------
# Filter: Llama 8B ONLY (both frameworks)
# -----------------------------
df = df[
    (df["model"].isin(["Llama-3.1-8B-Instruct"])) &
    (df["framework"].str.lower().isin(["sglang", "vllm"])) &
    (df["dataset"].str.lower().isin(["sharegpt"]))
].copy()

# -----------------------------
# Clean
# -----------------------------
df = df.dropna(subset=[
    "supercomputer",
    "framework",
    "power_avg_w",
    "total_used_gpus",
    "output_throughput_toks_s"
])

df = df[df["output_throughput_toks_s"] > 0].copy()

# -----------------------------
# Compute J/token
# -----------------------------
df["total_power_w"] = df["power_avg_w"] * df["total_used_gpus"]
df["j_per_token"] = df["total_power_w"] / df["output_throughput_toks_s"]

# -----------------------------
# Aggregate: supercomputer × framework
# -----------------------------
agg = (
    df.groupby(["supercomputer", "framework"])["j_per_token"]
    .mean()
    .reset_index()
)

# -----------------------------
# Pivot for grouped bar chart
# -----------------------------
pivot = agg.pivot(index="supercomputer", columns="framework", values="j_per_token")

# -----------------------------
# SORT by BEST performance (lowest J/token across frameworks)
# -----------------------------
pivot["best"] = pivot.min(axis=1)   # best (lowest) per supercomputer
pivot = pivot.sort_values("best")   # sort ascending (best first)
pivot = pivot.drop(columns=["best"])

# -----------------------------
# Plot
# -----------------------------
ax = pivot.plot(kind="bar", figsize=(10, 6))

plt.title("Energy Efficiency Across Supercomputers (Llama 8B)")
plt.xlabel("Supercomputer")
plt.ylabel("Joule per Token (lower is better)")
plt.grid(axis="y", linestyle="--", linewidth=0.5)

plt.xticks(rotation=30, ha="right")
plt.legend(title="Framework")

plt.tight_layout()

out_path = OUTPUT_DIR / "llama8b_j_per_token_sglang_vs_vllm_sorted.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {out_path}")