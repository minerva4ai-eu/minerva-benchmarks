"""Shared loading and cleaning for the Minerva benchmark CSVs.

Usage:
    from minerva_data import load_runs
    df, notes = load_runs()          # notes = human-readable data-quality log

Paths are resolved relative to this file, so scripts work from any directory.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
INFER_DIR = DATA_DIR / "inference"

RENAME = {
    "Supercomputer": "system", "Partition": "partition", "Model": "model",
    "Dataset": "dataset", "Framework": "framework",
    "Concurrency Level": "concurrency", "Number of Nodes": "nodes",
    "GPUs per Node": "gpus_per_node", "Total Used GPUs": "gpus",
    "Tensor": "tp", "Pipeline": "pp", "Max Model Length": "max_len",
    "Additional Arguments": "extra_args",
    "GPU Memory Usage Peak (GB)": "mem_peak_gb",
    "Power Usage Avg (W)": "power_w", "TTFT (ms)": "ttft_ms",
    "ITL (ms)": "itl_ms", "TPOT (ms)": "tpot_ms",
    "Output Throughput (tokens/s)": "tok_s", "Request Throughput (requests/s)": "req_s",
    "Latency Score Scaled": "latency_score",
    "Throughput Score": "throughput_score",
    "Energy Score (Tokens/Watt)": "energy_score",
    "Global Score": "global_score",
}
SCORES = ["latency_score", "throughput_score", "energy_score", "global_score"]


LABELS = {
    "ADASTRA MI250": "Adastra MI250", "ADASTRA MI300": "Adastra MI300",
    "MareNostrum5 ACC": "MareNostrum 5", "bsc-mn5-acc acc": "MareNostrum 5", "Jean Zay H100 gpu_p6": "Jean Zay H100",
    "Leonardo boost_usr_prod": "Leonardo Booster", "LUMI LUMI-G": "LUMI-G"}


def system_label(df):
    """Short, consistent system names (shared by the inference and training data)."""
    same = df.system.str.lower() == df.partition.str.lower()
    return pd.Series(np.where(same, df.system, df.system + " " + df.partition),
                     index=df.index).replace(LABELS)


def load_runs(data_dir: Path = INFER_DIR):
    """Return (clean DataFrame, list of data-quality notes)."""
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[list(RENAME)].rename(columns=RENAME)
    notes = []

    # 1. Rows whose result path was parsed one level too high.
    shifted = (df.framework == "results") & (df.dataset == "vllm")
    if shifted.any():
        df.loc[shifted, ["framework", "dataset"]] = ["vllm", "unknown"]
        notes.append(f"{shifted.sum()} runs had Framework='results' and Dataset='vllm' "
                     "(path parsing shifted). Framework set to vllm; dataset is not "
                     "recoverable and is shown as 'unknown'.")

    # 2. Failed runs: zero throughput / inf scores. Kept but flagged.
    df[SCORES] = df[SCORES].replace([np.inf, -np.inf], np.nan)
    df["failed"] = (df.tok_s <= 0) | df.global_score.isna()
    df.loc[df.failed, SCORES] = np.nan
    if df.failed.any():
        notes.append(f"{df.failed.sum()} failed runs (zero throughput or infinite score) "
                     "are hidden by default.")

    # 3. Derived metrics (NaN for failed runs so they never skew averages).
    # latency/throughput/energy/global scores come from inference/generateScores.py.
    ok = ~df.failed
    df["j_per_tok"] = (df.power_w / df.tok_s).where(ok)
    df["tok_s_per_gpu"] = (df.tok_s / df.gpus).where(ok)
    df["j_per_tok_all"] = (df.power_w * df.gpus / df.tok_s).where(ok)  # J/token for all GPUs together
    df["system_label"] = system_label(df)
    df["extra_args"] = df.extra_args.fillna("none")
    return df, notes


TRAIN_DIR = DATA_DIR / "training"
TRAIN_RENAME = {
    "Supercomputer": "system", "Partition": "partition", "Model": "model",
    "Dataset": "dataset", "Framework": "framework", "TypeParallelism": "parallelism",
    "Tensor Parallel (TP)": "tp", "Pipeline Parallel (PP)": "pp", "Data Parallel (DP)": "dp",
    "Context Parallel (CP)": "cp", "Sequence Parallel (SP)": "sp", "Expert Parallel (EP)": "ep",
    "Comment": "comment", "Number of Nodes": "nodes", "Total GPUs used": "gpus",
    "Precision Type": "precision", "Batch Size": "batch_size",
    "Accumulation Gradients": "accum", "Max Length": "max_len",
    "Number of Trainable Parameters": "params", "Repeat ID": "repeat",
    "Source Directory": "source", "Is Synthetic": "synthetic",
    "Avg. Power Usage (W)": "power_w", "Peak Power Usage (W)": "peak_power_w",
    "Avg. GPU Memory Usage (GB)": "mem_gb", "Peak GPU Memory Usage (GB)": "peak_mem_gb",
    "Avg. GPU Utilization": "util", "Peak GPU Utilization": "peak_util",
    "Avg. GPU FLOPs": "tflops", "Avg. GPU MFU": "mfu",
    "Training Time per Step (sec)": "step_s", "Training Time per Epoch (sec)": "epoch_s",
    "Total Execution Time (hours)": "hours", "Training Throughput (tokens/sec)": "tok_s",
    "Avg. Training Loss": "loss", "Avg. Validation Loss": "val_loss",
}
# "GPUs per Node" is deliberately not read: in the final results it is 1 for FSDP, ZeRO and Megatron
# runs although every GPU of each node is used (Total GPUs used = 4 x nodes).
TRAIN_REQUIRED = ["Supercomputer", "Partition", "Model", "Dataset", "Framework", "TypeParallelism",
                  "Number of Nodes", "Total GPUs used", "Precision Type", "Batch Size",
                  "Accumulation Gradients", "Max Length", "Training Throughput (tokens/sec)"]
NA = "n/a"


def _failure_kind(comment: str, no_result: bool) -> str:
    """Short category for a failed run, from its (long) error comment."""
    c = comment.lower()
    if not c:
        return "No result" if no_result else ""
    for words, kind in [(("out of memory",), "Out of memory"), (("did not run",), "Did not run"),
                        (("timeout", "timed out"), "Timeout"), (("nccl",), "NCCL error"),
                        (("killed", "terminated"), "Process killed")]:
        if any(w in c for w in words):
            return kind
    return "Other error"


def _parallel_config(r) -> str:
    """Readable parallelism setup: 'TP4 PP2 DP4' for runs that report TP/PP/DP/..., else the strategy name."""
    sizes = {k: r[k] for k in ("tp", "pp", "dp", "cp", "ep") if r[k] != NA}
    if not sizes:
        return "megatron (TP/PP/DP not recorded)" if "megatron" in str(r["parallelism"]) else str(r["parallelism"])
    parts = [f"{k.upper()}{int(v)}" for k, v in sizes.items() if k in ("tp", "pp", "dp") or v > 1]
    if r["sp"] == "yes":
        parts.append("SP")
    return " ".join(parts)


def _parallel_family(r) -> str:
    kind = str(r["parallelism"]).lower()
    if r["tp"] != NA or r["pp"] != NA or "megatron" in kind:
        return "Tensor / pipeline parallel"
    if kind in ("fsdp", "zero2", "zero3") or kind.startswith(("fsdp", "zero")):
        return "Sharded data parallel"
    if kind == "ddp":
        return "Data parallel (DDP)"
    return "No parallelism" if kind == "none" else kind


def load_training(data_dir: Path = TRAIN_DIR):
    """Return (clean DataFrame, notes) for the training summaries in data/training/.

    Raises FileNotFoundError when there are no CSVs yet, ValueError when a file lacks one of the
    required columns. Metric and parallelism-size columns are optional and may be empty.
    Rows with a Comment (an error message, or "Did not run") are flagged `failed`.
    """
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No training CSV files in {data_dir}")
    frames = []
    for f in files:
        part = pd.read_csv(f)
        missing = [c for c in TRAIN_REQUIRED if c not in part.columns]
        if missing:
            raise ValueError(f"{f.name} is missing required columns: {', '.join(missing)}")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    if "Is Example" in df.columns:                      # older name of the "Is Synthetic" column
        df["Is Synthetic"] = df["Is Synthetic"].fillna(df["Is Example"]) if "Is Synthetic" in df.columns else df["Is Example"]
    for c in TRAIN_RENAME:
        if c not in df.columns:
            df[c] = np.nan
    df = df[list(TRAIN_RENAME)].rename(columns=TRAIN_RENAME)
    notes = []

    df["synthetic"] = df.synthetic.map({True: True, "True": True, "true": True}).fillna(False).astype(bool)
    df["comment"] = (df.comment.fillna("").astype(str).str.replace(r"\\n|\n", " ", regex=True)
                     .str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, 300))
    no_result = (df.tok_s.isna() | (df.tok_s <= 0)).to_numpy()
    df["failed"] = no_result | (df.comment != "").to_numpy()
    df["failure"] = [_failure_kind(c, n) for c, n in zip(df.comment, no_result)]
    df.loc[df.failed & (df.failure == ""), "failure"] = "Other error"
    if df.failed.any():
        notes.append(f"{int(df.failed.sum())} runs with a comment (an error, or \"Did not run\") are hidden "
                     "by default; the Heatmap shows them as failed cells.")

    # Runs that never started have no GPU count: nodes x the usual GPUs per node of that system.
    df["system_label"] = system_label(df)
    per_node = (df.gpus / df.nodes).groupby(df.system_label).median()
    fill = df.gpus.isna() & df.nodes.notna()
    if fill.any():
        df.loc[fill, "gpus"] = df.loc[fill, "nodes"] * df.loc[fill, "system_label"].map(per_node)
        notes.append(f"{int(fill.sum())} runs had no GPU count (they did not run); it was set to nodes x "
                     "the GPUs per node of the same system.")

    # Samples per optimizer step: batch x accumulation x data-parallel size (all GPUs for DDP/FSDP/ZeRO).
    df["global_batch"] = (df.batch_size * df.accum * df.dp.where(df.dp.notna(), df.gpus)).round()

    # Parallelism: TP/PP/DP/CP/EP only exist for Megatron-style runs, "n/a" for the others.
    for c in ("tp", "pp", "dp", "cp", "ep"):
        df[c] = pd.Series([int(v) if v == v else NA for v in df[c]], index=df.index, dtype=object)
    df["sp"] = pd.Series([{True: "yes", False: "no", "True": "yes", "False": "no"}.get(v, NA) for v in df.sp],
                         index=df.index, dtype=object)
    df["parallel_config"] = df.apply(_parallel_config, axis=1)
    df["parallel_family"] = df.apply(_parallel_family, axis=1)

    ok = ~df.failed
    df["tok_s_gpu"] = (df.tok_s / df.gpus).where(ok)
    df["tok_per_j"] = (df.tok_s / (df.power_w * df.gpus)).where(ok)          # tokens per joule
    df["kwh"] = (df.power_w * df.gpus * df.hours / 1000).where(ok)
    df["tflops_total"] = (df.tflops * df.gpus).where(ok)                      # TFLOP/s of the whole run
    df["gflops_w"] = (df.tflops * 1000 / df.power_w).where(ok)                # GFLOP/s per watt
    # Parameter counts come either in billions (final results) or as absolute numbers.
    df["params_b"] = np.where(df.params > 1e5, df.params / 1e9, df.params)

    # Synthetic data must be recognisable in every chart, table and screenshot.
    df.loc[df.synthetic, "system_label"] = df.loc[df.synthetic, "system_label"] + " (synthetic)"
    synthetic = sorted(df.loc[df.synthetic, "system_label"].unique())
    if synthetic:
        notes.append("Generated numbers, not measurements: " + ", ".join(synthetic) + ".")
    return df, notes


def check_scores(df):
    """Check the CSV scores against the formulas documented in the dashboard.

    Returns the global-score weights fitted from the data and the largest relative
    deviation from the formulas in inference/generateScores.py.
    """
    ok = df[~df.failed]
    lat = 1e4 / (ok.ttft_ms + ok.itl_ms + ok.tpot_ms)
    thr = 2 / (1 / ok.tok_s + 1 / ok.req_s)
    en = ok.tok_s / ok.power_w
    err = max(((lat - ok.latency_score).abs() / lat).max(),
              ((thr - ok.throughput_score).abs() / thr).max(),
              ((en - ok.energy_score).abs() / en).max())
    parts = ok[["latency_score", "throughput_score", "energy_score"]].to_numpy()
    w = np.linalg.lstsq(parts, ok.global_score.to_numpy(), rcond=None)[0].round(3)
    err = max(err, float((np.abs(parts @ w - ok.global_score) / ok.global_score).max()))
    return {"weights": dict(zip(["latency", "throughput", "energy"], w.tolist())),
            "max_error": float(err), "ok": bool(err < 1e-6)}


if __name__ == "__main__":
    runs, log = load_runs()
    print(f"{len(runs)} inference runs from {runs.system_label.nunique()} systems")
    print(*log, sep="\n")
    try:
        train, tlog = load_training()
        print(f"{len(train)} training runs from {train.system_label.nunique()} systems")
        print(*tlog, sep="\n")
    except FileNotFoundError as e:
        print(e)
