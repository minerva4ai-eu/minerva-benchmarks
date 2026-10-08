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
    "MareNostrum5 ACC": "MareNostrum 5", "Jean Zay H100 gpu_p6": "Jean Zay H100",
    "Leonardo boost_usr_prod": "Leonardo Booster", "LUMI LUMI-G": "LUMI-G"}


def system_label(df):
    """Short, consistent system names (shared by the inference and training data)."""
    same = df.system.str.lower() == df.partition.str.lower()
    return pd.Series(np.where(same, df.system, df.system + " " + df.partition),
                     index=df.index).replace(LABELS)


def load_runs(data_dir: Path = DATA_DIR):
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
    df["system_label"] = system_label(df)
    df["extra_args"] = df.extra_args.fillna("none")
    return df, notes


TRAIN_DIR = DATA_DIR / "training"
TRAIN_RENAME = {
    "Supercomputer": "system", "Partition": "partition", "Model": "model",
    "Dataset": "dataset", "Framework": "framework", "TypeParallelism": "parallelism",
    "Comment": "comment", "Number of Nodes": "nodes", "Total GPUs used": "gpus",
    "Precision Type": "precision", "Batch Size": "batch_size",
    "Accumulation Gradients": "accum", "Max Length": "max_len",
    "Number of Trainable Parameters": "params",
    "Avg. Power Usage (W)": "power_w", "Peak Power Usage (W)": "peak_power_w",
    "Avg. GPU Memory Usage (GB)": "mem_gb", "Peak GPU Memory Usage (GB)": "peak_mem_gb",
    "Avg. GPU Utilization": "util", "Peak GPU Utilization": "peak_util",
    "Training Time per Step (sec)": "step_s", "Training Time per Epoch (sec)": "epoch_s",
    "Total Execution Time (hours)": "hours", "Training Throughput (tokens/sec)": "tok_s",
    "Avg. Training Loss": "loss", "Avg. Validation Loss": "val_loss",
}
TRAIN_REQUIRED = ["Supercomputer", "Partition", "Model", "Dataset", "Framework", "TypeParallelism",
                  "Number of Nodes", "Total GPUs used", "Precision Type", "Batch Size",
                  "Accumulation Gradients", "Max Length", "Training Throughput (tokens/sec)"]


def load_training(data_dir: Path = TRAIN_DIR):
    """Return (clean DataFrame, notes) for the training summaries in data/training/.

    Raises FileNotFoundError when there are no CSVs yet, ValueError when a file lacks
    one of the required columns. Metric columns are optional and may be empty.
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
    for c in TRAIN_RENAME:
        if c not in df.columns:
            df[c] = np.nan
    df = df[list(TRAIN_RENAME)].rename(columns=TRAIN_RENAME)
    notes = []

    df["failed"] = df.tok_s.isna() | (df.tok_s <= 0)
    df["comment"] = df.comment.fillna("")
    if df.failed.any():
        notes.append(f"{df.failed.sum()} failed training runs (no throughput, for example out of "
                     "memory) are hidden by default; the Run map shows them.")

    ok = ~df.failed
    df["tok_s_gpu"] = (df.tok_s / df.gpus).where(ok)
    df["tok_per_j"] = (df.tok_s / (df.power_w * df.gpus)).where(ok)   # tokens per joule
    df["kwh"] = (df.power_w * df.gpus * df.hours / 1000).where(ok)
    df["params_m"] = df.params / 1e6
    df["system_label"] = system_label(df)
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
