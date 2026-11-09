"""Generate SYNTHETIC training results for a second cluster (Leonardo Booster), for demonstration only.

    python tools/make_example_training_data.py

Reads the real MareNostrum 5 results in data/training/ and writes
data/training/example-synthetic-leonardo.csv with the same columns plus "Is Synthetic" = True.
The numbers are NOT measurements: each Leonardo run is derived from a MareNostrum 5 run of the same
configuration, scaled the way an A100 node differs from an H100 node, with seeded random variation:

* coverage: the full sweep of the small models, about 60% of the 70B/72B configurations,
* speed: about 0.42x of the H100 run (A100 vs H100 for LLM training), a little worse at many nodes,
* achieved TFLOP/s scales with throughput; MFU is recomputed against the A100 BF16 peak (312 TFLOP/s),
* time per step, per epoch and in total scale with 1 / speed,
* power about 0.72x (A100 vs H100), memory, loss and utilization about the same,
* a share of the large (70B/72B) multi-node configurations that nearly filled the 64 GB on MareNostrum 5
  run out of memory, a few other runs do too, and a few large multi-node runs time out, so that
  failures can be tried out in the dashboard (about 10% of the runs, like a real sweep with problems).

Delete the generated file once real Leonardo results exist.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
REAL = HERE / "data" / "training" / "new-training-results-concatenated-with-params-info.csv"
OUT = HERE / "data" / "training" / "example-synthetic-leonardo.csv"
A100_PEAK_TFLOPS = 312.0
SEED = 2026

CONFIG = ["Model", "Dataset", "Framework", "TypeParallelism", "Tensor Parallel (TP)", "Pipeline Parallel (PP)",
          "Data Parallel (DP)", "Context Parallel (CP)", "Sequence Parallel (SP)", "Expert Parallel (EP)",
          "Number of Nodes", "Batch Size", "Accumulation Gradients"]
OOM = ("CUDA out of memory. Tried to allocate {gib:.2f} GiB. GPU {gpu} has a total capacity of 63.43 GiB of which "
       "{free:.2f} GiB is free. [synthetic failure]")
TIMEOUT = "wait timeout after 1800000ms, keys: /default_pg/0//18//cpu//0/1 [synthetic failure]"


def main():
    rng = np.random.default_rng(SEED)
    real = pd.read_csv(REAL)
    ok = real[real.Comment.isna()]
    rows, step = [], 0
    for _, g in ok.groupby(CONFIG, dropna=False, sort=True):
        base = g.mean(numeric_only=True)
        first = g.iloc[0]
        big = first["Number of Trainable Parameters"] > 20
        if big and rng.random() > 0.6:                     # the 70B/72B runs cover a subset of the configurations
            continue
        step += 1
        nodes = int(first["Number of Nodes"])
        speed = np.clip(rng.normal(0.42, 0.04), 0.32, 0.52) * (1 - 0.02 * np.log2(nodes) * rng.uniform(0.5, 1.5))
        near_limit = base["Peak GPU Memory Usage (GB)"] >= 63.4
        fail = None
        if big and nodes >= 8 and near_limit and rng.random() < 0.5:   # as for the 70B/72B runs on MareNostrum 5
            fail = "oom"
        elif near_limit and rng.random() < 0.03:
            fail = "oom"
        elif nodes >= 8 and rng.random() < 0.08:
            fail = "timeout"
        for rep in range(3):
            r = first.copy()
            r["Job ID"] = 51000000 + step
            r["Step ID"] = step
            r["Repeat ID"] = rep
            r["Date"] = "02-11-2026"
            r["Supercomputer"] = "Leonardo"
            r["Partition"] = "boost_usr_prod"
            r["Source Directory"] = f"SYNTHETIC/leonardo/boost_usr_prod/step-{step}/repeatid-{rep}"
            r["Is Synthetic"] = True
            if fail:
                r["Comment"] = (OOM.format(gib=rng.uniform(8, 40), gpu=int(rng.integers(0, 4)), free=rng.uniform(5, 30))
                                if fail == "oom" else TIMEOUT)
                for c in real.columns:
                    if c.startswith(("Avg.", "Peak")) or c.startswith("Training") or c.startswith("Total Exec"):
                        r[c] = np.nan
            else:
                k = speed * rng.normal(1, 0.008)                          # this repeat's speed vs the H100 run
                thr = base["Training Throughput (tokens/sec)"]
                r["Comment"] = np.nan
                r["Training Throughput (tokens/sec)"] = thr * k
                for c in ("Training Time per Step (sec)", "Training Time per Epoch (sec)", "Total Execution Time (hours)"):
                    r[c] = base[c] / k
                r["Avg. GPU FLOPs"] = base["Avg. GPU FLOPs"] * k
                r["Avg. GPU MFU"] = r["Avg. GPU FLOPs"] / A100_PEAK_TFLOPS * 100
                r["Avg. GPU Utilization"] = float(np.clip(base["Avg. GPU Utilization"] + rng.normal(2, 3), 40, 99.5))
                r["Peak GPU Utilization"] = 100.0
                p = base["Avg. Power Usage (W)"] * rng.normal(0.72, 0.04)
                r["Avg. Power Usage (W)"] = p
                r["Peak Power Usage (W)"] = min(p + (base["Peak Power Usage (W)"] - base["Avg. Power Usage (W)"]) * 0.7, 440.0)
                r["Avg. GPU Memory Usage (GB)"] = min(base["Avg. GPU Memory Usage (GB)"] * rng.normal(1, 0.01), 63.7)
                r["Peak GPU Memory Usage (GB)"] = min(base["Peak GPU Memory Usage (GB)"] * rng.normal(1, 0.005), 63.7)
                r["Avg. Training Loss"] = base["Avg. Training Loss"] * rng.normal(1, 0.004)
            rows.append(r)
    out = pd.DataFrame(rows)[list(real.columns) + ["Is Synthetic"]]
    out.to_csv(OUT, index=False)
    failed = out.Comment.notna().sum()
    print(f"Wrote {OUT.relative_to(HERE)}: {len(out)} synthetic runs ({failed} failed) from {step} configurations")


if __name__ == "__main__":
    main()
