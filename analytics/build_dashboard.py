"""Build the static dashboard: analytics/data/*.csv -> analytics/dashboard/index.html

    python build_dashboard.py

The output is one self-contained file (data embedded, Plotly from a CDN).
Open it in a browser or publish the `dashboard/` folder with GitHub Pages.
"""
import base64
import json
from datetime import date
from pathlib import Path

from minerva_data import check_scores, load_runs, load_training

HERE = Path(__file__).resolve().parent
LOGO = HERE / "dashboard" / "assets" / "minerva-logo.png"
COLUMNS = ["system_label", "model", "framework", "dataset", "concurrency", "nodes",
           "gpus", "tp", "pp", "max_len", "extra_args", "failed", "mem_peak_gb",
           "power_w", "ttft_ms", "itl_ms", "tpot_ms", "tok_s", "req_s",
           "j_per_tok", "j_per_tok_all", "tok_s_per_gpu", "latency_score", "throughput_score",
           "energy_score", "global_score"]


TRAIN_EXPORT = {  # CSV-side name -> name used by the dashboard
    "system_label": "system_label", "model": "model", "dataset": "dataset", "framework": "framework",
    "parallelism": "parallelism", "precision": "precision", "nodes": "nodes", "gpus": "gpus",
    "batch_size": "batch_size", "accum": "accum", "max_len": "max_len", "comment": "comment",
    "failed": "failed", "params_m": "params_m", "tok_s": "tr_tok_s", "tok_s_gpu": "tr_tok_s_gpu",
    "epoch_s": "tr_epoch_s", "hours": "tr_hours", "step_s": "tr_step_s", "power_w": "tr_power_w",
    "peak_power_w": "tr_peak_power_w", "kwh": "tr_kwh", "tok_per_j": "tr_tok_per_j",
    "mem_gb": "tr_mem_gb", "peak_mem_gb": "tr_peak_mem_gb", "util": "tr_util",
    "peak_util": "tr_peak_util", "loss": "tr_loss", "val_loss": "tr_val_loss"}


def training_payload():
    """None when there is no training data yet; {"error": ...} when it cannot be read."""
    try:
        df, notes = load_training()
    except FileNotFoundError:
        return None
    except Exception as exc:  # a malformed training file must not break the inference dashboard
        print(f"Warning: training data not loaded: {exc}")
        return {"error": str(exc)}
    df = df[list(TRAIN_EXPORT)].rename(columns=TRAIN_EXPORT).round(4)
    print(f"Training: {len(df)} runs")
    return {"columns": list(df.columns),
            "data": {c: [None if v != v else v for v in df[c].tolist()] for c in df.columns},
            "notes": notes}


def build_html():
    """Return (html, number_of_inference_runs). Touches no files, so it also works from a
    read-only container image."""
    df, notes = load_runs()
    scores = check_scores(df)  # before rounding, so the check is exact
    if not scores["ok"]:
        notes.append("Warning: the scores in the CSVs no longer match the formulas shown in the "
                     f"dashboard guide (largest deviation {scores['max_error']:.2g}). "
                     "Check inference/generateScores.py.")
    df = df[COLUMNS].round(4)
    df[["j_per_tok", "j_per_tok_all"]] = df[["j_per_tok", "j_per_tok_all"]].round(6)  # tiny values need more digits
    payload = {
        "columns": COLUMNS,
        "data": {c: [None if v != v else v for v in df[c].tolist()] for c in COLUMNS},
        "notes": notes,
        "scores": scores,
        "training": training_payload(),
        "defaults": json.loads((HERE / "dashboard" / "defaults.json").read_text()),
        "built": date.today().isoformat(),
    }
    template = (HERE / "dashboard" / "template.html").read_text(encoding="utf-8")
    if LOGO.exists():
        logo = "data:image/png;base64," + base64.b64encode(LOGO.read_bytes()).decode("ascii")
    else:
        print(f"Warning: {LOGO.relative_to(HERE)} not found; building without the logo")
        logo = ""
    html = template.replace("__LOGO__", logo).replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    return html, len(df)


def main():
    html, n_runs = build_html()
    target = HERE / "dashboard" / "index.html"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(target)  # atomic: visitors never receive a half-written page
    print(f"Wrote dashboard/index.html ({n_runs} runs, {len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
