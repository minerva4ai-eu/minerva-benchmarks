"""Collect training summaries from Slurm jobs, steps, and repeats.

Example:
    python generateSummaryTable_jobs.py 45917001 45917002 \
        --root benchmark-runs-MN5-singularity/bsc-mn5-acc \
        --output results/training-summary.csv

The collector writes one row per ``(job, step, repeat)``. Rank/GPU summary
files inside a repeat are combined before the row is written.
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
from scripts.slurm.utils import load_config

DISPLAY_COLUMNS = [
    "Job ID",
    "Step ID",
    "Repeat ID",
    "Date",
    "Supercomputer",
    "Partition",
    "Model",
    "Dataset",
    "Framework",
    "TypeParallelism",
    "Comment",
    "Number of Nodes",
    "GPUs per Node",
    "Total GPUs used",
    "Precision Type",
    "Batch Size",
    "Accumulation Gradients",
    "Max Length",
    "Number of Trainable Parameters",
    "Learning Rate",
    "Dropout",
    "Avg. Power Usage (W)",
    "Peak Power Usage (W)",
    "Avg. GPU Memory Usage (GB)",
    "Peak GPU Memory Usage (GB)",
    "Avg. GPU Utilization",
    "Peak GPU Utilization",
    "Training Time per Step (sec)",
    "Training Time per Epoch (sec)",
    "Total Execution Time (hours)",
    "Training Throughput (tokens/sec)",
    "Avg. Training Loss",
    "Avg. Validation Loss",
    "Source Directory",
]

JSON_TO_DISPLAY = {
    "nodes": "Number of Nodes",
    "num_gpus_per_node": "GPUs per Node",
    "total_gpus": "Total GPUs used",
    "parallelism_type": "TypeParallelism",
    "gradient_accumulation": "Accumulation Gradients",
    "max_length": "Max Length",
    "precision": "Precision Type",
    "trainable_parameters": "Number of Trainable Parameters",
    "avg_gpu_power_watts": "Avg. Power Usage (W)",
    "peak_gpu_power_watts": "Peak Power Usage (W)",
    "avg_gpu_memory_gb": "Avg. GPU Memory Usage (GB)",
    "peak_gpu_memory_gb": "Peak GPU Memory Usage (GB)",
    "avg_gpu_utilization_percent": "Avg. GPU Utilization",
    "peak_gpu_utilization_percent": "Peak GPU Utilization",
    "avg_step_training_time_sec": "Training Time per Step (sec)",
    "avg_epoch_training_time_sec": "Training Time per Epoch (sec)",
    "total_execution_time_hours": "Total Execution Time (hours)",
    "training_throughput_tokens_per_sec_global": "Training Throughput (tokens/sec)",
    "training_loss": "Avg. Training Loss",
    "avg_training_loss": "Avg. Training Loss",
    "validation_loss": "Avg. Validation Loss",
    "avg_validation_loss": "Avg. Validation Loss",
}

PEAK_KEYS = {
    "peak_gpu_power_watts",
    "peak_gpu_memory_gb",
    "peak_gpu_utilization_percent",
}
NON_AGGREGATED_KEYS = {
    "nodes",
    "num_gpus_per_node",
    "total_gpus",
    "model",
    "dataset",
    "framework",
    "parallelism_type",
    "batch_size",
    "gradient_accumulation",
    "learning_rate",
    "precision",
    "max_length",
}

CONFIG_TO_DISPLAY = {
    "model.name": "Model",
    "model.path": "Model Path",
    "model.training.precision": "Precision Type",
    "model.training.batch_size": "Batch Size",
    "model.training.grad_accum": "Accumulation Gradients",
    "model.training.max_model_length": "Max Length",
    "model.training.lr": "Learning Rate",
    "dataset.name": "Dataset",
    "dataset.path": "Dataset Path",
    "framework.name": "Framework",
    "framework.parallelism_name": "TypeParallelism",
    "slurm.partition": "Partition",
    "slurm.sbatch.nodes": "Number of Nodes",
    "slurm.sbatch.gpus_per_node": "GPUs per Node",
    "machine.name": "Supercomputer",
}


def flatten_dict(value, prefix=""):
    flattened = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(flatten_dict(item, name))
        else:
            flattened[name] = item
    return flattened


def read_summary(path):
    try:
        with path.open() as summary_file:
            value = json.load(summary_file)
    except (OSError, json.JSONDecodeError):
        return {}
    return flatten_dict(value) if isinstance(value, dict) else {}


def yaml_path_from_step_log(job_dir, step_dir):
    """Extract the YAML path from runner output or traced shell commands."""
    log_dir = job_dir / "steps-slurm-logs"
    step_number = step_id(step_dir)
    for suffix in ("out", "err"):
        log_path = log_dir / f"srun-{job_dir.name}.{step_number}.{suffix}"
        try:
            lines = log_path.read_text(errors="ignore").splitlines()
        except OSError:
            continue

        for line in reversed(lines):
            matches = re.findall(r"yaml_path\s*=\s*(\S+\.ya?ml)", line) or re.findall(
                r"(?:--yaml|--yaml_file)\s+(\S+\.ya?ml)", line
            )
            if matches:
                return Path(matches[-1]).expanduser()
    return None


def load_step_config(job_dir, step_dir):
    yaml_path = yaml_path_from_step_log(job_dir, step_dir)
    if yaml_path is None:
        return None, {}, "YAML path not found in step log"
    if not yaml_path.exists():
        return yaml_path, {}, f"YAML file not found: {yaml_path}"

    try:
        config: BenchmarkConfig = load_config(str(yaml_path))
        from omegaconf import OmegaConf

        config_dict = OmegaConf.to_container(config, resolve=False)
        return yaml_path, flatten_dict(config_dict), ""
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        return yaml_path, {}, f"Could not load BenchmarkConfig: {error}"


def aggregate_rank_summaries(summary_paths):
    numeric = defaultdict(list)
    text = {}
    for path in summary_paths:
        for key, value in read_summary(path).items():
            if value is None or value == "":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key].append(value)
            elif key not in text:
                text[key] = value

    combined = dict(text)
    for key, values in numeric.items():
        short_key = key.rsplit(".", 1)[-1]
        if short_key in NON_AGGREGATED_KEYS:
            combined[key] = values[0]
        elif short_key in PEAK_KEYS:
            combined[key] = max(values)
        else:
            combined[key] = statistics.mean(values)
    return combined


def find_job_dirs(root, job_ids):
    requested = {str(job_id) for job_id in job_ids}
    matches = []
    for path in root.rglob("training-results"):
        if path.parent.name != "outputs":
            continue
        job_dir = path.parent.parent
        if job_dir.name not in requested:
            continue
        matches.append(job_dir)
    return sorted(set(matches), key=lambda path: (path.name, str(path)))


def step_id(step_dir):
    match = re.fullmatch(r"step-(.+)", step_dir.name)
    return match.group(1) if match else step_dir.name


def repeat_id(repeat_dir):
    match = re.fullmatch(r"repeatid-(.+)", repeat_dir.name)
    return match.group(1) if match else ""


def check_step_error(job_dir, step_dir):
    step_number = step_id(step_dir)
    log_dir = job_dir / "steps-slurm-logs"
    for suffix in ("err", "out"):
        log_path = log_dir / f"srun-{job_dir.name}.{step_number}.{suffix}"
        try:
            content = log_path.read_text(errors="ignore").lower()
        except OSError:
            continue
        if any(text in content for text in ("error", "out of memory", "oom")):
            return "Error during training"
    return ""


def build_row(
    job_dir,
    step_dir,
    repeat_dir,
    metrics,
    config_values,
    yaml_path,
    config_error,
    model_type_map,
    repeat_label=None,
):
    row = {column: "" for column in DISPLAY_COLUMNS}
    row.update(
        {
            "Job ID": job_dir.name,
            "Step ID": step_id(step_dir),
            "Repeat ID": (
                repeat_label if repeat_label is not None else repeat_id(repeat_dir)
            ),
            "Date": job_dir.parent.name,
            "Source Directory": str(repeat_dir),
        }
    )

    if yaml_path is not None:
        row["YAML Path"] = str(yaml_path)
    if config_error:
        row["Comment"] = config_error

    for key, value in config_values.items():
        row[f"Config.{key}"] = value
        display_key = CONFIG_TO_DISPLAY.get(key)
        if display_key and value not in (None, "???"):
            row[display_key] = value

    for key, value in metrics.items():
        display_key = JSON_TO_DISPLAY.get(key.rsplit(".", 1)[-1])
        if display_key:
            row[display_key] = value
        row[key] = value

    if not metrics and not config_error:
        row["Comment"] = metrics.get("error", "Error during training")
    elif metrics.get("error"):
        row["Comment"] = metrics["error"]
    return row


def numeric_step_key(path):
    value = step_id(path)
    return (0, int(value)) if value.isdigit() else (1, value)


def collect_rows(job_dirs, model_type_map):
    rows = []
    for job_dir in job_dirs:
        results_dir = job_dir / "outputs" / "training-results"
        step_dirs = sorted(
            (path for path in results_dir.glob("step-*") if path.is_dir()),
            key=numeric_step_key,
        )
        for step_dir in step_dirs:
            yaml_path, config_values, config_error = load_step_config(job_dir, step_dir)
            repeat_dirs = sorted(
                path for path in step_dir.glob("repeatid-*") if path.is_dir()
            )
            discovered_repeat_dirs = {}
            for repeat_dir in repeat_dirs:
                repeat_match = re.fullmatch(r"repeatid-(\d+)", repeat_dir.name)
                if repeat_match:
                    discovered_repeat_dirs[int(repeat_match.group(1))] = repeat_dir

            try:
                expected_repeats = int(config_values.get("experiment.repeat", 1))
            except (TypeError, ValueError):
                expected_repeats = max(discovered_repeat_dirs, default=-1) + 1
            expected_repeats = max(expected_repeats, 1)

            repeat_items = []
            for repeat_number in range(expected_repeats):
                repeat_dir = discovered_repeat_dirs.get(repeat_number)
                if repeat_dir is None:
                    repeat_dir = step_dir / f"repeatid-{repeat_number}"
                    repeat_items.append((repeat_dir, str(repeat_number)))
                else:
                    repeat_items.append((repeat_dir, None))

            # Preserve any non-numeric repeat directory rather than dropping it.
            repeat_items.extend(
                (repeat_dir, None)
                for repeat_dir in repeat_dirs
                if not re.fullmatch(r"repeatid-\d+", repeat_dir.name)
                or int(re.fullmatch(r"repeatid-(\d+)", repeat_dir.name).group(1))
                >= expected_repeats
            )

            for repeat_dir, repeat_label in repeat_items:
                summaries = sorted(repeat_dir.rglob("training_summary*.json"))
                metrics = aggregate_rank_summaries(summaries)
                row = build_row(
                    job_dir,
                    step_dir,
                    repeat_dir,
                    metrics,
                    config_values,
                    yaml_path,
                    config_error,
                    model_type_map,
                    repeat_label,
                )
                if repeat_label is not None:
                    row["Comment"] = "Did not run"
                elif not metrics:
                    row["Comment"] = (
                        check_step_error(job_dir, step_dir)
                        or row["Comment"]
                        or "Error during training"
                    )
                rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_ids", nargs="+", help="Slurm job IDs to collect")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmark-runs-MN5-singularity"),
        help="Root containing machine/date/job directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/training_summary_jobs.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.root.exists():
        raise SystemExit(f"Results root not found: {args.root}")

    job_dirs = find_job_dirs(args.root, args.job_ids)
    missing = sorted(set(args.job_ids) - {path.name for path in job_dirs})
    if missing:
        print(f"Warning: no training results found for job IDs: {', '.join(missing)}")

    rows = collect_rows(job_dirs, {})
    raw_columns = sorted(
        {key for row in rows for key in row if key not in DISPLAY_COLUMNS}
    )
    fieldnames = DISPLAY_COLUMNS + raw_columns
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Training summary CSV written to: {args.output}")
    print(f"Job directories processed: {len(job_dirs)}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()
