import csv
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# -------------------------
# Load environment & config
# -------------------------
load_dotenv(".env")

BASE_DIR = os.getcwd()
BASE_DIR_RESULTS = os.path.join(BASE_DIR, "results")
SUPCOMPUTER_NAME = os.getenv("SUPCOMPUTER_NAME", "Add to .env file")
GPUS_PER_NODE = os.getenv("GPUS_PER_NODE", "")
PARTITION_NAME = os.getenv("PARTITION_NAME", "Add to .env file")

MODEL_TYPE_MAP_PATH = os.path.join(BASE_DIR, "configs", "model_type_map.json")
MODEL_TYPE_MAP = {}
if os.path.exists(MODEL_TYPE_MAP_PATH):
    try:
        MODEL_TYPE_MAP = json.load(open(MODEL_TYPE_MAP_PATH))
    except Exception:
        MODEL_TYPE_MAP = {}

SUMMARY_FILENAME = "output/training_summary_0.json"
OUTPUT_FILE = (
    f"results/full_benchmark_training_summary_{SUPCOMPUTER_NAME}_{PARTITION_NAME}.csv"
)

# -------------------------
# Desired CSV columns (training)
# -------------------------
COLUMNS = [
    "Supercomputer",
    "Partition",
    "Model",
    "Dataset/Model Type",
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
]


# -------------------------
# Helpers
# -------------------------
def shorten_path(value):
    """If value looks like a path, return last path component; else return as-is."""
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return Path(value).name
    return value


def parse_run_path(launch_path: Path):
    """
    Expecting folder structure similar to:
    ./results/<framework>/<dataset>/<model>/<Nodes_4-GPUs_16-...>/launch-1

    Return a dict with fields for metadata.
    """
    parts = launch_path.parts
    # Default unknowns
    dataset_name = ""
    model_name = ""
    config_str = ""
    # Try to extract
    # parts example: ('.', 'results', 'torchrun', 'alpaca', 'Llama-3.1-8B-Instruct', 'Nodes_4-GPUs_16-...', 'launch-1')
    try:
        # find index of 'results' then 'torchrun' to be more robust
        if "results" in parts:
            idx = parts.index("results")
            # expect next is 'torchrun' or similar - try to take dataset/model afterwards
            # safe access:
            if len(parts) > idx + 2:
                # parts[idx+1] would be framework
                framework = parts[idx + 1]
            if len(parts) > idx + 3:
                # parts[idx+2] would be dataset
                dataset_name = parts[idx + 2]
            if len(parts) > idx + 4:
                model_name = parts[idx + 3]
            if len(parts) >= idx + 5:
                config_str = parts[idx + 4]
    except Exception:
        pass

    # parse config_str like Nodes_4-GPUs_16-Parallelism_ddp-Precision_bf16-BS_4-GAS_4-MaxModelLength_2048
    conf = {}

    if config_str:
        for chunk in config_str.split("-"):
            if "_" in chunk:
                k, v = chunk.split("_", 1)
                conf[k.lower()] = v

    return {
        "Framework": framework,
        "Dataset": dataset_name,
        "Model": model_name,
        "Number of Nodes": conf.get("nodes", ""),
        "GPUs per Node": GPUS_PER_NODE,
        "Total GPUs used": conf.get("gpus", ""),
        "TypeParallelism": conf.get("parallelism", ""),
        "Precision Type": conf.get("precision", ""),
        "Batch Size": conf.get("bs", ""),
        "Accumulation Gradients": conf.get("gas", ""),
        "Max Length": conf.get("maxmodellength", ""),
    }


def determine_model_type(model_name: str):
    """Find model type using MODEL_TYPE_MAP regex keys; fallback 'Unknown'."""
    if not model_name:
        return "Unknown"
    for pattern, mtype in MODEL_TYPE_MAP.items():
        try:
            if re.search(pattern, model_name, re.IGNORECASE):
                return mtype
        except re.error:
            # if pattern is not a valid regex, do simple substring match
            if pattern.lower() in model_name.lower():
                return mtype
    return "Unknown"


def check_cuda_oom_error(launch_dir: Path):
    """
    Look for any run-*.err files inside the given launch directory
    and return True if a CUDA OOM error is detected.
    """
    for err_file in launch_dir.glob("run-*.err"):
        if not err_file.is_file():
            continue
        try:
            with err_file.open("r", errors="ignore") as f:
                content = f.read().lower()
                if (
                    "cuda out of memory" in content
                    or "cudaerrormemoryallocation" in content
                    or "runtimeerror: cuda error" in content
                    or "out of memory" in content
                ):
                    return True
        except Exception:
            continue
    return False


def is_key_metrics_empty(metrics: dict):
    """
    Return True if the key metrics (GPU power, throughput, execution time) are all empty or missing.
    """
    keys_to_check = [
        "Number of Trainable Parameters",
        "Peak Power Usage (W)",
        "Total Execution Time (hours)",
    ]
    for key in keys_to_check:
        if key in metrics and metrics[key] not in ("", None):
            return False
    return True


def read_json_metrics(json_path: Path):
    """
    Read JSON and map fields to the CSV columns where appropriate.
    Return dict of column->value (only for columns that are JSON-derived).
    """
    if not json_path.exists():
        return {}

    try:
        with json_path.open("r") as f:
            raw = json.load(f)
    except Exception:
        return {}

    # copy and map keys: we include most keys from JSON that match our columns
    out = {}

    # direct mappings where json key names differ from desired column names:
    mappings = {
        "trainable_parameters": "Number of Trainable Parameters",
        "total_trainable_parameters": "Total Trainable Parameters",
        "trainable_parameters_percentage": "Trainable Parameters Percentage",
        "learning_rate": "Learning Rate",
        "dropout": "Dropout",
        "avg_gpu_power_watts": "Avg. Power Usage (W)",
        "peak_gpu_power_watts": "Peak Power Usage (W)",
        "avg_gpu_memory_gb": "Avg. GPU Memory Usage (GB)",
        "peak_gpu_memory_gb": "Peak GPU Memory Usage (GB)",
        "avg_gpu_utilization_percent": "Avg. GPU Utilization",
        "peak_gpu_utilization_percent": "Peak GPU Utilization",
        "avg_epoch_training_time_sec": "Training Time per Epoch (sec)",
        "avg_step_training_time_sec": "Training Time per Step (sec)",
        "total_execution_time_hours": "Total Execution Time (hours)",
        "training_throughput_tokens_per_sec": "Training Throughput (tokens/sec)",
        "avg_training_loss": "Avg. Training Loss",
        "avg_validation_loss": "Avg. Validation Loss",
    }

    # first shorten path-like entries
    processed = {}
    for k, v in raw.items():
        processed[k] = shorten_path(v)

    # populate mapped columns
    for jkey, col in mappings.items():
        if jkey in processed:
            val = processed[jkey]
            # convert epoch sec -> minutes if required
            # if jkey == "avg_epoch_training_time_sec":
            #     try:
            #         val_num = float(val)
            #         # Training Time per Epoch (min)
            #         out[col] = round(val_num / 60.0, 6)
            #     except Exception:
            #         out[col] = ""
            if jkey == "avg_epoch_training_time_hours":
                # only set if we didn't set from sec
                if out.get("Training Time per Epoch (min)", "") == "":
                    try:
                        val_num = float(val)
                        out["Training Time per Epoch (min)"] = round(val_num * 60.0, 6)
                    except Exception:
                        pass
            else:
                out[col] = val

    # some JSON keys directly match our column names; copy those too if present
    for col in [
        "Avg. Power Usage (W)",
        "Peak Power Usage (W)",
        "Avg. GPU Memory Usage (GB)",
        "Peak GPU Memory Usage (GB)",
        "Avg. GPU Utilization",
        "Peak GPU Utilization",
        "Avg. Training Loss",
        "Avg. Validation Loss",
        "Training Throughput (tokens/sec)",
        "Total Execution Time (hours)",
    ]:
        # reverse map: see if any JSON key maps to this column already set
        if col in out:
            continue

    # Also ensure framework/model/dataset from JSON (shortened)
    if "framework" in processed and not out.get("Framework"):
        out["Framework"] = processed["framework"]
    if "model" in processed and not out.get("Model"):
        out["Model"] = processed["model"]
    if "dataset" in processed and not out.get("Dataset"):
        out["Dataset"] = processed["dataset"]

    # last: some numeric values may be present under different keys; include them as best-effort
    # total_gpus already mapped above; ensure numeric types remain numbers
    for k, v in list(out.items()):
        # try to convert numeric strings to numbers (but keep as-is if not numeric)
        if isinstance(v, str):
            try:
                if v == "":
                    continue
                if re.match(r"^-?\d+\.?\d*(e-?\d+)?$", v, re.IGNORECASE):
                    if "." in v or "e" in v.lower():
                        out[k] = float(v)
                    else:
                        out[k] = int(v)
            except Exception:
                pass

    return out


# -------------------------
# Main traversal & CSV write
# -------------------------
import statistics
from collections import defaultdict


def read_and_average_all_json_summaries(launch_dir: Path):
    """
    Read all 'training_summary_*.json' files in a given launch directory.
    Average numeric 'Avg.*' metrics across GPUs, take max for 'Peak.*' metrics,
    and combine everything into one summary dict.
    """
    json_files = list(launch_dir.glob("output/training_summary_*.json"))
    if not json_files:
        return {}

    all_metrics = []
    for jf in json_files:
        metrics = read_json_metrics(jf)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        return {}

    combined = {}
    numeric_fields = defaultdict(list)
    text_fields = {}

    # classify which metrics should use max instead of mean
    MAX_FIELDS = {
        "Peak Power Usage (W)",
        "Peak GPU Memory Usage (GB)",
        "Peak GPU Utilization",
    }

    # Collect numeric values
    for m in all_metrics:
        for k, v in m.items():
            if v in ("", None):
                continue
            if isinstance(v, (int, float)):
                numeric_fields[k].append(v)
            elif isinstance(v, str):
                # Keep the first non-empty textual value
                if k not in text_fields or not text_fields[k]:
                    text_fields[k] = v

    # Compute mean for average metrics and max for peak metrics
    for k, vals in numeric_fields.items():
        if not vals:
            continue
        if k in MAX_FIELDS:
            combined[k] = round(max(vals), 6)
        else:
            combined[k] = round(statistics.mean(vals), 6)

    # Add text fields
    combined.update(text_fields)

    return combined


def main():
    base_results = Path(BASE_DIR_RESULTS)
    if not base_results.exists():
        print(f"[ERROR] results directory not found at: {BASE_DIR_RESULTS}")
        return

    # 1️⃣ Group all launch-* directories by their parent (config) folder
    config_groups = defaultdict(list)
    for launch_dir in base_results.rglob("launch-*"):
        if launch_dir.is_dir():
            config_groups[launch_dir.parent].append(launch_dir)

    if not config_groups:
        # fallback in case you only have JSON files without launch-* folders
        for p in base_results.rglob(SUMMARY_FILENAME):
            config_groups[p.parent].append(p.parent)

    rows = []

    # 2️⃣ Process each configuration folder once
    for config_dir, launch_dirs in sorted(config_groups.items()):
        meta = parse_run_path(config_dir)
        model_from_path = meta.get("Model", "")
        dataset_from_path = meta.get("Dataset", "")

        all_metrics = []
        # for launch_dir in launch_dirs:
        #     json_path = launch_dir / SUMMARY_FILENAME
        #     metrics = read_json_metrics(json_path)

        #     # Detect failed runs
        #     if is_key_metrics_empty(metrics):
        #         if check_cuda_oom_error(launch_dir):
        #             metrics["Comment"] = "CUDA OOM error detected"
        #         else:
        #             metrics["Comment"] = "Error during training"
        #     all_metrics.append(metrics)

        for launch_dir in launch_dirs:
            metrics = read_and_average_all_json_summaries(launch_dir)

            # Detect failed runs
            if is_key_metrics_empty(metrics):
                if check_cuda_oom_error(launch_dir):
                    metrics["Comment"] = "CUDA OOM error detected"
                else:
                    metrics["Comment"] = "Error during training"
            all_metrics.append(metrics)

        # 3️⃣ Aggregate results across all runs
        combined = {}
        numeric_fields = defaultdict(list)
        text_fields = {}

        # Fields that should *not* be averaged
        NON_AVERAGE_FIELDS = {
            "TypeParallelism",
            "Precision Type",
            "Batch Size",
            "Accumulation Gradients",
            "Max Length",
            "Comment",
        }

        # Collect metrics
        for m in all_metrics:
            for k, v in m.items():
                # Skip empty
                if v in ("", None):
                    continue

                if isinstance(v, (int, float)):
                    # Always numeric -> average later
                    numeric_fields[k].append(v)
                elif isinstance(v, str):
                    # Text field
                    if k in NON_AVERAGE_FIELDS:
                        if k not in text_fields or not text_fields[k]:
                            text_fields[k] = v
                    else:
                        if k not in text_fields or not text_fields[k]:
                            text_fields[k] = v

        # Average numeric fields
        for k, vals in numeric_fields.items():
            if vals:
                combined[k] = round(statistics.mean(vals), 6)

        # Add text fields
        combined.update(text_fields)

        # If any run failed, mark it
        comments = [m.get("Comment", "") for m in all_metrics]
        if any("oom" in c.lower() or "error" in c.lower() for c in comments):
            combined["Comment"] = "Contains failed runs (OOM/Error)"

        # 4️⃣ Build CSV row
        row = {c: "" for c in COLUMNS}
        row["Supercomputer"] = SUPCOMPUTER_NAME
        row["Partition"] = PARTITION_NAME
        row["Model"] = model_from_path
        row["Dataset"] = dataset_from_path
        row["Dataset/Model Type"] = determine_model_type(str(model_from_path))
        row["Framework"] = meta.get("Framework", "")
        row["TypeParallelism"] = meta.get("TypeParallelism", "")
        row["Number of Nodes"] = meta.get("Number of Nodes", "")
        row["GPUs per Node"] = meta.get("GPUs per Node", "")
        row["Total GPUs used"] = meta.get("Total GPUs used", "")
        row["Precision Type"] = meta.get("Precision Type", "")
        row["Batch Size"] = meta.get("Batch Size", "")
        row["Accumulation Gradients"] = meta.get("Accumulation Gradients", "")
        row["Max Length"] = meta.get("Max Length", "")

        # Copy over numeric & averaged fields
        for col in [
            "Number of Trainable Parameters",
            "Learning Rate",
            "Dropout",
            "Avg. Power Usage (W)",
            "Peak Power Usage (W)",
            "Avg. GPU Memory Usage (GB)",
            "Peak GPU Memory Usage (GB)",
            "Avg. GPU Utilization",
            "Peak GPU Utilization",
            "Training Time per Epoch (sec)",
            "Training Time per Step (sec)",
            "Total Execution Time (hours)",
            "Training Throughput (tokens/sec)",
            "Avg. Training Loss",
            "Avg. Validation Loss",
            "Comment",
        ]:
            if col in combined:
                row[col] = combined[col]

        rows.append(row)

    # 5️⃣ Write to CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            out = {c: r.get(c, "") for c in COLUMNS}
            writer.writerow(out)

    print(f"✅ Training summary CSV written to: {OUTPUT_FILE}")
    print(f"📊 Config folders processed: {len(rows)}")


if __name__ == "__main__":
    main()
