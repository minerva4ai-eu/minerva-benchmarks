import json
import os
from datetime import datetime
from pathlib import Path
from shutil import copytree, ignore_patterns

import yaml
from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig

import logging

logger = logging.getLogger(__name__)

# ANSI Escape Codes for Terminal Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[90m"
RESET = "\033[0m"

# --- Standard Unicode Icons ---
SUCCESS = f"{GREEN}✓{RESET}"
SUCCESS_HEAVY = f"{GREEN}✔{RESET}"
FAILURE = f"{RED}✗{RESET}"
FAILURE_HEAVY = f"{RED}✘{RESET}"
WARNING = f"{YELLOW}⚠{RESET}"
INFO = f"{BLUE}ℹ{RESET}"
PROGRESS = f"{CYAN}⟳{RESET}"
SKIPPED = f"{GRAY}↷{RESET}"

# --- Modern Emoji Icons (No color codes needed) ---
EMOJI_SUCCESS = "✅"
EMOJI_FAILURE = "❌"
EMOJI_WARNING = "⚠️"
EMOJI_INFO = "ℹ️"
EMOJI_PROGRESS = "⏳"
EMOJI_SKIPPED = "⏩"

# --- Arrow Indicators (For directions, flows, and pointers) ---
ARROW_RIGHT = f"{CYAN}→{RESET}"  # Sleek right arrow
ARROW_LEFT = f"{CYAN}←{RESET}"  # Sleek left arrow
ARROW_SUCCESS = f"{GREEN}➔{RESET}"  # Heavy right arrow for progress/next step
ARROW_FAT_RIGHT = f"{BLUE}►{RESET}"  # Solid triangle pointer
ARROW_CHEVRON = f"{MAGENTA}»{RESET}"  # Double chevron (great for prompts)
ARROW_SUB_ITEM = f"{GRAY}└──{RESET}"  # Directory tree style branch

# --- Bullet Points & Custom Dots (For lists and sub-items) ---
POINT_BULLET = f"{BLUE}•{RESET}"  # Standard clean bullet
POINT_SUCCESS = f"{GREEN}⦿{RESET}"  # Bullseye dot for positive states
POINT_WARNING = f"{YELLOW}⚠️{RESET}"  # (Emoji) Attention grabber
POINT_DIAMOND = f"{MAGENTA}◆{RESET}"  # Diamond bullet for premium/special logs
POINT_SQUARE = f"{GRAY}▪{RESET}"  # Small square for sub-bullets
POINT_STAR = f"{YELLOW}★{RESET}"  # Star for highlighted/important items

# --- Step / Numeric Indicators (For ordered workflows) ---
STEP_1 = f"{CYAN}➀{RESET}"
STEP_2 = f"{CYAN}➁{RESET}"
STEP_3 = f"{CYAN}➂{RESET}"


def write_jsonl(d: list[dict], p: str):

    with open(p, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in d)


def read_jsonl(path: str) -> list[dict]:

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                records.append(json.loads(line))
    return records


def load_yaml(filepath):
    try:
        with open(filepath, "r") as file:
            # safe_load automatically converts YAML into native Python dicts/lists
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError as e:
        print(f"{__name__}.load_yaml() | Error: The file '{filepath}' was not found.")
        raise e
    except yaml.YAMLError as e:
        print(f"{__name__}.load_yaml() | Error parsing YAML file: '{filepath}'")
        raise e


def get_cfg_folder(
    cfg: BenchmarkConfig,
    base_dir: Path,
    runs_dir: Path,
):

    parameters_combo = f"{cfg.model.name}/{cfg.framework.name}/{cfg.framework.parallelism_name}/{cfg.dataset.name}/nodes-{cfg.slurm.sbatch.nodes}"
    results_dir = os.path.join(base_dir.absolute(), runs_dir)
    machine_results_base = os.path.join(results_dir, cfg.machine.name)
    date_folder = os.path.join(
        machine_results_base,
        datetime.now().strftime("%d-%m-%Y"),
    )
    cfg_path = os.path.join(
        date_folder,
        parameters_combo,
    )
    logger.debug("cfg_path = %s", cfg_path)
    return cfg_path


def get_cfg_folder_from_launch(launch_path: str):

    config_dir = "/".join(launch_path.split("/")[:-2])
    config_dir = os.path.join(config_dir, "yaml-configs")

    return config_dir


def copy_launch_folder(source: str, target: str):
    copytree(
        source,
        target,
        ignore=ignore_patterns(
            "*.err", "*.out", "__pycache__", "*/outputs/*", "*/profiler/*"
        ),
        dirs_exist_ok=True,
    )
