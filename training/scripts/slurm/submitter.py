# benchmark/submitter.py
import json
import os
import shutil
import subprocess
from pathlib import Path
import yaml
from copy import deepcopy

from configs_hydra.dataclasses_hydra.arch import get_peak_flops
from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
from omegaconf import DictConfig, OmegaConf
from scripts.slurm import utils as u
from scripts.slurm.cli_utils import *

import logging

logger = logging.getLogger(__name__)

def build_launch_folder(
    cfg: BenchmarkConfig,
    base_dir: Path,
    runs_dir: Path,
    dry: bool | None = False,
    repeat_id: int | None = None,
    run_date: str = "DD-MM-YYYY"
) -> Path:
    combo_path = u.get_cfg_folder(cfg, base_dir, runs_dir)
    # logger.debug("combo_path = %s", combo_path)
    experiment_config_dir = os.path.join(combo_path, "yaml-configs")
    experiment_config_path = os.path.join(
        experiment_config_dir, cfg.experiment.yaml_filename
    )
    if dry:
        # OmegaConf.save(cfg, experiment_config_path)
        return Path(experiment_config_path)
    os.makedirs(combo_path, exist_ok=True)
    os.makedirs(experiment_config_dir, exist_ok=True)
    launch_folder = Path("")

    # TODO: check desired behavior
    # runs_dir = f"{runs_dir}-{config_name}"
    slurm_monitor_dir = os.path.join(runs_dir, "slurm-monitor")
    date_monitor_dir = os.path.join(slurm_monitor_dir, run_date)
    # os.makedirs(date_monitor_dir, exist_ok=True)
    run_id = 1
    short_id = f"run_id-{run_id}"
    run_monitor_dir = os.path.join(date_monitor_dir, short_id)
    # logger.debug("run_id, short_id = %s, %s", run_id, short_id)
    # FIXME: traceable
    while os.path.exists(run_monitor_dir):
        run_id += 1
        short_id = f"run_id-{run_id}"
        # logger.debug("run_id, short_id = %s, %s", run_id, short_id)
        run_monitor_dir = os.path.join(date_monitor_dir, short_id)
                   
    if repeat_id:
        run_folder = os.path.join(combo_path, short_id)
        launch_folder = Path(run_folder, f"launch-{repeat_id}")
        launch_folder.mkdir(parents=True, exist_ok=True)
        cfg.run_dir = str(launch_folder)
        if not os.path.exists(experiment_config_path):
            OmegaConf.save(cfg, experiment_config_path)

    else:
        raise ValueError(
            f"Argument 'repeat_id' must be provided! Instead got '{repeat_id}'"
        )
    return launch_folder

def build_env(envfig: dict) -> dict:

    env = {**os.environ}
    # TODO: try-except
    if envfig.get('machine'):
        env |= {
            "MODULES": ' '.join(envfig['machine'].get('modules')) if type(envfig['machine'].get('modules')) == type([]) else envfig['machine'].get('modules'),
            "EXECUTION_MODE": envfig['machine'].get('runtime_env_mode'),
            "SINGULARITY_BINDS": " ".join(envfig['machine'].get('singularity_binds')) if type(envfig['machine'].get('singularity_binds')) == type([]) else envfig['machine'].get('singularity_binds'),
            "SINGULARITY_ARGS": " ".join(envfig['machine'].get('singularity_args')) if type(envfig['machine'].get('singularity_args')) == type([]) else envfig['machine'].get('singularity_args')
        }

    env |= {
        "MINERVA_WORKDIR": envfig.get('minerva_workdir'),
    }

    # TODO: check
    def _serialize(value):
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value))
        return str(value)

    # Make sure that all values are serialized/cast to string
    for k, v in env.items():
        env[k] = _serialize(v)
    return env

def get_slurm_config(cfg_name: str, config_path: str):
    sbatch_config = {}
    # FIXME: work for GPP
    sysname = cfg_name.split("-")[0]
    # logger.debug("sysname = %s", sysname)
    try:
        with open(os.path.join(config_path, "slurm", f"{cfg_name.split('-')[0]}.yaml"), "r") as f:
            sbatch_config = yaml.safe_load(f)
        # logger.debug("sbatch_config = %s", sbatch_config)
    except FileNotFoundError:
        logger.exception("Slurm YAML config not found: cfg_name=%s, config_path=%s", cfg_name, config_path)
        print(
            f"\t{u.FAILURE_HEAVY} {u.RED}Slurm YAML config not found: cfg_name={cfg_name}, config_path={config_path}{u.RESET}"
        )
    except Exception as e:
        logger.exception("Exception occured while trying to read : cfg_name=%s, config_path=%s", cfg_name, config_path)
        print(
            f"\t{u.FAILURE_HEAVY} {u.RED} Exception occured while trying to read Slurm YAML config: cfg_name={cfg_name}, config_path={config_path}...{u.RESET}"
        )
        raise e
    return sbatch_config
   
def get_env_config(cfg_name: str, config_path: str, runs_dir: str, run_date: str):
    env_config = {}
    try:
        with open(os.path.join(config_path, f"{cfg_name}.yaml"), "r") as f:
            env_config = yaml.safe_load(f)
        # logger.debug("env_config = %s", env_config)
    except FileNotFoundError:
        logger.exception("system YAML config not found: cfg_name=%s, config_path=%s", cfg_name, config_path)
        print(
            f"\t{u.FAILURE_HEAVY} {u.RED}system YAML config not found: cfg_name={cfg_name}, config_path={config_path}{u.RESET}"
        )
    except Exception as e:
        logger.exception("Exception occured while trying to read : cfg_name=%s, config_path=%s", cfg_name, config_path)
        print(
            f"\t{u.FAILURE_HEAVY} {u.RED} Exception occured while trying to read system YAML config: cfg_name={cfg_name}, config_path={config_path}...{u.RESET}"
        )
        raise e
    # Get system-system
    try:
        with open(os.path.join(config_path, f"{env_config['defaults'][0]}.yaml"), "r") as f:
            envfig = yaml.safe_load(f)
        # logger.debug("envfig = %s", envfig)
        env_config['minerva_workdir'] = os.path.join(runs_dir, envfig['machine']['name'], run_date)
    except FileNotFoundError:
        logger.exception("system YAML config not found: cfg_name=%s, config_path=%s", cfg_name, config_path)
        print(
            f"\t{u.FAILURE_HEAVY} {u.RED}system YAML config not found: cfg_name={cfg_name}, config_path={config_path}{u.RESET}"
        )
    except Exception as e:
        logger.exception("Exception occured while trying to read : cfg_name=%s, config_path=%s", cfg_name, config_path)
        print(
            f"\t{u.FAILURE_HEAVY} {u.RED} Exception occured while trying to read system YAML config: cfg_name={cfg_name}, config_path={config_path}...{u.RESET}"
        )
        raise e
    return env_config
   
def submit_job(
    cfg_name: str,
    config_path: str,
    runs_dir: str,
    # run_dir: str,
    # cfgs: list,
    run_date: str
) -> str:

    job_id = ""

    # logger.debug("cfg_name = %s", cfg_name)
    # logger.debug("config_path = %s", config_path)
    # logger.debug("runs_dir = %s", runs_dir)
    # # logger.debug("run_dir = %s", run_dir)
    # logger.debug("run_date = %s", run_date)
    # logger.debug("len(cfgs) = %s", len(cfgs))


    ################################################################################
    # Get system-level configs
    ################################################################################
    # Get system slurm configs
    s = get_slurm_config(cfg_name, config_path)
    # logger.debug("s = %s", s)
    # logger.debug("s['sbatch']['nodes'] = %s", s['sbatch']['nodes'])
    # Get system env configs
    env_config = get_env_config(cfg_name, config_path, runs_dir, run_date)
    # logger.debug("env_config = %s", env_config)

    # Build sbatch command to submit
    cmd = [
        "sbatch",
        "--parsable",
        f"--nodes={s['sbatch']['nodes']}",
        f"--gres={s['sbatch']['gres']}",
        f"--cpus-per-task={s['sbatch']['cpus_per_task']}",
        f"--tasks-per-node={s['sbatch']['tasks_per_node']}",
        f"--output={env_config['minerva_workdir']}/run-%j.out", # TODO: get output from s
        f"--error={env_config['minerva_workdir']}/run-%j.err",
        f"--partition={s['partition']}",
    ]
    # TODO: check desired behavior, joint condition?
    if s.get('qos') is not None and s.get('account') is not None:
        cmd.extend(
            [
                f"--account={s['account']}",
                f"--qos={s['qos']}",
            ]
        )

    if s.get('constraint') is not None:
        cmd.extend([f"--constraint={s['constraint']}"])

    cmd.extend(
        [
            *s['sbatch']['extra_args'],
            "MINERVA.job",
        ]
    )
    # logger.debug("os.getcwd() = %s", os.getcwd())

    try:
        # logger.info("cmd = %s", cmd)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=build_env(env_config)
        )
        if result.returncode != 0:
            print(f"{u.RED} {u.FAILURE_HEAVY} No job_id assigned - {cfg_name} {u.RESET}")
            print(f"\t  {u.ARROW_RIGHT}{u.YELLOW} {result} {u.RESET}")
            return "-100"
        job_id = result.stdout.strip()
        # job_id = "0"
    except Exception as e:
        raise e

    # update_status(cfg, runs_dir, "running", job_id)
    print(f"{u.GREEN} {u.SUCCESS_HEAVY} {job_id} - {cfg_name} {u.RESET}")
    return job_id
