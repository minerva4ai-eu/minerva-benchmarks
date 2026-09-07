# benchmark/submitter.py
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig, MachineConfig
from omegaconf import OmegaConf
from scripts.slurm import utils as u
from scripts.slurm.cli_utils import *

logger = logging.getLogger(__name__)


def write_config(
    cfg: BenchmarkConfig,
    base_dir: str,
    runs_dir: str,
    dry: bool | None = False,
) -> str:
    cfg_path = u.get_cfg_folder(cfg, base_dir, runs_dir)
    # logger.debug("combo_path = %s", combo_path)
    experiment_config_dir = os.path.join(cfg_path, "yaml-configs")
    experiment_config_path = os.path.join(
        experiment_config_dir, cfg.experiment.yaml_filename
    )
    if not dry:
        os.makedirs(experiment_config_dir, exist_ok=True)
        OmegaConf.save(cfg, experiment_config_path)
    return experiment_config_path


def _build_launch_folder(
    cfg: BenchmarkConfig,
    base_dir: Path,
    runs_dir: Path,
    dry: bool | None = False,
    repeat_id: int | None = None,
    run_date: str = "DD-MM-YYYY",
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


def copy_scripts(cfg: BenchmarkConfig, dest: Path):

    shutil.copy(cfg.framework.scripts.run, dest)
    if hasattr(cfg.framework.scripts, "finetune"):
        shutil.copy(cfg.framework.scripts.finetune, dest)

    for src in list(cfg.framework.scripts.copy_files):
        src_path = Path(src)
        if src_path.is_dir():
            shutil.copytree(
                src_path, os.path.join(dest, src_path.name), dirs_exist_ok=True
            )
        else:
            shutil.copy(src_path, dest)


def build_srun_env():
    from omegaconf import OmegaConf

    yaml_path = os.environ["YAML_PATH"]
    temp_env = os.environ["TEMP_ENV_FILE"]

    cfg: BenchmarkConfig
    cfg = OmegaConf.load(yaml_path)

    env = {
        **(
            {"LOAD_MODULES": f"module load {' '.join(cfg.machine.modules)}"}
            if cfg.machine.modules is not None
            else {}
        ),
        "EXECUTION_MODE": cfg.machine.runtime_env_mode,
        **(
            {
                "VENV_PATH": cfg.framework.python_environment,
            }
            if cfg.framework.python_environment is not None
            else {}
        ),
        **(
            {
                "SINGULARITY_CONTAINER": cfg.framework.singularity_container,
                "SINGULARITY_BINDS": " ".join(cfg.machine.singularity_binds)
                if cfg.machine.singularity_binds
                else "",
                "SINGULARITY_ARGS": " ".join(cfg.machine.singularity_args)
                if cfg.machine.singularity_args
                else "",
            }
            if cfg.framework.singularity_container is not None
            else {}
        ),
        "SRUN_SCRIPT": cfg.framework.scripts.run,
        "TRAIN_SCRIPT": cfg.framework.scripts.finetune,
        "FRAMEWORK": cfg.framework.name,
        "PARALLELISM": cfg.framework.parallelism_name,
        "NNODES": cfg.slurm.sbatch.nodes,
        "TOKENIZERS_PARALLELISM": str(False),
        **(
            {
                "TP": str(cfg.framework.megatron_parallelism.tp),
                "PP": str(cfg.framework.megatron_parallelism.pp),
                "DP": str(cfg.framework.megatron_parallelism.dp),
                "CP": str(cfg.framework.megatron_parallelism.cp),
                "SP": str(cfg.framework.megatron_parallelism.cp),
                "EP": str(cfg.framework.megatron_parallelism.cp),
            }
            if cfg.framework.megatron_parallelism
            else {}
        ),
        "DATASET_PATH": cfg.dataset.path,
        **(
            {
                "DATASET_TRAIN": str(cfg.dataset.train),
            }
            if cfg.dataset.train is not None
            else {}
        ),
        **(
            {
                "DATASET_VALIDATION": str(cfg.dataset.validation),
            }
            if cfg.dataset.validation is not None
            else {}
        ),
    }

    if cfg.machine.env:
        env.update(**cfg.machine.env)
    if cfg.experiment.env:
        env.update(**cfg.experiment.env)
    if cfg.framework.env:
        env.update(**cfg.framework.env)

    with open(temp_env, "w") as f:
        f.writelines(f"export {var}='{value}'\n" for var, value in env.items())


def build_sbatch_env(
    machine: MachineConfig, yamls: list[str], results_dir: str
) -> dict:

    env = {**os.environ}
    # TODO: try-except
    env |= {
        "MODULES": " ".join(machine.modules) if machine.modules else "",
        "EXECUTION_MODE": machine.runtime_env_mode.value,
        "SINGULARITY_BINDS": "".join(machine.singularity_binds)
        if machine.singularity_binds
        else "",
        "SINGULARITY_ARGS": " ".join(machine.singularity_args)
        if machine.singularity_args
        else "",
    }

    env |= {"MINERVA_WORKDIR": results_dir, "YAMLS": ":".join(yamls)}

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


def get_job_nodes(cfgs: list[BenchmarkConfig]) -> int:
    max_nodes = -1
    for cfg in cfgs:
        max_nodes = max(max_nodes, cfg.slurm.sbatch.nodes)
    return max_nodes


def submit_job(
    cfgs: list[BenchmarkConfig],
    cfgs_paths: list[str],
    config_path: str,
    config_name: str,
    runs_dir: str,
    run_date: str,
) -> str:

    job_id = ""

    logger.debug("cfg_name = %s", config_name)
    logger.debug("config_path = %s", config_path)
    logger.debug("runs_dir = %s", runs_dir)
    logger.debug("run_date = %s", run_date)
    logger.debug("len(cfgs) = %s", len(cfgs))

    ################################################################################
    # Get system-level configs
    ################################################################################
    # Get nodes to run job on
    job_nodes = get_job_nodes(cfgs)

    # Get system slurm & machine configs
    # Sample only one of valid configurations, cause
    # properties used downstream from objects are fixed
    # for each profile (--config-name) and common among
    # all combinations
    s = cfgs[0].slurm
    m = cfgs[0].machine
    logger.debug("Max nodes = %s", job_nodes)
    # Get system env configs
    minerva_dir = os.path.join(runs_dir, m.name, run_date)
    sbatch_logs_dir = f"{minerva_dir}/%j/sbatch"
    logger.debug("sbatch_logs_dir = %s", sbatch_logs_dir)

    # Build sbatch command to submit
    cmd = [
        "sbatch",
        "--parsable",
        f"--nodes={job_nodes}",
        f"--gres={s.sbatch.gres}",
        f"--cpus-per-task={s.sbatch.cpus_per_task}",
        f"--tasks-per-node={s.sbatch.tasks_per_node}",
        # f"--output={sbatch_logs_dir}/run-%j.out",
        # f"--error={sbatch_logs_dir}/run-%j.err",
        f"--partition={s.partition}",
    ]
    # TODO: check desired behavior, joint condition?
    if s.qos is not None and s.account is not None:
        cmd.extend(
            [
                f"--account={s.account}",
                f"--qos={s.qos}",
            ]
        )

    if s.constraint is not None:
        cmd.extend([f"--constraint={s.constraint}"])

    cmd.extend(
        [
            *s.sbatch.extra_args,
            "MINERVA.job",
        ]
    )
    logger.debug("os.getcwd() = %s", os.getcwd())

    try:
        logger.info("cmd = %s", cmd)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=build_sbatch_env(machine=m, yamls=cfgs_paths, results_dir=minerva_dir),
        )
        if result.returncode != 0:
            click.echo(
                f"{u.RED} {u.FAILURE_HEAVY} No job_id assigned - {config_name} {u.RESET}"
            )
            click.echo(f"\t  {u.ARROW_RIGHT}{u.YELLOW} {result} {u.RESET}")
            return "-100"
        job_id = result.stdout.strip()
        # job_id = "0"
    except Exception as e:
        raise e

    click.echo(f"{u.GREEN} {u.SUCCESS_HEAVY} {job_id} - {config_name} {u.RESET}")
    return job_id


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    argsparser = ArgumentParser()
    argsparser.add_argument("--build-srun-env", action="store_true", default=False)

    args = argsparser.parse_args()
    if args.build_srun_env:
        try:
            build_srun_env()
        except Exception:
            sys.exit(1)
