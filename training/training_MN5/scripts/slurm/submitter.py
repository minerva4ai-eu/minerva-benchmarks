# benchmark/submitter.py
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from configs_hydra.dataclasses_hydra.arch import get_peak_flops
from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
from omegaconf import DictConfig
from scripts.slurm import utils as u
from omegaconf import OmegaConf
from typing import Optional

def build_launch_folder(
    cfg: BenchmarkConfig, base_dir: Path, runs_dir: Path, run_id: str, dry: Optional[bool] = False, repeat_id: Optional[int] = None, 
) -> Path:

    parameters_combo = f"{cfg.model.name}/{cfg.framework.name}/{cfg.dataset.name}/nodes-{cfg.slurm.sbatch.nodes}"
    results_dir = os.path.join(base_dir.absolute(), runs_dir)
    machine_results_base = os.path.join(results_dir, cfg.machine.name)
    date_folder = os.path.join(
        machine_results_base,
        datetime.now().strftime("%d-%m-%Y"),
    )
    combo_path = os.path.join(
        date_folder,
        parameters_combo,
    )
    if dry:
        experiment_config_path = os.path.join(combo_path, cfg.experiment.yaml_filename)
        os.makedirs(combo_path, exist_ok=True)
        OmegaConf.save(cfg, experiment_config_path)
        return Path(experiment_config_path)
    launch_folder = Path('')
    if repeat_id:
        launch_folder = Path(combo_path, f"launch-{repeat_id}")
        launch_folder.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"Argument 'repeat_id' must be provided! Instead got '{repeat_id}'")
    return launch_folder

def copy_scripts(cfg: BenchmarkConfig, dest: Path):
    # Copy folder with shared among frameworks
    shutil.copytree(cfg.framework.scripts.shared, dest / "shared", dirs_exist_ok=True)

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


def build_env(cfg: BenchmarkConfig, launch_folder: Path, run_id: int) -> dict:
    m, t, f, d, s = (
        cfg.model,
        cfg.model.training,
        cfg.framework,
        cfg.dataset,
        cfg.slurm,
    )
    assert isinstance(f.parallelism, DictConfig) and len(f.parallelism) == 1, (
        f"cfg.framework is expected to be of type DictConfig, but received {type(cfg.framework)} cfg.framework"
    )
    assert (t.steps is not None) != (t.epochs is not None), (
        "Must provide exactly only one of 'epochs' or 'step'! "
    )
    return {
        **os.environ,
        "MODULES": cfg.machine.modules,
        "SINGULARITY_CONTAINER": cfg.machine.singularity_container,
        "SINGULARITY_BINDS": " ".join(cfg.machine.singularity_binds),
        "SINGULARITY_ARGS": " ".join(cfg.machine.singularity_args),
        "NODES": str(s.sbatch.nodes),
        "GPUS_PER_NODE": str(s.sbatch.gpus_per_node),
        "GPU_NODE": str(s.sbatch.nodes * s.sbatch.gpus_per_node),
        "FRAMEWORK": f.name,
        "DATASET": d.name,
        "DATASET_PATH": d.path,
        "MODEL": m.name,
        "MODEL_PATH": m.path,
        "PARALLELISM": f.parallelism_name,
        "PRECISION": t.precision,
        "BATCH_SIZE": str(t.batch_size),
        "GRAD_ACCUM": str(t.grad_accum),
        "MAX_MODEL_LENGTH": str(d.max_seq_len),
        "LR": str(t.lr),
        "STEPS": str(t.steps) if t.steps else "-1",
        "EPOCHS": str(t.epochs) if t.epochs else "-1",
        "REPEAT_ID": str(run_id),
        "MACHINE": cfg.machine.name,
        "LAUNCH_FOLDER": launch_folder.absolute(),
        "TRAIN_SCRIPT": os.path.join(launch_folder.absolute(), f.scripts.finetune.split("/")[-1]),
        "ENVIRONMENT_FINETUNING": cfg.machine.python_environment,
        "ZERO_STAGE": f.parallelism_name if f.name == "deepspeed" else "",
        "GPU_PEAK_TFLOPS": str(
            get_peak_flops(cfg.arch.gpu, cfg.model.training.precision)
        ),
    }


def submit_job(
    cfg: BenchmarkConfig,
    launch_folder: Path,
    repeat_id: int,
    dependency: str,
) -> str:
    copy_scripts(cfg, launch_folder)

    m, d, f, s, t = (
        cfg.model,
        cfg.dataset,
        cfg.framework,
        cfg.slurm,
        cfg.model.training,
    )
    cmd = [
        "sbatch",
        "--parsable",
        f"--chdir={launch_folder}",
        f"--nodes={s.sbatch.nodes}",
        f"--gres=gpu:{s.sbatch.gpus_per_node}",
        f"--cpus-per-task={s.sbatch.cpus_per_gpu}",
        f"--tasks-per-node={s.sbatch.tasks_per_node}",
        f"--output={s.sbatch.output}",
        f"--error={s.sbatch.error}",
        f"--account={s.account}",
        f"--qos={s.qos}",
        f"--partition={s.partition}",
        *([f"--dependency={dependency}"] if dependency else []),
        *s.sbatch.extra_args,
        os.path.join(launch_folder, f.scripts.run.split("/")[-1]), # path to training script
        #str(launch_folder),
    ]
    # print("\n".join(cmd))
    job_cfg = (
        f"{m.name}-{f.name}-{cfg.dataset.name}"
        + f"-Par_{f.parallelism_name}"
        + f"-Nodes_{s.sbatch.nodes}-GPUs_{s.sbatch.gpus_per_node * s.sbatch.nodes}"
        + f"-Prec_{t.precision}"
        + f"-BS_{t.batch_size}"
        + f"-GAS_{t.grad_accum}"
        + f"-Len_{d.max_seq_len}"
    )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=build_env(cfg, launch_folder, repeat_id),
        )
        if result.returncode != 0:
            print(f"{u.RED} {u.FAILURE_HEAVY} No job_id assigned - {job_cfg} {u.RESET}")
            print(f"\t  {u.ARROW_RIGHT}{u.YELLOW} {result} {u.RESET}")
            return "-100"
        job_id = result.stdout.strip()
    except Exception as e:
        raise e

    # update_status(cfg, runs_dir, "running", job_id)
    print(f"{u.GREEN} {u.SUCCESS_HEAVY} {job_id} - {job_cfg} {u.RESET}")
    return job_id
