# benchmark/submitter.py
import os
import shutil
import subprocess
from pathlib import Path

from configs_hydra.dataclasses_hydra.arch import get_peak_flops
from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
from omegaconf import DictConfig
from scripts.slurm import utils as u


def build_launch_folder(cfg: BenchmarkConfig, base_dir: Path, run_id: int) -> Path:
    folder = Path(
        os.path.join(os.path.join(base_dir, cfg.slurm.sbatch.chdir), f"launch-{run_id}")
    )

    folder.mkdir(parents=True, exist_ok=True)
    return folder


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


def build_env(cfg: BenchmarkConfig, run_id: int) -> dict:
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
        "TRAIN_SCRIPT": cfg.framework.scripts.finetune.split("/")[-1],
        "ENVIRONMENT_FINETUNING": cfg.machine.python_environment,
        "ZERO_STAGE": cfg.framework.parallelism_name
        if cfg.framework.name == "deepspeed"
        else "",
        "GPU_PEAK_TFLOPS": str(
            get_peak_flops(cfg.arch.gpu, cfg.model.training.precision)
        ),
    }


def submit_job(
    cfg: BenchmarkConfig,
    base_dir: Path,
    run_id: int,
    dependency: str,
) -> str:
    launch = build_launch_folder(cfg, base_dir, run_id=run_id)
    copy_scripts(cfg, launch)

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
        f"--chdir={launch}",
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
        os.path.join(launch, f.scripts.run.split("/")[-1]),
        str(launch),
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
            cmd, capture_output=True, text=True, env=build_env(cfg, run_id)
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
