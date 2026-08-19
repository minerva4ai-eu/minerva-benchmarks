# benchmark/cli.py
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
import scripts.slurm.monitor as m
import scripts.slurm.utils as u
from configs_hydra.hydra_app import generate_valid_combos
from omegaconf import DictConfig
from scripts.slurm.cli_utils import *
from scripts.slurm.submitter import build_launch_folder, submit_job

if TYPE_CHECKING:
    from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


@click.group()
def cli():
    """MINERVA SLURM job submission CLI for LLM training and fine-tuning benchmarks."""


@cli.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate configs and build launch folders without submitting jobs.",
)
@click.option(
    "--per-model-jobs",
    is_flag=True,
    help="Chain job dependencies per model (each model's jobs complete before the next starts). Cannot be combined with --dry-run.",
)
@click.option(
    "--configs-path",
    default=DEFAULT_CONFIGS_PATH,
    help="Path to the Hydra config directory (default: ./configs_hydra/configs).",
)
@click.option(
    "--config-name",
    default=DEFAULT_CONFIG_NAME,
    help="Base config name to compose (e.g., 'base', 'base-MN5').",
    required=True,
)
@click.option(
    "--runs-dir",
    default=RUNS_DIR,
    help="Output directory for generated configs and results (default: benchmark-runs/).",
)
@click.option(
    "--yaml",
    "yamls",
    multiple=True,
    default=None,
    help=(
        "Run a benchmark configuration by providing the path to a BenchmarkConfig YAML file. "
        "Can be repeated for multiple configs, e.g. '--yaml path1.yaml --yaml path2.yaml. "
        "First run a '--dry-run' to compose YAML configuration files and then use their paths to run them individually."
    ),
)
def run(dry_run, per_model_jobs, configs_path, config_name, runs_dir, yamls):
    """Generate benchmark configurations and submit SLURM jobs.

    Composes valid config combinations from Hydra configs or runs specific
    YAML files. Supports dry-run mode for config generation without submission
    and job dependency chaining per model.
    """
    print("\n")
    print(
        f"{u.POINT_DIAMOND} {u.CYAN} Running {u.MAGENTA} MINERVA Benchmarks {u.CYAN} for LLMs training and fine-tuning {u.POINT_DIAMOND} {u.RESET}"
    )

    if config_name != DEFAULT_CONFIG_NAME:
        runs_dir = f"{runs_dir}-{config_name}"

    valid = []
    if yamls:
        if isinstance(yamls, tuple) and len(yamls) == 1:
            yamls = [y.strip() for y in yamls[0].split("--yaml") if y != ""]
        for y in yamls:
            click.echo(f"\t{u.POINT_SQUARE} {u.YELLOW}Searching for {y}{u.RESET}")

            try:
                _cfg: BenchmarkConfig = DictConfig(u.load_yaml(y))
                runs_dir = Path(_cfg.experiment.output_dir)
                valid.append(_cfg)
                click.echo(f"\t{u.SUCCESS_HEAVY} {u.GREEN}YAML config FOUND!{u.RESET}")
            except FileNotFoundError:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED}YAML config could NOT be FOUND!{u.RESET}"
                )
                continue
            except Exception as e:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED} Exception occured while trying to read file...{u.RESET}"
                )
                raise e
    else:
        if config_name == DEFAULT_CONFIG_NAME:
            click.echo(
                f"\t{u.FAILURE_HEAVY} {u.RED}!WARNING! Argument '--config-name' is defaulting to '{DEFAULT_CONFIG_NAME}'...{u.RESET}"
            )
            exit(1)
        valid, _ = generate_valid_combos(
            config_path=configs_path, config_name=config_name, outpath=runs_dir
        )

    run_date = datetime.now().date().strftime("%d-%m-%Y")
    slurm_monitor_dir = os.path.join(runs_dir, "slurm-monitor")
    date_monitor_dir = os.path.join(slurm_monitor_dir, run_date)
    # os.makedirs(date_monitor_dir, exist_ok=True)
    run_id = 1
    short_id = f"run_id-{run_id}"
    run_monitor_dir = os.path.join(date_monitor_dir, short_id)
    while os.path.exists(run_monitor_dir):
        run_id += 1
        short_id = f"run_id-{run_id}"
        run_monitor_dir = os.path.join(date_monitor_dir, short_id)

    dependency_jobid = ""
    if dry_run:
        click.echo(f"\nSlurm monitor ID: {run_date} | dry-run")
        click.echo(f"\nDry running {len(valid)} experiment configuration...")
    else:
        click.echo(f"\nSlurm monitor ID: {run_date} | {short_id}")
        click.echo(f"\nSubmitting {len(valid)} jobs...")
    jobs_submitted = []
    cfgs_seen = set()
    # Submit job with depencies per model
    if per_model_jobs:
        if dry_run:
            exit(0)
        # Group job dependencies per model
        valid_models = set([c.model.name for c in valid])
        cfgs_per_model = {}
        for m in valid_models:
            if m not in cfgs_per_model:
                cfgs_per_model[m] = []
            cfgs_per_model[m].extend([cfg for cfg in valid if cfg.model.name == m])
        for model, model_cfgs in cfgs_per_model.items():
            dependency_jobid = ""
            for cfg in model_cfgs:
                if cfg.id in cfgs_seen:
                    print(
                        f"{u.YELLOW}Config id '{cfg.id} has been seen already, skipping duplicate job sbmission...'{u.RESET}"
                    )
                    continue
                for repeat_id in range(1, cfg.experiment.repeat + 1):
                    job_desc = {
                        "id": None,
                        "cfg_id": None,
                        "dependency": dependency_jobid,
                        "launch_folder": "",
                        "yaml_filename": "",
                    }
                    launch_folder = build_launch_folder(
                        cfg,
                        base_dir=BASE_DIR,
                        runs_dir=runs_dir,
                        run_id=short_id,
                        repeat_id=repeat_id,
                    )
                    dependency_jobid = submit_job(
                        cfg=cfg,
                        launch_folder=launch_folder,
                        repeat_id=repeat_id,
                        dependency=dependency_jobid,
                    )
                    job_desc["id"] = dependency_jobid
                    job_desc["cfg_id"] = cfg.id
                    job_desc["launch_folder"] = str(launch_folder.absolute())
                    job_desc["yaml_filename"] = cfg.experiment.yaml_filename
                    jobs_submitted.append(job_desc)
                    cfgs_seen.add(cfg.id)
        os.makedirs(run_monitor_dir, exist_ok=True)

        u.write_jsonl(
            d=jobs_submitted,
            p=os.path.join(run_monitor_dir, "jobs_submitted.jsonl"),
        )
        return

    # Sumbit all jobs interedependent
    for cfg in valid:
        if dry_run:
            _ = build_launch_folder(
                cfg, base_dir=BASE_DIR, runs_dir=runs_dir, run_id=short_id, dry=dry_run
            )
            click.echo(f"  {u.YELLOW}[dry]{u.RESET} {cfg.id}")
            continue

        if cfg.id in cfgs_seen:
            print(
                f"{u.YELLOW}Config id '{cfg.id} has been seen already, skipping duplicate job sbmission...'{u.RESET}"
            )
            continue
        for repeat_id in range(1, cfg.experiment.repeat + 1):
            job_desc = {
                "id": None,
                "cfg_id": None,
                "dependency": dependency_jobid,
                "launch_folder": "",
                "yaml_filename": "",
            }

            launch_folder = build_launch_folder(
                cfg,
                base_dir=BASE_DIR,
                runs_dir=runs_dir,
                run_id=short_id,
                repeat_id=repeat_id,
            )
            dependency_jobid = submit_job(
                cfg=cfg,
                launch_folder=launch_folder,
                repeat_id=repeat_id,
                dependency=dependency_jobid,
            )
            job_desc["id"] = dependency_jobid
            job_desc["cfg_id"] = cfg.id
            job_desc["launch_folder"] = str(launch_folder.absolute())
            job_desc["yaml_filename"] = cfg.experiment.yaml_filename
            jobs_submitted.append(job_desc)
            cfgs_seen.add(cfg.id)
        os.makedirs(run_monitor_dir, exist_ok=True)

        u.write_jsonl(
            d=jobs_submitted,
            p=os.path.join(run_monitor_dir, "jobs_submitted.jsonl"),
        )
    return


@cli.command()
@click.option(
    "--run-date",
    "run_date",
    type=str,
    help="Date of run in format '%d-%m-%Y'! If not provided in the correct form, rerun will fail.",
    required=True,
)
@click.option(
    "--run-id",
    "run_id",
    type=int,
    help="Serial id of run on provided date! If not provided, command will fail.",
    required=True,
)
@click.option(
    "--runs-dir",
    default=RUNS_DIR,
    help="Output directory for benchmark results (default: benchmark-runs/).",
)
@click.option(
    "--all",
    "all",
    is_flag=True,
    default=False,
    help="Rerun all jobs from the specified run.",
)
@click.option(
    "--only-failed",
    "only_failed",
    is_flag=True,
    default=False,
    help="Rerun only failed jobs from the specified run.",
)
@click.option(
    "--only-pending",
    "only_pending",
    is_flag=True,
    default=False,
    help="Rerun only pending jobs from the specified run.",
)
@click.option(
    "--yaml",
    "yamls",
    multiple=True,
    default=None,
    help=(
        "Re-run a benchmark configuration by providing the path to a BenchmarkConfig YAML file. "
        "Can be repeated for multiple configs. "
        "Rerun reuses scripts from the original run-id without copying them again. "
        "If you wish to apply or try changes made on the original scripts, use the 'run' subcommand instead."
    ),
)
def rerun(run_date, run_id, runs_dir, all, only_failed, only_pending, yamls):
    """Re-submit jobs from a previous benchmark run.

    Reuses scripts from the original run without re-copying them. Can rerun
    all jobs, only failed jobs, only pending jobs, or specific YAML configs.
    Requires --run-date and --run-id to identify the original run.
    """

    print("\n")
    print(
        f"{u.POINT_DIAMOND} {u.CYAN} Re-running {u.MAGENTA} MINERVA Benchmarks {u.CYAN} for LLMs training and fine-tuning {u.POINT_DIAMOND} {u.RESET}"
    )

    run_date = str2date2str(run_date)
    if not is_valid_date(run_date):
        raise ValueError(
            f"{u.RED}Provided invalid '--run-date' value '{run_date}'.Either faulty format or future timestamp!"
        )
    run_monitor_folder = f"{runs_dir}/slurm-monitor/{run_date}/run_id-{run_id}"
    run_monitor_path = f"{run_monitor_folder}/jobs_submitted.jsonl"

    rerun_id = 1
    rerun_logs = f"jobs_resubmitted-rerun_id-{rerun_id}.jsonl"
    rerun_monitor_path = os.path.join(run_monitor_folder, rerun_logs)
    while os.path.exists(rerun_monitor_path):
        rerun_id += 1
        rerun_logs = f"jobs_resubmitted-rerun_id-{rerun_id}.jsonl"
        rerun_monitor_path = os.path.join(run_monitor_folder, rerun_logs)

    rerun_id = f"run_id-{run_id}--rerun_id-{rerun_id}"

    run_jobs = m.load_all(run_monitor_path)

    mode = "all"
    if not yamls:
        assert sum([all, only_failed, only_pending]) == 1, (
            f"{u.RED}Only one of '--all', '--only-failed' or '--only-pending' must be provided!{u.RESET}"
        )
    cfgs_to_rerun = []
    if yamls:
        mode = "yamls"

        assert isinstance(yamls, tuple) and len(yamls) == 1, (
            f"\t{u.FAILURE_HEAVY} {u.RED}No YAML benchmark configurations provided to rerun!{u.RESET}"
        )

        yamls = [y.strip() for y in yamls[0].split("--yaml") if y != ""]
        try:
            cfgs_to_rerun.extend(
                [
                    (
                        DictConfig(u.load_yaml(y)),
                        job["launch_folder"],
                        job["launch_folder"].replace(f"run_id-{run_id}", rerun_id),
                    )
                    for y in yamls
                    for job in run_jobs
                    if job["yaml_filename"] in y
                ]
            )
        except FileNotFoundError:
            click.echo(
                f"\t{u.FAILURE_HEAVY} {u.RED}YAML config could NOT be FOUND!{u.RESET}"
            )

        except Exception as e:
            click.echo(
                f"\t{u.FAILURE_HEAVY} {u.RED} Exception occured while trying to read file...{u.RESET}"
            )
            raise e

        assert len(yamls) == len(cfgs_to_rerun), (
            f"\t{u.FAILURE_HEAVY} {u.RED}Number of YAMLs to rerun is not equal to the number of registered jobs found in '{run_monitor_path}'{u.RESET}"
        )
    else:
        if only_failed:
            mode = "only_failed"
            slurm_mode = [
                "node_fail",
                "out_of_memory",
                "cancelled",
                "timeout",
                "failed",
                "stopped",
                "suspended",
            ]
            run_jobs = [
                rj
                for rj in sorted(run_jobs, key=lambda j: j["id"])
                if m.get_job_info(rj["id"]).status_meta["code_complete"] in slurm_mode
            ]

        if only_pending:
            mode = "only_pending"
            slurm_mode = [
                "pending",
            ]
            run_jobs = [
                rj
                for rj in sorted(run_jobs, key=lambda j: j["id"])
                if m.get_job_info(rj["id"]).status_meta["code_complete"] in slurm_mode
            ]

        cfgs_to_rerun = []
        for job in run_jobs:
            config_dir = u.get_cfg_folder_from_launch(job["launch_folder"])
            try:
                cfg: BenchmarkConfig = DictConfig(
                    u.load_yaml(os.path.join(config_dir, job["yaml_filename"]))
                )
                click.echo(
                    f"\t{u.SUCCESS_HEAVY} {u.GREEN}YAML config '{job['yaml_filename']}' FOUND in [date|runid]: [{run_date}|run_id-{run_id}]{u.RESET}"
                )
            except FileNotFoundError:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED}YAML config '{job['yaml_filename']}' could NOT be FOUND!{u.RESET}"
                )
                return
            except Exception:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED} Exception occured while trying to read YAML '{job['yaml_filename']}'{u.RESET}"
                )
                return
            cfgs_to_rerun.append(
                (
                    cfg,
                    job["launch_folder"],
                    job["launch_folder"].replace(f"run_id-{run_id}", rerun_id),
                )
            )
    click.echo(
        f"\n{u.YELLOW}Resubmitting {len(cfgs_to_rerun)} jobs (mode={mode}) | {rerun_id}...{u.RESET}"
    )

    cfgs_seen = set()
    jobs_resubmitted = []
    dependency_jobid = ""
    for cfg, old_launchf, new_launchf in cfgs_to_rerun:
        u.copy_launch_folder(old_launchf, new_launchf)
        for repeat_id in range(1, cfg.experiment.repeat + 1):
            job_desc = {
                "id": None,
                "cfg_id": None,
                "dependency": dependency_jobid,
                "launch_folder": "",
                "yaml_filename": "",
            }
            if cfg.id in cfgs_seen:
                print(
                    f"{u.YELLOW}Config id '{cfg.id} has been seen already, skipping duplicate job sbmission...'{u.RESET}"
                )
                continue
            dependency_jobid = submit_job(
                cfg=cfg,
                launch_folder=Path(new_launchf),
                repeat_id=repeat_id,
                dependency=dependency_jobid,
            )
            job_desc["id"] = dependency_jobid
            job_desc["cfg_id"] = cfg.id
            job_desc["launch_folder"] = new_launchf
            job_desc["yaml_filename"] = cfg.experiment.yaml_filename
            jobs_resubmitted.append(job_desc)
            cfgs_seen.add(cfg.id)

        u.write_jsonl(d=jobs_resubmitted, p=rerun_monitor_path)


def _parse_space_separated(value: str | None) -> set | None:
    """Parse a space-separated string into a set of values, or return None."""
    if not value:
        return None
    return set(value.split())


def _filter_jobs_by_config(
    run_jobs: list[dict],
    runs_dir: str,
    run_date: str,
    run_id: str,
    rerun_id: int | None,
    model_names: set | None = None,
    framework_names: set | None = None,
    parallelism_names: set | None = None,
    nodes_values: set | None = None,
) -> list[dict]:
    """
    Filter jobs by loading their YAML configs and matching against provided criteria.
    All provided filters use AND logic — a job must match ALL specified filters.
    Returns the filtered list of jobs.
    """
    filtered = []
    for job in run_jobs:
        config_dir = "/".join(job["launch_folder"].split("/")[:-2])
        yaml_path = os.path.join(config_dir, job["yaml_filename"])

        try:
            cfg: BenchmarkConfig = DictConfig(u.load_yaml(yaml_path))
        except Exception:
            # If we can't load the config, skip this job
            continue

        # Check each filter (AND logic)
        match = True

        if model_names is not None:
            if cfg.model.name not in model_names:
                match = False

        if match and framework_names is not None:
            if cfg.framework.name not in framework_names:
                match = False

        if match and parallelism_names is not None:
            if cfg.framework.parallelism_name not in parallelism_names:
                match = False

        if match and nodes_values is not None:
            if str(cfg.slurm.sbatch.nodes) not in nodes_values:
                match = False

        if match:
            filtered.append(job)

    return filtered


@cli.command()
@click.option(
    "--run-date",
    "run_date",
    type=str,
    help="Date of run in format '%d-%m-%Y'! If not provided in the correct form, rerun will fail.",
    required=True,
)
@click.option(
    "--run-id",
    "run_id",
    type=str,
    help="Serial ID of the run on the provided date.",
    required=True,
)
@click.option(
    "--rerun-id",
    "rerun_id",
    type=int,
    default=None,
    help="Check status of a specific rerun 'rerun-id' within the provided 'run_id'.",
)
@click.option(
    "--runs-dir",
    "runs_dir",
    default=RUNS_DIR,
    help="Output directory for benchmark results (default: benchmark-runs/).",
)
@click.option(
    "--model",
    "model",
    type=str,
    default=None,
    help="Filter by model name(s). Space-separated for multiple values, e.g. '--model llama3-7b mistral-7b'.",
)
@click.option(
    "--framework",
    "framework",
    type=str,
    default=None,
    help="Filter by framework name(s). Space-separated for multiple values, e.g. '--framework accelerate deepspeed'.",
)
@click.option(
    "--parallelism-type",
    "parallelism",
    type=str,
    default=None,
    help="Filter by parallelism type(s). Space-separated for multiple values, e.g. '--parallelism-type ddp fsdp'.",
)
@click.option(
    "--nodes",
    type=str,
    default=None,
    help="Filter by number of nodes (exact match). Space-separated for multiple values, e.g. '--nodes 4 8 16'.",
)
@click.option(
    "--state",
    "state",
    type=str,
    help="Filter by SLURM job state, e.g 'running', 'pending', 'failed', etc.",
)
def status(
    run_date, run_id, rerun_id, runs_dir, model, framework, parallelism, nodes, state
):
    """Display SLURM job status for a benchmark run.

    Shows job states (running, pending, failed, etc.) with optional filtering
    by model, framework, parallelism type, number of nodes, and SLURM state.
    Can also check status of specific reruns within a run.
    """
    is_valid_date(run_date)
    assert is_valid_date(run_date), (
        f"{u.RED}--run-date must be in the format DD-MM-YYYY or d-m-YYYY!{u.RESET}"
    )
    run_date = str2date2str(run_date)

    run_monitor_folder = (
        f"{runs_dir}/slurm-monitor/{run_date}/run_id-{run_id}/jobs_submitted.jsonl"
    )
    if rerun_id:
        run_monitor_folder = f"{runs_dir}/slurm-monitor/{run_date}/run_id-{run_id}/jobs_resubmitted-rerun_id-{rerun_id}.jsonl"
    try:
        run_jobs = m.load_all(run_monitor_folder)
    except FileNotFoundError:
        click.echo(
            f"{u.RED}Could not find any jobs submited @ {run_date} - run_id:{run_id}{u.RESET}"
        )
        exit(1)

    # Apply config-based filtering
    model_names = _parse_space_separated(model)
    framework_names = _parse_space_separated(framework)
    parallelism_names = _parse_space_separated(parallelism)
    nodes_values = _parse_space_separated(nodes)
    states = _parse_space_separated(state)
    if state in states:
        if state not in m.SLURM_STATUS_DASHBOARD.keys():
            print(
                f"{u.RED}Argument '--state' is not valid to filter benchmark jobs for requested run."
                + f"\n{u.YELLOW}Valid job states: {', '.join(list(m.SLURM_STATUS_DASHBOARD.keys()))}{u.RESET}"
            )

    if any([model_names, framework_names, parallelism_names, nodes_values]):
        run_jobs = _filter_jobs_by_config(
            run_jobs=run_jobs,
            runs_dir=runs_dir,
            run_date=run_date,
            run_id=run_id,
            rerun_id=rerun_id,
            model_names=model_names,
            framework_names=framework_names,
            parallelism_names=parallelism_names,
            nodes_values=nodes_values,
        )
        if not run_jobs:
            click.echo(
                f"{u.YELLOW}No jobs match the provided filter criteria.{u.RESET}"
            )
            return

    print(f"\nJob status for run {u.CYAN}{run_id}{u.RESET}:\n")
    s1 = " " * 20
    s2 = " " * 50
    s3 = " " * 49
    print(f"{u.YELLOW}JOBID | RUNID | DEPJOB")
    for job in sorted(run_jobs, key=lambda j: j["id"]):
        job_info = m.get_job_info(job["id"])
        if state and state != job_info.status_meta["code_complete"]:
            continue
        m.print_job_status(job, job_info)


@cli.command()
@click.option(
    "--run-date",
    "run_date",
    type=str,
    help="Date of run in format '%d-%m-%Y'.",
    required=True,
)
@click.option(
    "--run-id",
    "run_id",
    type=str,
    help="Serial id of run on provided date.",
    required=True,
)
@click.option(
    "--runs-dir",
    "runs_id",
    default=RUNS_DIR,
    help="Output directory for benchmark results (default: benchmark-runs/).",
)
@click.option(
    "--model",
    "model",
    type=str,
    default=None,
    help="Filter by model name(s). Space-separated for multiple values, e.g. '--model llama3-7b mistral-7b'.",
)
@click.option(
    "--framework",
    "framework",
    type=str,
    default=None,
    help="Filter by framework name(s). Space-separated for multiple values, e.g. '--framework vllm sglang'.",
)
@click.option(
    "--parallelism-type",
    "parallelism",
    type=str,
    default=None,
    help="Filter by parallelism type(s). Space-separated for multiple values, e.g. '--parallelism-type dp fsdp'.",
)
@click.option(
    "--nodes",
    type=str,
    default=None,
    help="Filter by number of nodes (exact match). Space-separated for multiple values, e.g. '--nodes 4 8 16'.",
)
def cancel(run_date, run_id, runs_id, model, framework, parallelism, nodes):
    """Cancel all running and pending SLURM jobs for a benchmark run.

    Cancels jobs matching the specified run date and ID, with optional
    filtering by model, framework, parallelism type, and number of nodes.
    Only affects running and pending jobs; completed/failed jobs are ignored.
    """

    print("\n")
    print(
        f"{u.POINT_DIAMOND} {u.CYAN} Cancelling {u.MAGENTA} MINERVA Benchmarks {u.CYAN} jobs {u.POINT_DIAMOND} {u.RESET}"
    )

    run_date = str2date2str(run_date)
    if not is_valid_date(run_date):
        raise ValueError(
            f"{u.RED}Provided invalid '--run-date' value '{run_date}'. Either faulty format or future timestamp!{u.RESET}"
        )

    run_monitor_folder = f"{runs_id}/slurm-monitor/{run_date}/run_id-{run_id}"
    run_monitor_path = os.path.join(run_monitor_folder, "jobs_submitted.jsonl")

    if not os.path.exists(run_monitor_path):
        click.echo(f"{u.RED}Could not find jobs file @ {run_monitor_path}{u.RESET}")
        exit(1)

    run_jobs = m.load_all(run_monitor_path)
    if not run_jobs:
        click.echo(
            f"{u.YELLOW}No jobs found for run {run_date} | run_id-{run_id}{u.RESET}"
        )
        return

    # Apply config-based filtering
    model_names = _parse_space_separated(model)
    framework_names = _parse_space_separated(framework)
    parallelism_names = _parse_space_separated(parallelism)
    nodes_values = _parse_space_separated(nodes)

    if any([model_names, framework_names, parallelism_names, nodes_values]):
        run_jobs = _filter_jobs_by_config(
            run_jobs=run_jobs,
            runs_dir=runs_id,
            run_date=run_date,
            run_id=run_id,
            rerun_id=None,
            model_names=model_names,
            framework_names=framework_names,
            parallelism_names=parallelism_names,
            nodes_values=nodes_values,
        )
        if not run_jobs:
            click.echo(
                f"{u.YELLOW}No jobs match the provided filter criteria.{u.RESET}"
            )
            return

    import subprocess

    cancelled = []
    skipped = []

    for job in sorted(run_jobs, key=lambda j: j["id"]):
        job_id = job["id"]
        cfg_id = job["cfg_id"]
        try:
            job_info = m.get_job_info(job_id)
            state = job_info.status_meta["code_complete"]
        except Exception:
            click.echo(
                f"{u.WARNING} Could not query job {job_id} ({cfg_id}), skipping.{u.RESET}"
            )
            skipped.append({"id": job_id, "cfg_id": cfg_id, "reason": "query_failed"})
            continue

        if state in ("running", "pending"):
            try:
                result = subprocess.run(
                    ["scancel", job_id],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    cancelled.append({"id": job_id, "cfg_id": cfg_id, "state": state})
                    click.echo(
                        f"{u.SUCCESS_HEAVY} {u.GREEN}Cancelled {job_id} ({cfg_id}) [{state}]{u.RESET}"
                    )
                else:
                    skipped.append(
                        {
                            "id": job_id,
                            "cfg_id": cfg_id,
                            "reason": result.stderr.strip(),
                        }
                    )
                    click.echo(
                        f"{u.FAILURE_HEAVY} {u.RED}Failed to cancel {job_id} ({cfg_id}): {result.stderr.strip()}{u.RESET}"
                    )
            except Exception as e:
                skipped.append({"id": job_id, "cfg_id": cfg_id, "reason": str(e)})
                click.echo(
                    f"{u.FAILURE_HEAVY} {u.RED}Exception cancelling {job_id} ({cfg_id}): {e}{u.RESET}"
                )
        else:
            skipped.append(
                {
                    "id": job_id,
                    "cfg_id": cfg_id,
                    "state": state,
                    "reason": "not_running_or_pending",
                }
            )
            click.echo(f"{u.INFO} {job_id} ({cfg_id}) is [{state}], skipping.{u.RESET}")

    click.echo(
        f"\n{u.POINT_DIAMOND} Summary: {u.GREEN}{len(cancelled)} cancelled{u.RESET}, {u.YELLOW}{len(skipped)} skipped{u.RESET}"
    )


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

INTERACTIVE_HELP = """
Interactive mode — type a command and press Enter:

  run          Submit new benchmark jobs (will prompt for arguments)
  rerun        Rerun failed/pending jobs (will prompt for arguments)
  status       Check status of a run (will prompt for arguments)
  cancel       Cancel jobs of running/pending jobs of a run (will prompt for arguments)
  help         Show this help
  quit/exit    Exit

You can also pass flags directly, e.g.:  run --dry-run
Or let the interactive prompts guide you:  run
"""


def interactive_loop(subcmd: str = ""):
    """Run an interactive REPL loop for the CLI commands."""
    click.echo(f"{u.YELLOW}MINERVA benchmarks CLI — interactive mode{u.RESET}")
    click.echo(
        f"{u.YELLOW}Type 'help' for available commands, 'quit' to exit.\n{u.RESET}"
    )

    while True:
        try:
            if subcmd == "":
                user_input = read_user_input()
            else:
                user_input = subcmd
                subcmd = ""
        except (EOFError, KeyboardInterrupt):
            click.echo("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            click.echo("Bye.")
            break

        if user_input.lower() == "help":
            click.echo(INTERACTIVE_HELP)
            continue

        # Parse the command and its arguments
        parts = user_input.split()
        if len(parts) > 1:
            click.echo(
                f"{u.RED}Unknown command: '{' '.join(parts)}'. Type 'help' for options.{u.RESET}\n"
            )
            continue
        cmd_name = parts[0].lower()

        if cmd_name not in subcommands_list(cli):
            click.echo(
                f"{u.RED}\tUnknown command: '{cmd_name}'! Type 'help' for options.{u.RESET}\n"
            )
            continue

        # Collect arguments interactively based on command
        extra_args = []
        try:
            if cmd_name == "run":
                click.echo("\n  -- run options --")
                extra_args.extend(prompt_options_interactive(RUN_OPTIONS))

            elif cmd_name == "rerun":
                click.echo("\n  -- re-run options --")
                extra_args.extend(prompt_options_interactive(RERUN_OPTIONS))

            elif cmd_name == "status":
                click.echo("\n  -- status options --")
                extra_args.extend(prompt_options_interactive(STATUS_OPTIONS))
            elif cmd_name == "cancel":
                click.echo("\n  -- cancel options --")
                extra_args.extend(prompt_options_interactive(CANCEL_OPTIONS))
        except EOFError:
            click.echo("\nCancelled.")
            continue

        # Build a Click context and invoke the command
        cmd_map = {"run": run, "rerun": rerun, "status": status, "cancel": cancel}
        _cmd_mao = command_tree(cli)
        cmd = cmd_map[cmd_name]
        try:
            # Use sys.argv temporarily so Click can parse the sub-command args
            old_argv = sys.argv
            sys.argv = ["argv0", *extra_args]
            try:
                cmd(standalone_mode=False)
            except Exception as e:
                click.echo(f"Error: {e}", err=True)
            finally:
                sys.argv = old_argv
        except Exception as e:
            click.echo(f"Error: {e}", err=True)

        click.echo()  # blank line between commands


def cli_entry():
    """Entry point — if no args given, enter interactive mode."""
    if len(sys.argv) == 1:
        interactive_loop()
    elif len(sys.argv) == 2:
        if sys.argv[1].strip() in ["help", "--help", "-h"]:
            sys.argv[1] = "--help"
            cli()
        interactive_loop(sys.argv[1])
    else:
        cli()


def command_tree(obj):

    if isinstance(obj, click.Group):
        return {name: value for name, value in obj.commands.items()}


def subcommands_list(obj) -> list[str]:
    cmd_tree = command_tree(obj)
    return list(cmd_tree.keys())


if __name__ == "__main__":
    cli_entry()
