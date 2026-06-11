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

RUNS_DIR = Path("benchmark-runs/")
DEFAULT_CONFIGS_PATH = "./configs_hydra/configs"
DEFAULT_CONFIG_NAME = "base"
BASE_DIR = Path(".")

# TODO: Folder is not working, check how to implement
# MINERVA_USER_FOLDER = os.path.join(os.path.expanduser("~/"), ".minerva-benchmarks")
# os.makedirs(MINERVA_USER_FOLDER, exist_ok=True)
HISTORY_FILE_PATH = os.path.expanduser("~/.minerva-history")
USER_CONFIG_PATH = os.path.expanduser(".minerva-benchmarks-config.json")

# @dataclass
# class UserBenchmarkConfig:
#
#    runs_dir: str
#    configs_path: str = USER_CONFIG_PATH
#
#    def save_json(self,):
#        open(self.configs_path)


@click.group()
def cli():
    pass


# TODO: Add descriptions to options
@cli.command()
@click.option(
    "--dry-run",
    is_flag=True,
)
@click.option("--per-model-jobs", is_flag=True)
@click.option("--configs-path", default=DEFAULT_CONFIGS_PATH)
@click.option(
    "--config-name",
    default=DEFAULT_CONFIG_NAME,
)
@click.option(
    "--output",
    default=RUNS_DIR,
)
def run(dry_run, per_model_jobs, configs_path, config_name, output):
    # TODO: Optimize this function on code redundancy
    """Generate all valid configs and submit all pending jobs."""
    print("\n")
    print(
        f"{u.POINT_DIAMOND} {u.CYAN} Running {u.MAGENTA} MINERVA Benchmarks {u.CYAN} for LLMs training and fine-tuning {u.POINT_DIAMOND} {u.RESET}"
    )

    valid, skipped = generate_valid_combos(
        config_path=configs_path, config_name=config_name, outpath=output
    )

    run_date = datetime.now().date().strftime("%d-%m-%Y")
    slurm_monitor_dir = os.path.join(output, "slurm-monitor")
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
            if m not in cfgs_per_model.keys():
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
                        runs_dir=output,
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
                cfg, base_dir=BASE_DIR, runs_dir=output, run_id=short_id, dry=dry_run
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
                runs_dir=output,
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


class InvalidRerun(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


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
    "--output",
    default=RUNS_DIR,
)
@click.option("--all", "all", is_flag=True, default=False)
@click.option("--only-failed", "only_failed", is_flag=True, default=False)
@click.option("--only-pending", "only_pending", is_flag=True, default=False)
@click.option(
    "--cfg-id",
    "cfg_ids",
    multiple=True,
    default=None,
    help=(
        'Rerun a benchmark configuration by cfg_id. Multiple ids may be provided by repeating the input argument, e.g. "...--cfg-id cfgid1 --cfg-id cfgid2 --cfg-id cfgid3 etc..."'
    ),
)
def rerun(run_date, run_id, output, all, only_failed, only_pending, cfg_ids):
    """Rerun all, failed, pending jobs, or a specific run/combo by id."""

    print("\n")
    print(
        f"{u.POINT_DIAMOND} {u.CYAN} Re-running {u.MAGENTA} MINERVA Benchmarks {u.CYAN} for LLMs training and fine-tuning {u.POINT_DIAMOND} {u.RESET}"
    )

    run_monitor_folder = f"{RUNS_DIR}/slurm-monitor/{run_date}/run_id-{run_id}"
    run_monitor_path = f"{run_monitor_folder}/jobs_submitted.jsonl"

    rerun_id = 1
    rerun_logs = f"jobs_resubmitted-rerun_id-{rerun_id}.jsonl"
    rerun_monitor_path = os.path.join(run_monitor_folder, rerun_logs)
    while os.path.exists(rerun_monitor_path):
        rerun_id += 1
        rerun_logs = f"jobs_resubmitted-rerun_id-{rerun_id}.jsonl"
        rerun_monitor_path = os.path.join(run_monitor_folder, rerun_logs)

    run_jobs = m.load_all(run_monitor_path)
    _combos = run_jobs

    mode = "all"
    if not cfg_ids:
        assert sum([all, only_failed, only_pending]) == 1, (
            f"{u.RED}Only one of '--all', '--only-failed' or '--only-pending' must be provided!{u.RESET}"
        )
    if cfg_ids:
        mode = "cfg-ids"
        _combos = []
        for cfg_id in cfg_ids:
            try:
                cfgid_job = [rj for rj in run_jobs if rj["cfg_id"] == cfg_id][0]
                _combos.append(cfgid_job)
                click.echo(
                    f"\t{u.SUCCESS_HEAVY} {u.GREEN}YAML config '{cfg_id}' FOUND in [date|runid]: [{run_date}|run_id-{run_id}]{u.RESET}"
                )
            except IndexError:
                click.echo(
                    f"\t{u.FAILURE_HEAVY} {u.RED}YAML config '{cfg_id}' could NOT be FOUND among runs for [date|runid]: [{run_date}|run_id-{run_id}]{u.RESET}"
                )

    combos = _combos
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
        combos = [
            rj
            for rj in sorted(_combos, key=lambda j: j["id"])
            if m.get_job_info(rj["id"]).status_meta["code_complete"] in slurm_mode
        ]

    if only_pending:
        mode = "only_pending"
        slurm_mode = [
            "pending",
        ]
        combos = [
            rj
            for rj in sorted(_combos, key=lambda j: j["id"])
            if m.get_job_info(rj["id"]).status_meta["code_complete"] in slurm_mode
        ]

    click.echo(
        f"\n{u.YELLOW}Resubmitting {len(combos)} jobs (status={mode})...{u.RESET}"
    )
    cfgs_seen = set()
    jobs_resubmitted = []
    dependency_jobid = ""

    rerun_id = f"run_id-{run_id}--rerun_id-{rerun_id}"
    for job in combos:
        config_dir = "/".join(job["launch_folder"].split("/")[:-2])

        cfg: "BenchmarkConfig" = DictConfig(
            u.load_yaml(os.path.join(config_dir, job["yaml_filename"]))
        )

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
            rerun_launch_folder = build_launch_folder(
                cfg=cfg,
                base_dir=BASE_DIR,
                runs_dir=output,
                run_id=rerun_id,
                repeat_id=repeat_id,
            )
            dependency_jobid = submit_job(
                cfg=cfg,
                launch_folder=Path(rerun_launch_folder),
                repeat_id=repeat_id,
                dependency=dependency_jobid,
            )
            job_desc["id"] = dependency_jobid
            job_desc["cfg_id"] = cfg.id
            job_desc["launch_folder"] = job["launch_folder"]
            job_desc["yaml_filename"] = cfg.experiment.yaml_filename
            jobs_resubmitted.append(job_desc)
            cfgs_seen.add(cfg.id)

        u.write_jsonl(d=jobs_resubmitted, p=rerun_monitor_path)


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
    help="",
    required=True,
)
@click.option(
    "--rerun-id",
    "rerun_id",
    type=int,
    help="Get status of jobs of a rerun 'rerun-id' on provided 'run_id'.",
    required=False,
)
@click.option(
    "--model",
    "model",
    type=str,
    help="",
    required=False,
)
@click.option(
    "--framework",
    "framework",
    type=str,
    help="",
    required=False,
)
@click.option(
    "--parallelism-type",
    "parallelism",
    type=str,
    help="",
    required=False,
)
def status(run_date, run_id, rerun_id, model, framework, parallelism):
    """Print a summary of all run statuses."""
    is_valid_date(run_date)
    assert is_valid_date(run_date), (
        f"{u.RED}--run-date must be in the format DD-MM-YYYY or d-m-YYYY!{u.RESET}"
    )
    run_date = str2date2str(run_date)

    run_monitor_folder = (
        f"{RUNS_DIR}/slurm-monitor/{run_date}/run_id-{run_id}/jobs_submitted.jsonl"
    )
    if rerun_id:
        run_monitor_folder = f"{RUNS_DIR}/slurm-monitor/{run_date}/run_id-{run_id}/jobs_resubmitted-rerun_id-{rerun_id}.jsonl"
    try:
        run_jobs = m.load_all(run_monitor_folder)
    except FileNotFoundError:
        click.echo(
            f"{u.RED}Could not find any jobs submited @ {run_date} - run_id:{run_id}{u.RESET}"
        )
        exit(1)

    print(f"\nJob status for run {u.CYAN}{run_id}{u.RESET}:\n")
    s1 = " " * 20
    s2 = " " * 50
    s3 = " " * 49
    print(f"{s1} {u.YELLOW}JOBID{s2}RUNID{s3}DEPJOB")
    for job in sorted(run_jobs, key=lambda j: j["id"]):
        job_info = m.get_job_info(job["id"])

        m.print_job_status(job, job_info)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

INTERACTIVE_HELP = """
Interactive mode — type a command and press Enter:

  run          Submit new benchmark jobs (will prompt for arguments)
  rerun        Rerun failed/pending jobs (will prompt for arguments)
  status       Check status of a run (will prompt for --run-id)
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

        if cmd_name not in ("run", "rerun", "status"):
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
        except EOFError:
            click.echo("\nCancelled.")
            continue

        # Build a Click context and invoke the command
        cmd_map = {"run": run, "rerun": rerun, "status": status}
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
        if sys.argv[1].strip() in ["help", "--help"]:
            sys.argv[1] = "--help"
            cli()
        interactive_loop(sys.argv[1])
    else:
        cli()


if __name__ == "__main__":
    cli_entry()
