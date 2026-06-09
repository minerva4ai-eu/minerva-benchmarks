# benchmark/cli.py
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import click
import scripts.slurm.monitor as m
import scripts.slurm.utils as u
from configs_hydra.hydra_app import generate_valid_combos
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from scripts.slurm.submitter import build_launch_folder, submit_job

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

RUNS_DIR = Path("results/")
DEFAULT_CONFIGS_PATH = "./configs_hydra/configs"
DEFAULT_CONFIG_NAME = "base"
BASE_DIR = Path(".")

MINERVA_USER_FOLDER = os.path.join(os.path.expanduser("~/"), ".minerva-benchmarks")
os.makedirs(MINERVA_USER_FOLDER, exist_ok=True)
HISTORY_FILE_PATH = os.path.join(os.path.expanduser("~/"), ".history")
USER_CONFIG_PATH = os.path.join(MINERVA_USER_FOLDER, "benchmarks-config.json")

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


@cli.command()
@click.option("--dry-run", is_flag=True)
@click.option("--configs-path", default=DEFAULT_CONFIGS_PATH)
@click.option(
    "--config-name",
    default=DEFAULT_CONFIG_NAME,
)
@click.option(
    "--output",
    default=RUNS_DIR,
)
def run(dry_run, configs_path, config_name, output):
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
    os.makedirs(date_monitor_dir, exist_ok=True)
    run_id = 1
    short_id = f"run_id-{run_id}"
    run_monitor_dir = os.path.join(date_monitor_dir, short_id)
    while os.path.exists(run_monitor_dir):
        run_id += 1
        short_id = f"run_id-{run_id}"
        run_monitor_dir = os.path.join(date_monitor_dir, short_id)

    click.echo(f"\nSlurm monitor ID: {run_date} | {short_id}")
    click.echo(f"\nSubmitting {len(valid)} jobs...")
    dependency_jobid = ""
    jobs_submitted = []
    for cfg in valid:
        if dry_run:
            _ = build_launch_folder(
                cfg, base_dir=BASE_DIR, runs_dir=RUNS_DIR, run_id=short_id, dry=dry_run
            )
            click.echo(f"  [dry] {cfg.id}")
        else:
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
                    runs_dir=RUNS_DIR,
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
            os.makedirs(run_monitor_dir, exist_ok=True)
            print(f"jobs_submitted {jobs_submitted}")
            print(f"run_monitor_dir {run_monitor_dir}")
            u.write_jsonl(
                d=jobs_submitted,
                p=os.path.join(run_monitor_dir, "jobs_submitted.jsonl"),
            )


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
@click.option("--failed", "mode", flag_value="failed", default=True)
@click.option("--pending", "mode", flag_value="pending")
@click.option(
    "--id", "cfg_id", default=None, help="Rerun a specific benchmark configuration"
)
def rerun(run_date, run_id, mode, cfg_id):
    """Rerun failed or pending jobs, or a specific combo by id."""

    run_jobs = m.load_all(
        f"{RUNS_DIR}/slurm-monitor/{run_date}/run_id-{run_id}/jobs_submitted.jsonl"
    )

    click.echo(f"Resubmitting {len(combos)} jobs (status={mode})...")
    for combo in combos:
        submit_job(combo, BASE_DIR, RUNS_DIR)


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
def status(run_date, run_id, model, framework, parallelism):
    """Print a summary of all run statuses."""

    run_jobs = m.load_all(
        f"{RUNS_DIR}/slurm-monitor/{run_date}/run_id-{run_id}/jobs_submitted.jsonl"
    )

    print(f"\nJob status for run {u.CYAN}{run_id}{u.RESET}:\n")
    s1 = " " * 20
    s2 = " " * 40
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


# Create key bindings
bindings = KeyBindings()


# Define Ctrl+Enter (Ctrl+M) to submit input (Linux)
# Define Ctrl+P to submit input (macOS)
@bindings.add("c-m")
@bindings.add("c-p")
def _(event):
    event.app.exit(result=event.app.current_buffer.text)


# Prevent Enter from submitting (we just want to move to the next line)
# @bindings.add("enter")
# def _(event):
#    event.app.current_buffer.insert_text(
#        "\n"
#    )  # Inserts a newline instead of submitting


def read_user_input(
    history_path: str = HISTORY_FILE_PATH, input_text: str = "minerva-benchmarks > "
) -> str:
    style = Style.from_dict(
        {
            "prompt": "ansicyan bold",
            "input": "ansigreen",
        }
    )

    session = PromptSession(
        key_bindings=bindings,
        style=style,
        history=FileHistory(history_path),
        completer=PathCompleter(expanduser=True),
        complete_while_typing=True,
        multiline=False,
    )

    return session.prompt(f"{input_text}")


def prompt_arg(message, required=True) -> str:
    """Prompt user for an argument value."""
    while True:
        value = read_user_input(input_text=message)
        if value or not required:
            return value
        click.echo("  This argument is required. Please enter a value.")


@dataclass
class OptionConfig:
    name: str  # CLI flag, e.g. "--configs-path"
    prompt: str  # Text shown to user
    default: Optional[str] = None  # Default value (None = not added to args)
    required: bool = False  # Force non-empty input
    validator: Optional[Callable[[str], bool]] = None  # Return True if valid
    error_msg: str = "Invalid input."  # Shown when validator fails
    transform: Optional[Callable[[str], str]] = None  # Transform before storing


@dataclass
class BoolOptionConfig(OptionConfig):
    condition_is_true: Callable[[str], bool] = lambda x: False


RUN_OPTIONS = [
    OptionConfig(
        name="--configs-path",
        prompt="configs-path",
        default="./configs_hydra/configs",
        validator=lambda p: os.path.exists(os.path.abspath(p)),
        error_msg="--configs-path must exist and point to the root directory of all OmegaConf .yaml!",
    ),
    OptionConfig(
        name="--config-name",
        prompt="config-name",
        default="base",
    ),
    OptionConfig(
        name="--output",
        prompt="output",
        default="./results",
    ),
    BoolOptionConfig(
        name="--dry-run",
        prompt="dry-run? [y/N]",
        default="N",
        transform=lambda x: x.lower(),
        validator=lambda x: x in ("y", "n"),
        error_msg="--dry-run can only be y or n!",
        condition_is_true=lambda x: x in ("y", "yes", "si", "oui"),
    ),
]


def is_valid_date(value: str, fmt: str = "%d-%m-%Y") -> bool:
    try:
        datetime.strptime(value, fmt)
        return True
    except ValueError:
        return False


def str2date2str(value: str, fmt: str = "%d-%m-%Y") -> str:
    date = datetime.strptime(value, fmt).date()
    return date.strftime(fmt)


RERUN_OPTIONS = [
    OptionConfig(name="--run-date", prompt="run-date", validator=is_valid_date),
    OptionConfig(
        name="--run-id",
        prompt="run-id",
    ),
    OptionConfig(
        name="--cfg-id",
        prompt="yaml-cfg-id",
    ),
    BoolOptionConfig(
        name="--only-failed",
        prompt="only-failed? [y/N]",
        default="y",
        transform=lambda x: x.lower(),
        validator=lambda x: x in ("y", "n"),
        error_msg="--dry-run can only be y or n!",
        condition_is_true=lambda x: x in ("y", "yes", "si", "oui"),
    ),
]

STATUS_OPTIONS = [
    OptionConfig(
        name="--run-date",
        prompt="run-date",
        transform=str2date2str,
        validator=is_valid_date,
    ),
    OptionConfig(
        name="--run-id",
        prompt="run-id",
    ),
    OptionConfig(name="--nodes", prompt="nodes", validator=lambda x: int(x) >= 1),
    OptionConfig(name="--model", prompt="model", validator=lambda x: int(x) >= 1),
    OptionConfig(
        name="--framework", prompt="framework", validator=lambda x: int(x) >= 1
    ),
    OptionConfig(
        name="--parallelism-type",
        prompt="parallelism-type",
        validator=lambda x: int(x) >= 1,
    ),
]


def prompt_options_interactive(options: list[OptionConfig]) -> list[str]:
    """
    Iterate over a list of OptionConfig, prompt user for each,
    validate, and return aggregated CLI args.
    Ctrl+C goes back one step; invalid input re-prompts.
    """
    run_args = []
    values = {opt.name: opt.default for opt in options}  # Seed with defaults

    empty_inputs = {"", "\n"}

    i = 0
    no_default_hint = "no default"
    while i < len(options):
        opt = options[i]
        try:
            while True:
                default_hint = (
                    f"default='{opt.default}'"
                    if opt.default is not None
                    else no_default_hint
                )
                raw = read_user_input(input_text=f"    {opt.prompt} ({default_hint}): ")

                # Empty input: use default or re-prompt if required
                if raw in empty_inputs or raw == "":
                    if opt.required:
                        click.echo(f"{u.RED}  This field is required.{u.RESET}")
                        continue
                    value = opt.default  # Keep default (may be None)
                    click.echo(f"\tUsing default: {opt.default}")
                    break

                # Transform if needed (e.g. .lower())
                if opt.transform:
                    raw = opt.transform(raw)

                # Validate
                if opt.validator and not opt.validator(raw):
                    click.echo(f"  {opt.error_msg}")
                    continue

                value = raw
                if not default_hint == no_default_hint:
                    click.echo(f"\tUsing user input: {value}")
                break

            # Append to args only if we have a value
            if value is not None:
                values[opt.name] = value
                if isinstance(opt, BoolOptionConfig):
                    if opt.condition_is_true(value):
                        run_args.extend([opt.name])
                    print(run_args)
                    i += 1
                    continue
                run_args.extend([opt.name, value])
            i += 1

        except KeyboardInterrupt:
            click.echo("")
            if i == 0:
                click.echo("\tCan't go back further — re-prompting first option.")
                continue
            # Go back one step: remove the last added arg pair
            prev_opt = options[i - 1]
            if prev_opt.name in values and values[prev_opt.name] is not None:
                # Remove "--flag value" pair (2 elements) if it was added
                if len(run_args) >= 2 and run_args[-2] == prev_opt.name:
                    run_args.pop()
                    run_args.pop()
            click.echo("\tReturning to previous argument...")
            i -= 1

    return run_args


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
        print(sys.argv)
        interactive_loop(sys.argv[1])
    else:
        cli()


if __name__ == "__main__":
    cli_entry()
