# benchmark/cli.py
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import click
import scripts.slurm.utils as u
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

RUNS_DIR = Path("benchmark-runs/")
DEFAULT_CONFIGS_PATH = "./configs_hydra/configs"
DEFAULT_CONFIG_NAME = "base"
BASE_DIR = Path(".")

# TODO: Folder is not working, check how to implement
# MINERVA_USER_FOLDER = os.path.join(os.path.expanduser("~/"), ".minerva-benchmarks")
# os.makedirs(MINERVA_USER_FOLDER, exist_ok=True)
HISTORY_FILE_PATH = os.path.expanduser("~/.minerva-history")
USER_CONFIG_PATH = os.path.expanduser(".minerva-benchmarks-config.json")

# Create key bindings
bindings = KeyBindings()


# Define Ctrl+Enter (Ctrl+M) to submit input (Linux)
# Define Ctrl+P to submit input (macOS)
# @bindings.add("c-m")
# @bindings.add("c-p")
# def _(event):
#     event.app.exit(result=event.app.current_buffer.text)
@bindings.add("c-m")
@bindings.add("c-p")
def _(event):
    event.current_buffer.validate_and_handle()


style = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "input": "ansigreen",
    }
)

session = PromptSession(
    key_bindings=bindings,
    style=style,
    history=FileHistory(HISTORY_FILE_PATH),
    completer=PathCompleter(expanduser=True),
    complete_while_typing=True,
    multiline=False,
)


# try:
#    with open(HISTORY_FILE_PATH, "a") as f:
#        pass
#    print(f"DEBUG: history file writable at {HISTORY_FILE_PATH}")
# except Exception as e:
#    print(f"DEBUG: history file FAILED: {e}")
# Prevent Enter from submitting (we just want to move to the next line)
# @bindings.add("enter")
# def _(event):
#    event.app.current_buffer.insert_text(
#        "\n"
#    )  # Inserts a newline instead of submitting


def read_user_input(
    history_path: str = HISTORY_FILE_PATH, input_text: str = "minerva-benchmarks > "
) -> str:
    return session.prompt(f"{input_text}")


# def prompt_arg(message, required=True) -> str:
#    """Prompt user for an argument value."""
#    while True:
#        value = read_user_input(input_text=message)
#        if value or not required:
#            return value
#        click.echo("  This argument is required. Please enter a value.")


@dataclass
class OptionConfig:
    name: str  # CLI flag, e.g. "--configs-path"
    prompt: str  # Text shown to user
    default: Optional[str] = None  # Default value (None = not added to args)
    required: bool = False  # Force non-empty input
    validator: Optional[Callable[[str], bool]] = None  # Return True if valid
    error_msg: str = "Invalid input."  # Shown when validator fails
    transform: Optional[Callable[[str], Any]] = None  # Transform before storing


@dataclass
class BoolOptionConfig(OptionConfig):
    condition_is_true: Callable[[str], bool] = lambda x: False


@dataclass
class CommaSeparatedOptionConfig(OptionConfig):
    parse: Callable[[str], list[str]] = lambda x: x.split(",")


@dataclass
class SpaceSeparatedOptionConfig(OptionConfig):
    parse: Callable[[str], list[str]] = lambda x: x.split(" ")


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
        name="--runs-dir",
        prompt="runs-dir",
        default=str(RUNS_DIR),
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
    BoolOptionConfig(
        name="--per-model-jobs",
        prompt="per-model-jobs? [y/N]",
        default="N",
        transform=lambda x: x.lower(),
        validator=lambda x: x in ("y", "n"),
        error_msg="--per-model-jobs can only be y or n!",
        condition_is_true=lambda x: x in ("y", "yes", "si", "oui"),
    ),
    OptionConfig(
        name="--yaml",
        prompt="yamls (',' comma separated)",
        validator=lambda x: "--yaml" in x,
        transform=lambda x: " ".join([f"--yaml {_x}" for _x in x.split(",")]),
    ),
]


def is_valid_date(value: str, fmt: str = "%d-%m-%Y") -> bool:

    try:
        date = datetime.strptime(value, fmt)
        if date.date() > datetime.now().date():
            return False
        return True
    except ValueError:
        return False


def str2date2str(value: str, fmt: str = "%d-%m-%Y") -> str:
    date = datetime.strptime(value, fmt).date()
    return date.strftime(fmt)


RERUN_OPTIONS = [
    OptionConfig(
        name="--runs-dir",
        prompt="runs-dir",
        default=str(RUNS_DIR),
    ),
    OptionConfig(
        name="--run-date",
        prompt="run-date",
        transform=str2date2str,
        validator=is_valid_date,
        error_msg="--run-date must be in the format DD-MM-YYYY or d-m-YYYY!Check date for errors in format & or future timestamps.",
        required=True,
    ),
    OptionConfig(
        name="--run-id",
        prompt="run-id",
    ),
    OptionConfig(
        name="--yaml",
        prompt="yamls (',' comma separated)",
        validator=lambda x: "--yaml" in x,
        transform=lambda x: " ".join([f"--yaml {_x}" for _x in x.split(",")]),
    ),
    BoolOptionConfig(
        name="--all",
        prompt="all? [y/N]",
        default="N",
        transform=lambda x: x.lower(),
        validator=lambda x: x in ("y", "n"),
        error_msg="--all can only be y or n!",
        condition_is_true=lambda x: x in ("y", "yes", "si", "oui"),
    ),
    BoolOptionConfig(
        name="--only-failed",
        prompt="only-failed? [y/N]",
        default="y",
        transform=lambda x: x.lower(),
        validator=lambda x: x in ("y", "n"),
        error_msg="--only-failed can only be y or n!",
        condition_is_true=lambda x: x in ("y", "yes", "si", "oui"),
    ),
    BoolOptionConfig(
        name="--only-pending",
        prompt="only-pending? [y/N]",
        default="y",
        transform=lambda x: x.lower(),
        validator=lambda x: x in ("y", "n"),
        error_msg="--only-pending can only be y or n!",
        condition_is_true=lambda x: x in ("y", "yes", "si", "oui"),
    ),
]

STATUS_OPTIONS = [
    OptionConfig(
        name="--runs-dir",
        prompt="runs-dir (root folder of benchmark runs)",
        default=str(RUNS_DIR),
        validator=lambda x: os.path.exists(x),
    ),
    OptionConfig(
        name="--run-date",
        prompt="run-date (date of benchmark runs: 01-01-2026)",
        transform=str2date2str,
        validator=is_valid_date,
    ),
    OptionConfig(
        name="--run-id",
        prompt="run-id (serial id number of run e.g. 1)",
        default=str(RUNS_DIR),
        # validator=lambda x: isinstance(x, int) and x > 0,
    ),
    OptionConfig(
        name="--model",
        prompt="model (space-separated, e.g. 'llama3-7b mistral-7b')",
    ),
    OptionConfig(
        name="--framework",
        prompt="framework (space-separated, e.g. 'vllm sglang')",
    ),
    OptionConfig(
        name="--parallelism-type",
        prompt="parallelism-type (space-separated, e.g. 'ddp fsdp')",
    ),
    OptionConfig(
        name="--nodes",
        prompt="nodes (space-separated exact match, e.g. '4 8 16')",
    ),
    OptionConfig(
        name="--state",
        prompt="state (space-separated SLURM states, e.g. 'running pending failed')",
    ),
]


CANCEL_OPTIONS = [
    OptionConfig(
        name="--runs-dir",
        prompt="runs-dir (root folder of benchmark runs)",
        validator=lambda x: os.path.exists(x),
        default=str(RUNS_DIR),
    ),
    OptionConfig(
        name="--run-date",
        prompt="run-date",
        transform=str2date2str,
        validator=is_valid_date,
        error_msg="--run-date must be in the format DD-MM-YYYY or d-m-YYYY! Check date for errors in format & or future timestamps.",
        required=True,
    ),
    OptionConfig(
        name="--run-id",
        prompt="run-id",
        required=True,
    ),
    OptionConfig(
        name="--model",
        prompt="model (space-separated, e.g. 'llama3-7b mistral-7b')",
    ),
    OptionConfig(
        name="--framework",
        prompt="framework (space-separated, e.g. 'vllm sglang')",
    ),
    OptionConfig(
        name="--parallelism-type",
        prompt="parallelism-type (space-separated, e.g. 'dp fsdp')",
    ),
    OptionConfig(
        name="--nodes",
        prompt="nodes (space-separated exact match, e.g. '4 8 16')",
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
