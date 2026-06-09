import os
from argparse import ArgumentParser
from copy import deepcopy
from itertools import product

from configs_hydra.constraints import rules
from configs_hydra.dataclasses_hydra import BenchmarkConfig, register_configs

# sweep/generator.py
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

# Color Codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"

# Style Codes
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
BG_RED = "\033[41m"

# Reset Code (turns colors back to normal)
RESET = "\033[0m"


##############################################################
# All configuration modules to be included are written below #
# Keep as is, to have a reference of complete foncigurations #
# to be used for all benchmarks generated.                   #
#                                                            #
# If needed to generate only specific or customized new      #
# benchmarks, copy and override values.                      #
#
# Example:
#
#   MODELS = ["llama3_8b", "mistral_7b", "llama3_70b"]
#   FRAMEWORKS = ["torchrun", "deepspeed"]
#   DATASETS = ["alpaca", "squadv2"]
#   ...
#   MODELS = ["new_model_config"]
#   FRAMEWORKS = ["torchrun", "deepspeed"]
#   DATASETS = ["alpaca", "squadv2", "new_dataset_config"]
MODELS = [
    "llama3_8b",
]
FRAMEWORKS = [
    "accelerate",
]
DATASETS = ["alpaca"]

################################################################


console = Console()


def expand_arch_gpu_configs(cfg: DictConfig) -> list[int]:
    """Replicates your GPU_CONFIGS=(1 $GPUS_PER_NODE) logic."""
    _p = list(cfg.framework.parallelism.keys())
    if len(_p) > 1:
        raise Exception(
            f"Fuction expand_arch_gpu_configs() received List instead of Dict for cfg.framework.parallelism: '{cfg.framework.parallelism}'"
        )
    p_name = _p[0]
    if (
        cfg.get("single_gpu_also_valid")
        and (cfg.slurm.sbatch.nodes == 1)
        and (cfg.framework.parallelism[p_name]["min_gpus"] == 1)
    ):
        return [1, cfg.slurm.sbatch.gpus_per_node]
    return [cfg.slurm.sbatch.gpus_per_node]


def generate_valid_combos(
    config_path: str, config_name: str, outpath: str
) -> tuple[list[BenchmarkConfig], list[rules.RuleResult]]:
    valid, skipped = [], []

    register_configs()
    GlobalHydra.instance().clear()

    # outpath = os.path.join(outpath, "bencharks_to_run")
    os.makedirs(outpath, exist_ok=True)

    with initialize_config_dir(
        config_dir=os.path.abspath(config_path), version_base="1.3"
    ):
        raw_total = 0
        for model, framework, dataset in product(MODELS, FRAMEWORKS, DATASETS):
            _init_cfg: BenchmarkConfig = compose(
                config_name,
            )

            cfg: BenchmarkConfig = compose(
                config_name,
                overrides=[
                    f"model={model}-{_init_cfg.machine.name_pattern}",
                    f"framework={framework}",
                    f"dataset={dataset}",
                ],
            )
            print(
                f"Composed base configuration from {os.path.abspath(f'{config_name}.yaml')} for:"
                + f"\n\t· model: {model}"
                + f"\n\t· framework: {framework}"
                + f"\n\t· dataset: {dataset}"
            )
            if framework not in cfg.model.frameworks_supported:
                continue

            for parallelism in cfg.model.parallelism_supported:
                tmp_cfg = deepcopy(cfg)

                if parallelism not in cfg.framework.parallelism.keys():
                    continue
                # print(f"cfg.framework.parallelism: \n{cfg.framework.parallelism}")
                # print(f"parallelism: \n{parallelism}")
                parallelism_spex = cfg.framework.parallelism[parallelism]

                target_parallelism = OmegaConf.create(
                    DictConfig({parallelism: parallelism_spex})
                )
                # print(f"target_parallelism: \n{target_parallelism}")
                # print(f"cfg.framework.parallelism: \n{cfg.framework.parallelism}")
                tmp_cfg.framework.parallelism_name = parallelism
                tmp_cfg.framework.parallelism = target_parallelism
                # print(f"\tCreating configuration for parallelism {parallelism}:")
                #  print(OmegaConf.to_yaml(cfg))

                for (
                    bs,
                    grad_acc,
                    precision,
                    lr,
                    optimizer,
                    steps,
                ) in product(
                    cfg.trainings.batch_sizes,
                    cfg.trainings.grad_accums,
                    cfg.trainings.precisions,
                    cfg.trainings.lr,
                    cfg.trainings.optimizer,
                    cfg.trainings.steps,
                ):
                    tmp_cfg.model.training.batch_size = bs
                    tmp_cfg.model.training.grad_accum = grad_acc
                    tmp_cfg.model.training.precision = precision
                    tmp_cfg.model.training.lr = lr
                    tmp_cfg.model.training.optimizer = optimizer
                    tmp_cfg.model.training.steps = steps

                    gpus_per_node = expand_arch_gpu_configs(tmp_cfg)

                    # Get min number of nodes to run based on model and hpc architecture
                    min_nodes = rules.MinNodesMemoryRule()._min_nodes_required(tmp_cfg)
                    candidate_nodes = rules.MinNodesMemoryRule()._nodes_candidates(
                        tmp_cfg.slurm.sbatch.gpus_per_node,
                        max_gpus_scale=tmp_cfg.model.max_gpus_scale,
                    )
                    nodes_to_run = [n for n in candidate_nodes if n >= min_nodes]

                    for nodes in nodes_to_run:
                        tmp_cfg.slurm.sbatch.nodes = nodes

                        parameters_combo = f"{cfg.machine.name}/{cfg.model.name}/{cfg.framework.name}/{cfg.dataset.name}/nodes-{nodes}"

                        tmp_cfg.slurm.sbatch.chdir = os.path.join(
                            tmp_cfg.experiment.output_dir, parameters_combo
                        )

                        yaml_filename = (
                            f"{parallelism}--"
                            + f"bs{bs}-"
                            + f"grad_accum{grad_acc}-"
                            + f"prec{precision}-"
                            + f"steps{steps}"
                            + ".yaml"
                        )
                        # MAKE SURE BEFORE DELETE
                        # run_path = os.path.join(
                        #    outpath,
                        #    parameters_combo,
                        # )
                        # os.makedirs(run_path, exist_ok=True)
                        for g in gpus_per_node:
                            if g == 1 and nodes == 1 and parallelism == "none":
                                tmp_cfg.slurm.sbatch.gpus_per_node = 1
                                tmp_cfg.slurm.sbatch.gres = "gpu:1"
                            msg = f"\t· parallelism: {parallelism} | nodes:{nodes} | bs:{bs} | grad_accum:{grad_acc} | precision:{precision} | steps:{steps}"

                            # MAKE SURE BEFORE DELETE
                            # outpath_yaml = os.path.join(run_path, yaml_filename)
                            passed, results_msg = rules.is_valid(tmp_cfg)
                            if not passed:
                                skipped.append([tmp_cfg, results_msg])
                                msg += f"-> {RED}❌ Failed:{RESET}"
                                raw_total += 1
                                for f in results_msg:
                                    msg += f"\n\t{YELLOW}  {f.rule_name} --> {f.reason}{RESET}"
                                print(msg)
                                continue
                            msg += f" --> {GREEN}✅ Passed!{RESET}"
                            for s in results_msg:
                                if s.reason == "":
                                    continue
                                # msg += (
                                #    f"\n\t{YELLOW}  {s.rule_name} --> {s.reason}{RESET}"
                                # )

                            print(msg)
                            raw_total += 1
                            tmp_cfg.id = (
                                f"{cfg.machine.name}"
                                + f"_{cfg.model.name}"
                                + f"_{cfg.framework.name}"
                                + f"_{cfg.dataset.name}"
                                + f"_nodes-{nodes}"
                                + f"_{parallelism}"
                                + f"--bs{bs}"
                                + f"-grad_accum{grad_acc}"
                                + f"-prec{precision}"
                                + f"-steps{steps}"
                            )

                            # MAKE SURE BEFORE DELETE
                            # tmp_cfg.slurm.sbatch.chdir = run_path
                            tmp_cfg.experiment.yaml_filename = yaml_filename
                            # Resolve all yaml config parameter references before finish
                            OmegaConf.resolve(tmp_cfg)
                            valid.append(deepcopy(tmp_cfg))

    _print_summary(raw_total, valid, skipped)
    return valid, skipped


def _print_summary(
    total: int,
    valid: list[BenchmarkConfig],
    skipped: list[tuple[BenchmarkConfig, rules.RuleResult]],
):
    table = Table(title="Sweep Summary")
    table.add_column("Status")
    table.add_column("Count")
    table.add_row("Total combos", str(total))
    table.add_row("[green]Valid[/green]", str(len(valid)))
    table.add_row("[yellow]Skipped[/yellow]", str(len(skipped)))
    print("")
    console.print(table)
    # ToDo: Write summary file for failed or skipped configurations


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="no-config-path-provided",
        required=True,
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="no-config-path-provided",
        required=True,
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    GlobalHydra.instance().clear()
    """with initialize_config_dir(
        config_dir=os.path.abspath(args.config_path), version_base=None
    ):
        cfg = compose(config_name=args.config_name)
        my_app(cfg)"""
    generate_valid_combos(
        config_path=os.path.abspath(args.config_path),
        config_name=args.config_name,
        outpath="./benchmarks_to_run",
    )
