import math
from copy import deepcopy
from functools import cache
from typing import Literal

from configs_hydra.dataclasses_hydra.benchmark import BenchmarkConfig
from omegaconf import DictConfig

from .base import ConstraintRule, RuleResult


class ParallelismGPUFloor(ConstraintRule):
    """ZeRO/FSDP/DDP require >1 GPU. 'none' requires exactly 1."""

    def check(self, c: BenchmarkConfig) -> RuleResult:
        if c.framework.megatron_parallelism:
            _check = RuleResult(False, "Default fail! Check code for bug!")
            for p, min_max in c.framework.parallelism.items():
                _c = deepcopy(c)
                _c.framework.parallelism = {p: min_max}
                _check = self._check(_c)
                if not _check.passed:
                    return _check
            return _check
        return self._check(c)

    def _check(self, c: BenchmarkConfig) -> RuleResult:
        gpus = c.slurm.sbatch.nodes * c.slurm.sbatch.gpus_per_node

        _p = list(c.framework.parallelism.keys())
        if len(_p) > 1:
            raise Exception(
                f"Fuction ParallelismGPUFloor.check() received List instead of Dict for cfg.framework.parallelism: '{c.framework.parallelism}'"
            )
        p = c.framework.parallelism[_p[0]]

        min_gpus = p.min_gpus
        max_gpus = p.max_gpus if p.max_gpus else 1024
        if gpus < min_gpus:
            return RuleResult(
                False,
                "parallelism_gpu_floor",
                f"{p} needs ≥{min_gpus} GPUs, got {gpus}",
            )
        if gpus > max_gpus:
            return RuleResult(
                False,
                "parallelism_gpu_ceiling",
                f"{p} needs ≤{max_gpus} GPUs, got {gpus}",
            )
        return RuleResult(True, "parallelism_gpu_floor")


class FrameworkParallelismValidityRule(ConstraintRule):
    """ZeRO/FSDP/DDP require >1 GPU. 'none' requires exactly 1."""

    @property
    def rule_name(
        self,
    ):
        return "framework_parallelism_validity_check"

    def check(self, c: BenchmarkConfig) -> RuleResult:
        if c.framework.megatron_parallelism:
            supported_parallelisms, framework_parallelism = (
                c.model.parallelism_supported,
                list(c.framework.parallelism.keys()),
            )
            for supported_parallelism in supported_parallelisms:
                print(f"{framework_parallelism} ~  {supported_parallelisms}")
                if supported_parallelism not in framework_parallelism:
                    return RuleResult(
                        False,
                        self.rule_name,
                        f"Parallelism '{framework_parallelism}' is invalid for framework '{c.framework.name}'! Skipping...",
                    )
            return RuleResult(True, "parallelism_gpu_floor")
        return self._check(c)

    def _check(self, c: DictConfig) -> RuleResult:
        supported_parallelisms, framework_parallelism = (
            c.model.parallelism_supported,
            list(c.framework.parallelism.keys())[0],
        )
        if framework_parallelism not in supported_parallelisms:
            return RuleResult(
                False,
                self.rule_name,
                f"Parallelism '{framework_parallelism}' is invalid for framework '{c.framework.name}'! Skipping...",
            )
        return RuleResult(True, "parallelism_gpu_floor")


BYTES_PER_PARAM = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}
OPTIMIZER_BYTES = {"adamw": 8, "adam": 8, "sgd": 0, "adafactor": 4}
MAX_GPUS_SCALE = 256
PARALLELISM_DIVISOR = {
    "none": lambda n: 1,
    "ddp": lambda n: 1,
    "fsdp": lambda n: n,
    # Single gpu
    # M = P×4 (params) + P×4 (grads) + P×8 (Adam states) + activations
    #     = P×16 + activations
    # Zero 1 shards only optimizer states
    # M = P×4 (params, replicated)
    #    + P×4 (grads, replicated)
    #    + P×8/n (optimizer, sharded)
    #    + activations
    #    = P×8 + P×8/n + activations
    # divisor = M_single / M_per_gpu
    #    = (P×16 + act) / (P×8 + P×8/n + act)
    #    ≈ 16 / (8 + 8/n)           [for large P]
    #    = 2n / (n + 1)
    # Zero 2 only shards gradients and optimizer states
    # M = P×4 (params, replicated)
    #    + P×4/n (grads, sharded)
    #    + P×8/n (optimizer, sharded)
    #    + activations
    #    = P×4 + P×12/n + activations
    # divisor = (P×16 + act) / (P×4 + P×12/n + act)
    #    ≈ 16 / (4 + 12/n)          [for large P]
    #    = 4n / (n + 3)
    "zero1": lambda n: 2 * n / (n + 1),
    "zero2": lambda n: 4 * n / (n + 3),
    "zero3": lambda n: n,
}

# Parallelism strategy names (as keys of c.framework.parallelism) that should
# be routed through the Megatron-style TP/PP/CP/DP search instead of the
# single-axis PARALLELISM_DIVISOR path above.
MEGATRON_PARALLELISM_KEYS = {"tp", ""}


def _cfg_get(node, key, default=None):
    """Read `key` off a DictConfig-or-plain-object node without blowing up
    if it isn't present (OmegaConf's `.get` and plain attribute access both
    show up in this codebase, so support either)."""
    if node is None:
        return default
    if hasattr(node, "get"):
        val = node.get(key, default)
        return default if val is None else val
    return getattr(node, key, default)


def megatron_divisors(
    combo: dict[str, int],
    arch_type: Literal["dense", "moe"],
    use_distributed_optimizer: bool = True,
    expert_ratio: float = 0.80,
) -> dict[str, float]:
    """Per-component memory divisors to compute single-GPU memory footprint.

    Divisor definition:
        single_gpu_memory_component = total_model_component_memory / divisor

    Args:
        combo: Parallelism configuration dict containing any subset of:
               {"tp", "pp", "cp", "dp", "ep", "sp"}
        arch_type: "dense" or "moe"
        use_distributed_optimizer: Megatron's --use-distributed-optimizer (ZeRO-1).
        expert_ratio: For MoE models, the fraction of total parameters that reside
                      inside expert layers (default 0.80 = 80% expert, 20% dense).

    Returns:
        Dict of memory divisors per component: params, gradients, optimizer, activations.
    """
    tp = combo.get("tp", 1)
    pp = combo.get("pp", 1)
    cp = combo.get("cp", 1)
    dp = combo.get("dp", 1)
    sp = combo.get("sp", 1)
    ep = combo.get("ep", 1)

    # ---------------------------------------------------------------------
    # 1. PARAMETERS & GRADIENTS
    # ---------------------------------------------------------------------
    # Dense parameters shard across TP (intra-layer) and PP (inter-layer)
    dense_param_divisor = max(tp * pp, 1)

    if arch_type == "moe":
        #########################################################################
        # ToDo: Introduc'e paramenter field 'expert_ratio' in model config      #
        #       Or calculate it using a combination of model field/properties   #
        #########################################################################

        # Expert parameters shard across TP, PP, AND EP
        expert_param_divisor = max(tp * pp * ep, 1)

        # Weighted Harmonic Mean to find the exact effective divisor
        # across the composite (Dense + MoE) model footprint
        param_grad_divisor = 1.0 / (
            ((1.0 - expert_ratio) / dense_param_divisor)
            + (expert_ratio / expert_param_divisor)
        )
    else:
        param_grad_divisor = float(dense_param_divisor)

    # ---------------------------------------------------------------------
    # 2. OPTIMIZER STATES
    # ---------------------------------------------------------------------
    if use_distributed_optimizer:
        # ZeRO-1 shards optimizer states across DP and CP ranks.
        # Dense Optim Divisor  = tp * pp * (dp * cp)
        # Expert Optim Divisor = (tp * pp * ep) * ((dp / ep) * cp) = tp * pp * dp * cp
        # Because both evaluate to (tp * pp * dp * cp), the optimizer divisor is exact!
        optim_divisor = max(tp * pp * dp * cp, 1)
    else:
        # Without Distributed Optimizer, optimizer states mirror parameters
        optim_divisor = param_grad_divisor

    # ---------------------------------------------------------------------
    # 3. ACTIVATIONS
    # ---------------------------------------------------------------------
    # Sequence parallelism (sp) or Tensor parallelism (tp) shards sequence activations.
    # Context parallelism (cp) further chunks sequence length across CP ranks.
    act_tp_sp = max(tp, sp)
    activation_divisor = max(act_tp_sp * cp, 1)

    return {
        "params": float(param_grad_divisor),
        "gradients": float(param_grad_divisor),
        "optimizer": float(optim_divisor),
        "activations": float(activation_divisor),
    }


class MinNodesMemoryRule(ConstraintRule):
    SAFETY_MARGIN = 0.85

    def check(self, c: DictConfig) -> RuleResult:

        min_gpus, breakdown = self._min_gpus_required(c)
        if min_gpus == -1:
            return RuleResult(
                False,
                "min_nodes_memory",
                f"Model needs ~{breakdown['total_gb']} GB total or ~{breakdown.get('per_gpu_gb')} GB per gpu; "
                f"no feasible config found up to {self._gpu_candidates(c.arch.node.gpus_per_node)[-1]} GPUs",
            )

        actual_gpus = c.arch.node.gpus_per_node * c.slurm.sbatch.nodes
        if actual_gpus < min_gpus:
            min_nodes = math.ceil(min_gpus / c.arch.node.gpus_per_node)
            return RuleResult(
                False,
                "min_nodes_memory",
                f"{c.model.name} needs ≥{min_gpus} GPUs / ≥{min_nodes} nodes. \n"
                f"{breakdown}",
            )
        return RuleResult(True, "min_nodes_memory", str(breakdown))

    def _min_nodes_required(self, c: BenchmarkConfig) -> int:
        min_gpus, _ = self._min_gpus_required(c)
        return math.ceil(min_gpus / c.arch.node.gpus_per_node)

    def _min_gpus_required(self, c: BenchmarkConfig) -> tuple[int, dict]:

        arch_type = c.model.architecture_type
        precision = c.model.training.precision
        optimizer = c.model.training.get("optimizer", "adamw")

        gpu_vram = c.arch.gpu.vram_gb * 1e9 * self.SAFETY_MARGIN

        bpp = BYTES_PER_PARAM[precision]
        opt_bytes = OPTIMIZER_BYTES.get(optimizer, 8)

        # ── parameter memory ──────────────────────────────────────────
        # MoE: ALL experts must be in VRAM even though only top-k run
        # Dense / GQA: straightforward
        if arch_type == "moe":
            P_vram = c.model.total_params_billions * 1e9  # full weight load
            P_compute = c.model.active_params_billions * 1e9  # for grad/optim
        else:
            P_vram = c.model.total_params_billions * 1e9
            P_compute = P_vram

        M_params = P_vram * bpp
        M_gradients = P_compute * bpp  # grads only for active params in MoE
        M_optimizer = P_compute * opt_bytes  # optimizer states for active params

        if c.framework.megatron_parallelism:
            return self._min_gpus_required_megatron(
                c=c,
                bpp=bpp,
                M_params=M_params,
                M_gradients=M_gradients,
                M_optimizer=M_optimizer,
                gpu_vram=gpu_vram,
            )

        batch = c.model.training.batch_size
        seq_len = c.dataset.max_seq_len
        gpus_per_node = c.arch.node.gpus_per_node

        # ── single-axis path (none/ddp/fsdp/zero*), unchanged ─────────
        M_activations = self._activation_memory(c, arch_type, batch, seq_len, bpp)
        M_total = M_params + M_gradients + M_optimizer + M_activations

        assert isinstance(c.framework.parallelism, DictConfig) and (
            len(c.framework.parallelism) == 1
        ), (
            f"c.framework.parallelism is expected to be dict of length 1, instead received: {c.framework.parallelism}"
        )
        parallelism = list(c.framework.parallelism.keys())[0]
        divisor_fn = PARALLELISM_DIVISOR.get(parallelism, lambda n: 1)

        per_gpu = n = -1
        for n in self._gpu_candidates(gpus_per_node):
            per_gpu = M_total / divisor_fn(n)
            if per_gpu <= gpu_vram:
                return n, {
                    "params_gb": round(M_params / 1e9, 2),
                    "gradients_gb": round(M_gradients / 1e9, 2),
                    "optimizer_gb": round(M_optimizer / 1e9, 2),
                    "activations_gb": round(M_activations / 1e9, 2),
                    "total_gb": round(M_total / 1e9, 2),
                    "per_gpu_gb": round(per_gpu / 1e9, 2),
                    "gpu_usable_gb": round(gpu_vram / 1e9, 2),
                    "min_gpus": n,
                    "arch_type": arch_type,
                }

        return -1, {
            "params_gb": round(M_params / 1e9, 2),
            "gradients_gb": round(M_gradients / 1e9, 2),
            "optimizer_gb": round(M_optimizer / 1e9, 2),
            "activations_gb": round(M_activations / 1e9, 2),
            "total_gb": round(M_total / 1e9, 2),
            "per_gpu_gb": round(per_gpu / 1e9, 2),
            "gpu_usable_gb": round(gpu_vram / 1e9, 2),
            "min_gpus": n,
            "arch_type": arch_type,
        }

    def _min_gpus_required_megatron(
        self,
        c: BenchmarkConfig,
        bpp: int,
        M_params: float,
        M_gradients: float,
        M_optimizer: float,
        gpu_vram: float,
    ) -> tuple[int, dict]:
        """Search node counts, and within each node count every valid
        (tp, pp, cp, dp) tiling of the cluster, picking the smallest total
        GPU count for which *some* tiling fits in VRAM (and among tilings
        at that GPU count, the one that uses the least memory per GPU)."""
        gpus_per_node = c.arch.node.gpus_per_node
        arch_type = c.model.architecture_type
        batch = c.model.training.batch_size
        seq_len = c.dataset.max_seq_len

        megatron_parallelisms = c.model.megatron_parallelism_supported

        assert "pp" in megatron_parallelisms, (
            f"Megatron parallelism axis 'pp' not found in model.megatron_parallelism_supported {megatron_parallelisms}"
        )
        use_dist_optim = bool(
            _cfg_get(c.model.training, "use_distributed_optimizer", True)
        )

        # Optional fixed dimensions from config — if the user pinned tp/pp/cp
        # explicitly, only consider tilings that match.

        num_layers = getattr(c.model, "num_layers", None)

        best_infeasible_breakdown = None

        # Iterate over candidate nodes based on gpus_per_node of system
        for nnodes in self._nodes_candidates(gpus_per_node):
            total_gpus = nnodes * gpus_per_node
            # Get all possible megatron parallelismi combos based on nnodes
            combos = megatron_parallelism_combos(
                gpus_per_node, nnodes, megatron_parallelisms
            )

            best_for_this_gpu_count = None  # dict with lowest per_gpu memory

            for combo in combos:
                divisors = megatron_divisors(combo, arch_type, use_dist_optim)
                layers_per_stage = (
                    math.ceil(num_layers / combo["pp"]) if num_layers else None
                )
                M_act_single_gpu_base = self._activation_memory(
                    c,
                    arch_type,
                    batch,
                    seq_len,
                    bpp,
                    num_layers_override=layers_per_stage,
                )
                M_activations_per_gpu = M_act_single_gpu_base / divisors["activations"]

                per_gpu = (
                    M_params / divisors["params"]
                    + M_gradients / divisors["gradients"]
                    + M_optimizer / divisors["optimizer"]
                    + M_activations_per_gpu
                )

                if (
                    best_for_this_gpu_count is None
                    or per_gpu < best_for_this_gpu_count["per_gpu"]
                ):
                    best_for_this_gpu_count = {
                        "per_gpu": per_gpu,
                        "activations": M_activations_per_gpu,
                        "combo": combo,
                    }

            if best_for_this_gpu_count is None:
                continue  # no valid tp/pp/cp/dp tiling for this node count

            per_gpu = best_for_this_gpu_count["per_gpu"]
            combo = best_for_this_gpu_count["combo"]
            breakdown = {
                "params_gb": round(M_params / 1e9, 2),
                "gradients_gb": round(M_gradients / 1e9, 2),
                "optimizer_gb": round(M_optimizer / 1e9, 2),
                "activations_gb": round(
                    best_for_this_gpu_count["activations"] / 1e9, 2
                ),
                "total_gb": round(
                    (
                        M_params
                        + M_gradients
                        + M_optimizer
                        + best_for_this_gpu_count["activations"]
                    )
                    / 1e9,
                    2,
                ),
                "per_gpu_gb": round(per_gpu / 1e9, 2),
                "gpu_usable_gb": round(gpu_vram / 1e9, 2),
                "min_gpus": total_gpus,
                "arch_type": arch_type,
                "tp": combo.get("tp", None),
                "pp": combo.get("pp", None),
                "cp": combo.get("cp", None),
                "dp": combo.get("dp", None),
                "ep": combo.get("ep", None),
                "sp": combo.get("sp", None),
                "use_distributed_optimizer": use_dist_optim,
            }

            if per_gpu <= gpu_vram:
                return total_gpus, breakdown

            # keep the closest-fitting infeasible result around in case we
            # exhaust every node candidate, so the failure message is useful
            best_infeasible_breakdown = breakdown

        return -1, best_infeasible_breakdown or {
            "total_gb": round((M_params + M_gradients + M_optimizer) / 1e9, 2),
            "per_gpu_gb": None,
            "gpu_usable_gb": round(gpu_vram / 1e9, 2),
            "min_gpus": -1,
            "arch_type": arch_type,
            "note": "no feasible tp/pp/cp/dp tiling found for any candidate node count",
        }

    def _activation_memory(
        self, c, arch_type, batch, seq_len, bpp, num_layers_override: int | None = None
    ) -> int:
        m = c.model
        use_grad_checkpointing = m.training.get("gradient_checkpointing", True)

        if not (hasattr(m, "num_layers") and hasattr(m, "hidden_dim")):
            # fallback: empirical ~1 GB per billion active params per batch item
            P_active = m.get("active_params_billions", m.params_billions)
            return int(P_active * 1e9 * batch * 0.1)

        # `num_layers_override` lets callers pass the layer count for a
        # single pipeline stage (ceil(total_layers / pp)) instead of the
        # model's full depth, so PP's reduction in activation memory is
        # reflected without changing the base single-GPU calculation.
        L = num_layers_override if num_layers_override is not None else m.num_layers
        H = m.hidden_dim
        FFN = m.get("ffn_intermediate_dim", 4 * H)

        # attention activations — GQA reduces KV side
        n_q = m.get("num_attention_heads", 32)
        n_kv = m.get("num_kv_heads", n_q)  # = n_q for MHA, < n_q for GQA/MQA
        head_dim = H // n_q

        M_attn_scores = batch * n_q * seq_len * seq_len * bpp  # QK^T matrix
        M_kv_cache = batch * n_kv * seq_len * head_dim * 2 * bpp  # K and V buffers

        # FFN activations
        if arch_type == "moe":
            # only top_k experts run per token, but all are loaded
            # activation buffer = top_k experts × FFN size
            top_k = m.get("top_k_experts", 2)
            M_ffn = batch * seq_len * FFN * top_k * bpp
        else:
            M_ffn = batch * seq_len * FFN * 2 * bpp  # gate + up projections

        M_per_layer = M_attn_scores + M_kv_cache + M_ffn

        # gradient checkpointing: store ~sqrt(L) layers instead of all L
        layer_factor = math.ceil(math.sqrt(L)) if use_grad_checkpointing else L

        return int(M_per_layer * layer_factor)

    @staticmethod
    def _gpu_candidates(
        gpus_per_node: int, max_gpus_scale: int = MAX_GPUS_SCALE
    ) -> list[int]:
        candidates, n = [1], gpus_per_node
        while n <= max_gpus_scale:
            if n not in candidates:
                candidates.append(n)
            n *= 2
        return sorted(candidates)

    @staticmethod
    def _nodes_candidates(
        gpus_per_node: int, max_gpus_scale: int = MAX_GPUS_SCALE
    ) -> list[int]:

        candidates, n = [1], 2
        assert max_gpus_scale % gpus_per_node == 0, (
            "_nodes_candidates: max_gpus_scale must be multiple of gpus_per_node"
        )
        max_nodes_scale = max_gpus_scale / gpus_per_node
        while n <= max_nodes_scale:
            if n not in candidates:
                candidates.append(n)
            n *= 2
        return sorted(candidates)


# Registry — just add new rules here, no other changes needed
ALL_RULES = [
    ParallelismGPUFloor(),
    FrameworkParallelismValidityRule(),
    MinNodesMemoryRule(),
]


def validate(combo: DictConfig) -> list[RuleResult]:
    return [r.check(combo) for r in ALL_RULES]


def is_valid(combo: DictConfig) -> tuple[bool, list[RuleResult]]:
    checks = validate(combo)
    scores = [r.passed for r in checks]
    passed = all(scores)
    if passed:
        return passed, checks
    fails = [f for f, p in zip(checks, scores) if not p]
    return passed, fails


def megatron_parallelism_combos(
    gpus_per_node: int, nnodes: int, axes: list[str]
) -> list[dict[str, int]]:
    """Return all valid parallelism combinations for given hardware and selected axes.

    Args:
        gpus_per_node: Number of GPUs per node.
        nnodes: Number of nodes.
        axes: List of target parallelism axes (e.g., ["tp", "pp", "cp", "dp", "ep", "sp"]).

    Rules & Constraints:
        - `tp` is fixed to `gpus_per_node` (if present in `axes`).
        - `pp <= nnodes` (if present in `axes`).
        - `sp` can only be > 1 if `tp > 1` (if present in `axes`).
        - Product of all selected axes equals `total_gpus`.
        - All axis values >= 1.
    """
    # Remove duplicates while maintaining insertion order
    clean_axes = list(dict.fromkeys(axes))
    total_gpus = gpus_per_node * nnodes

    fixed_values: dict[str, int] = {}

    # 1. Handle fixed TP constraint
    if "tp" in clean_axes:
        fixed_values["tp"] = gpus_per_node

    # Determine effective TP to enforce SP rule
    effective_tp = fixed_values.get("tp", 1)

    # 2. Handle SP constraint when TP <= 1
    if "sp" in clean_axes and effective_tp <= 1:
        fixed_values["sp"] = 1

    dynamic_axes = [a for a in clean_axes if a not in fixed_values]

    # Calculate remaining allocation space after applying fixed axes
    fixed_product = 1
    for val in fixed_values.values():
        fixed_product *= val

    if total_gpus % fixed_product != 0:
        return []

    target_product = total_gpus // fixed_product

    # Special Case: No dynamic axes left to tile
    if not dynamic_axes:
        return [{"nnodes": nnodes, **fixed_values}] if target_product == 1 else []

    @cache
    def find_factor_tuples(target: int, count: int) -> list[tuple[int, ...]]:
        """Find all tuples of length `count` of positive integers multiplying to `target`."""
        if count == 1:
            return [(target,)]

        combos = []
        for factor in range(1, target + 1):
            if target % factor == 0:
                sub_tuples = find_factor_tuples(target // factor, count - 1)
                for st in sub_tuples:
                    combos.append((factor,) + st)
        return combos

    results: list[dict[str, int]] = []

    # Get all possible integer partitions for dynamic axes
    all_tuples = find_factor_tuples(target_product, len(dynamic_axes))

    # Map and validate combinations against rules
    for combo in all_tuples:
        assignment = dict(zip(dynamic_axes, combo))

        # Enforce PP constraint
        if assignment.get("pp", 1) > nnodes:
            continue

        # Enforce SP constraint (sp > 1 requires tp > 1)
        if assignment.get("sp", 1) > 1 and effective_tp <= 1:
            continue

        results.append({"nnodes": nnodes, **fixed_values, **assignment})

    return results
