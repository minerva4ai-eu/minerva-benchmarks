import math

from dataclasses_hydra.benchmark import BenchmarkConfig
from omegaconf import DictConfig

from .base import ConstraintRule, RuleResult


class ParallelismGPUFloor(ConstraintRule):
    """ZeRO/FSDP/DDP require >1 GPU. 'none' requires exactly 1."""

    def check(self, c: BenchmarkConfig) -> RuleResult:
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

    def check(self, c: DictConfig) -> RuleResult:
        supported_parallelisms, framework_parallelism = (
            c.model.parallelism_supported,
            list(c.framework.parallelism.keys())[0],
        )
        # print(f"supported_parallelisms {supported_parallelisms}")
        # print(f"framework_parallelism {framework_parallelism}")
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


class MinNodesMemoryRule(ConstraintRule):
    SAFETY_MARGIN = 0.85

    def check(self, c: DictConfig) -> RuleResult:

        min_gpus, breakdown = self._min_gpus_required(c)
        if min_gpus == -1:
            return RuleResult(
                False,
                "min_nodes_memory",
                f"Model needs ~{breakdown['total_gb']} GB total or ~{breakdown['per_gpu_gb']} GB per gpu; no feasible config found up to {self._gpu_candidates(c.arch.node.gpus_per_node)[-1]} GPUs",
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

    def _min_nodes_required(self, c: DictConfig) -> int:
        min_gpus, _ = self._min_gpus_required(c)
        return math.ceil(min_gpus / c.arch.node.gpus_per_node)

    def _min_gpus_required(self, c: DictConfig) -> tuple[int, dict]:
        arch_type = c.model.get("architecture_type", "dense")
        precision = c.model.training.precision
        optimizer = c.model.get("optimizer", "adamw")
        assert isinstance(c.framework.parallelism, DictConfig) and (
            len(c.framework.parallelism) == 1
        ), (
            f"c.framework.parallelism is expected to be dict of length 1, instead received: {c.framework.parallelism}"
        )
        parallelism = list(c.framework.parallelism.keys())[0]
        batch = c.model.training.batch_size
        seq_len = c.dataset.max_seq_len
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

        # ── activation memory ─────────────────────────────────────────
        M_activations = self._activation_memory(c, arch_type, batch, seq_len, bpp)

        M_total = M_params + M_gradients + M_optimizer + M_activations

        # ── find minimum GPU count where parallelism makes it fit ─────
        divisor_fn = PARALLELISM_DIVISOR.get(parallelism, lambda n: 1)
        gpus_per_node = c.arch.node.gpus_per_node

        per_gpu = n = -1
        for n in self._gpu_candidates(gpus_per_node):
            per_gpu = M_total / divisor_fn(n)
            # print(f"mtotal: {M_total} | per_gpu: {per_gpu} | gpu_vram: {gpu_vram}" )
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

    def _activation_memory(self, c, arch_type, batch, seq_len, bpp) -> int:
        m = c.model
        use_grad_checkpointing = m.training.get("gradient_checkpointing", True)

        if not (hasattr(m, "num_layers") and hasattr(m, "hidden_dim")):
            # fallback: empirical ~1 GB per billion active params per batch item
            P_active = m.get("active_params_billions", m.params_billions)
            return int(P_active * 1e9 * batch * 0.1)

        L = m.num_layers
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
