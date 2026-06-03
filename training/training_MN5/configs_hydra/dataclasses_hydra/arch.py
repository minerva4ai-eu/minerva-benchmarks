from dataclasses import dataclass, field
from enum import Enum

from omegaconf import MISSING


class AcceleratorType(str, Enum):
    CUDA = "cuda"
    ROCM = "rocm"


@dataclass
class GpuConfig:
    name: str = MISSING
    accelerator_type: str = "cuda"
    vram_gb: float = 1
    theoretical_peak_fp64_tflops: int = MISSING
    theoretical_peak_fp64_tensor_tflops: int = MISSING
    theoretical_peak_fp32_tflops: int = MISSING
    theoretical_peak_tf32_tensor_tflops: int = MISSING
    theoretical_peak_fp16_tensor_tflops: int = MISSING
    theoretical_peak_bf16_tensor_tflops: int = MISSING
    theoretical_peak_fp8_tensor_tflops: int = MISSING
    theoretical_peak_int8_tensor_tops: int = MISSING

    def __post_init__(self):
        self._check_accelerator_type()
        self._check_vram_gb()

    def _check_accelerator_type(
        self,
    ):
        try:
            self.accelerator_type = AcceleratorType(self.accelerator_type)
        except ValueError:
            raise ValueError(
                f"Invalid accelerator_type: '{self.accelerator_type}'. "
                f"Valid values: {[e.value for e in AcceleratorType]}"
            )

    def _check_vram_gb(self):
        if self.vram_gb < 0:
            raise ValueError(
                f"Invalid value for vram_gb: '{self.vram_gb}' < 0. Make sure to set the correct VRAM of the device!"
            )


@dataclass
class NodeConfig:
    gpus_per_node: int = MISSING


@dataclass
class CommConfig:
    internode_comm: str = MISSING
    internode_comm_bandwidth_GBs: float = MISSING
    intranode_comm: str = MISSING
    intranode_comm_bandwidth_GBs: float = MISSING


@dataclass
class HPCArchitecture:
    gpu: GpuConfig = field(default_factory=GpuConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    node: NodeConfig = field(default_factory=NodeConfig)
