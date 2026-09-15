import torch
import torch._inductor.config as inductor_config

# Save tuning results to disk so next run skips benchmarking
# inductor_config.autotune_local_cache = True
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# torch.compiler.config.assume_static_by_default = False
# torch._dynamo.config.capture_dynamic_shapes = True
# torch._dynamo.config.recompile_limit = 16
