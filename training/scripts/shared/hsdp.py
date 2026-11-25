"""Hybrid Sharded Data Parallel (HSDP) device-mesh helpers.

"Shard to fit, replicate to fill": the world is split into a shard
dimension of size S (ranks cooperating to hold one copy of the sharded
model state — kept (sub-)intra-node whenever S <= gpus_per_node) and a
replicate dimension of size R = W / S (data-parallel replicas).

FSDP2 (``fully_shard``) takes a 2D ``DeviceMesh`` for HSDP: dim 0
("dp_replicate") is the replica axis, dim 1 ("dp_shard") the sharding
axis; the dim names are mandatory. Ranks map row-major onto (R, S):
global rank g sits in shard group g // S (contiguous ranks of size S),
so with one process per GPU a shard group of size S <= gpus_per_node
stays inside one node, i.e. on the NVLink fabric.
"""

from __future__ import annotations


def build_hsdp_mesh(world_size: int, shard_group_size: int | None):
    """Build the 2D HSDP device mesh over the default process group.

    Returns ``None`` when the whole world is a single shard group
    (shard_group_size is None or >= world_size) — plain FSDP, i.e. the
    previous full-world sharding behavior.
    """
    if shard_group_size is None or shard_group_size >= world_size:
        return None
    if shard_group_size < 1 or world_size % shard_group_size != 0:
        raise ValueError(
            f"shard_group_size ({shard_group_size}) must be >= 1 and divide "
            f"world_size ({world_size})"
        )

    from torch.distributed.device_mesh import init_device_mesh

    dp_replicas = world_size // shard_group_size
    return init_device_mesh(
        "cuda",
        (dp_replicas, shard_group_size),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
