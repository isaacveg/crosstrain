import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.optim import SGD

from .shard_utils import (
    get_layer_shards,
    print_shard_param_counts_from_shards,
)


def init_streaming_state(
    model: torch.nn.Module, args: Any, logger: Any
) -> Dict[str, Any]:
    """
    Initialize state for the 'streaming' algorithm.

    Returns:
        {
            "shard_tracker": Dict[int, Dict[str, Any]],
            "base_sync_points": List[int],
            "num_layers": int,
        }
    """
    if logger:
        logger.info(
            f"Initializing state for 'streaming' algorithm with {args.num_shards} shards (pattern={args.pattern})."
        )

    num_shards = args.num_shards
    shard_params, num_layers = get_layer_shards(model, num_shards, args.pattern)

    # Each shard syncs inside one sync_interval at a fixed relative offset
    sync_shard_interval = args.sync_interval // num_shards
    base_sync_points = [
        i * sync_shard_interval + args.offset for i in range(num_shards)
    ]
    if logger:
        logger.info(
            f"分片方式 {args.pattern} 每个分片在间隔内的相对同步时间点: {base_sync_points}"
        )

    # Global stats per shard (param count / bytes); also logs per-shard stats
    global_param_counts, global_byte_counts = print_shard_param_counts_from_shards(
        shard_params, logger
    )

    shard_tracker: Dict[int, Dict[str, Any]] = {}
    for shard_idx, param_list in enumerate(shard_params):
        shard_tracker[shard_idx] = {
            # A working copy used as the 'global' reference for outer updates
            "params": [p.data.clone() for p in param_list],
            # Snapshot of local params at send time; will be set during runtime
            "staged_params": None,
            "sent_at_step": 0,
            "next_receive_step": 0,
            # Reference to the actual model parameters for this shard
            "param_refs": param_list,
            # Static stats for accounting
            "global_num_params": global_param_counts[shard_idx],
            "global_num_bytes": global_byte_counts[shard_idx],
        }

        # Optional outer optimizer per shard (if enabled)
        outer_optimizer = None
        if args.outer_lr != 1.0:
            # The codebase uses cloned tensors for "params" and attaches grads to them
            outer_optimizer = SGD(
                shard_tracker[shard_idx]["params"],
                lr=args.outer_lr,
                momentum=0.9,
                nesterov=bool(getattr(args, "use_nesterov", False)),
            )
        shard_tracker[shard_idx]["outer_optimizer"] = outer_optimizer

    if logger:
        logger.info(f"模型层数: {num_layers}, 分片数: {num_shards}")

    return {
        "shard_tracker": shard_tracker,
        "base_sync_points": base_sync_points,
        "num_layers": num_layers,

    }

def sync_streaming_diloco(
    model,
    shard_tracker,
    sync_shard_idx,
    world_size,
    logger,
    comm_delay,
    num_shards,
    alpha,
):
    """
    分块同步模型参数 (Corrected Logic)
    model: 当前工作模型 (state at t_now)
    shard_tracker: 包含旧状态的字典
      - params: state at t_rec - H (base for outer gradient)
      - staged_params: state at t_rec (记录于 delay_steps 之前)
    """
    print("--------------------sdiloco is executing-------------------------")
    cur_shard = shard_tracker[sync_shard_idx]

    # --- 1. Calculate Outer Gradient Delta: Δm,p = θ(t_rec - H)_m,p - θ(t_rec)_m,p ---
    with torch.no_grad():
        sync_grads = [
            p_old.data - p_rec.data
            for p_old, p_rec in zip(cur_shard["params"], cur_shard["staged_params"])
        ]

    # --- 2. Communicate and Average Delta: Δp = (1/M) * Σ Δm,p ---
    comm_start = time.time()
    # Batch communication - flatten gradients for this shard
    try:
        flat_grads = torch.cat([g.flatten() for g in sync_grads])
    except RuntimeError as e:
        logger.error(f"Error flattening gradients for shard {sync_shard_idx}: {e}")
        for i, g in enumerate(sync_grads):
            logger.error(
                f"  Grad {i} shape: {g.shape}, dtype: {g.dtype}, device: {g.device}"
            )
        return 0.0  # Skip if flattening fails

    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    flat_grads.div_(world_size)  # flat_grads now holds averaged Δp

    # Unflatten the averaged delta back into sync_grads list structure
    # sync_grads will now hold the averaged Δp, structured like the parameters
    offset = 0
    for grad in sync_grads:
        numel = grad.numel()
        if offset + numel > flat_grads.numel():
            logger.error(
                f"Error unflattening: offset {offset} + numel {numel} > flat_grads size {flat_grads.numel()}"
            )
            return 0.0  # Skip if unflattening calculation is wrong
        grad.copy_(flat_grads[offset : offset + numel].view_as(grad))
        offset += numel
    comm_time = time.time() - comm_start
    del flat_grads  # Free memory

    # --- 3. Apply Outer Optimization: θ_outer = OuterOpt(θ(t_rec - H)_p, Δp) ---
    if cur_shard["outer_optimizer"]:
        cur_shard["outer_optimizer"].zero_grad()  # Clean up gradients
        for param, avg_delta in zip(cur_shard["params"], sync_grads):
            if param.grad is None:
                param.grad = avg_delta.clone()  # Create grad buffer if needed
            else:
                param.grad.copy_(avg_delta)
        # Perform the optimizer step (updates cur_shard["params"])
        cur_shard["outer_optimizer"].step()
    else:  # Equivalent to simple averaging (outer_lr=1.0 means SGD with lr=1.0)
        with torch.no_grad():
            for param, avg_delta in zip(cur_shard["params"], sync_grads):
                param.data.sub_(avg_delta.data)
        # cur_shard["params"] now holds θ_outer(t_now)_p
    del sync_grads  # Free memory associated with the averaged delta list

    # --- 4. Merge: θ(t_now)_m,p = α*θ(t_now)_m,p + (1-α)*θ_outer ---
    globally_updated = cur_shard["params"]
    param_refs = cur_shard["param_refs"]
    with torch.no_grad():
        for local_p, updated_p in zip(param_refs, globally_updated):
            local_p.data.mul_(alpha).add_(updated_p.data, alpha=1 - alpha)

    # --- 5. Prepare State for Next Cycle ---
    with torch.no_grad():
        cur_shard["staged_params"] = None  # Optional memory optimization

    # --- Logging and Return ---
    if comm_delay:
        # Simulate delay based on config, dividing total delay by shards for average effect
        actual_delay = comm_delay / num_shards
        logger.info(f"分片 {sync_shard_idx + 1} 模拟通信时间: {actual_delay:.4f} 秒")
        return actual_delay
    else:
        # Return measured all-reduce time
        logger.info(
            f"分片 {sync_shard_idx + 1} 通信时间 (all-reduce): {comm_time:.4f} 秒"
        )
        return comm_time
