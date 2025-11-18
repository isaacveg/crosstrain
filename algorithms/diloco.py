import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.optim import SGD


def init_diloco_state(
    model: torch.nn.Module, args: Any, logger: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Initialize state for the 'diloco' algorithm.

    Returns:
        {
            "original_snapshot": deepcopy(model),
            "outer_optimizer": SGD(...) or None,
        }
    """
    if logger:
        logger.info("Initializing state for 'DiLoCo' algorithm.")
    offload_to_cpu = getattr(args, "outer_opt_on_cpu", getattr(args, "outer_opt_cpu", False))
    if not hasattr(args, "outer_opt_cpu"):
        setattr(args, "outer_opt_cpu", offload_to_cpu)
    target_device = torch.device("cpu") if offload_to_cpu else next(model.parameters()).device
    original_snapshot = deepcopy(model).to(target_device)
    if args.outer_lr != 1.0:
        for param in original_snapshot.parameters():
            param.requires_grad_(True)
        outer_optimizer = SGD(
            original_snapshot.parameters(),
            lr=args.outer_lr,
            momentum=0.9,
            nesterov=bool(getattr(args, "use_nesterov", False)),
        )
    else:
        outer_optimizer = None

    return {
        "original_snapshot": original_snapshot,
        "outer_optimizer": outer_optimizer,
    }


def sync_diloco(
    model, original_model, outer_optimizer, world_size, logger, comm_delay=None, offload_cpu=None
):
    """
    Synchronizes model parameters across distributed processes.

    Handles two cases:
    1. If outer_optimizer is provided (DiLoCo-like): Calculates the update direction
       (original_param - current_param), aggregates it across workers, applies
       it to the original_model using the outer_optimizer, and copies the
       updated parameters back to the model.
    2. If outer_optimizer is None: Directly averages the model parameters across
       all workers using all_reduce.

    Args:
        model: The current model instance on the worker.
        original_model: A snapshot of the model before local steps (used with outer_optimizer).
        outer_optimizer: The optimizer for the global update step (e.g., SGD).
        world_size: Total number of distributed processes.
        logger: Logger instance.
        comm_delay: Optional simulated communication delay in seconds.

    Returns:
        The communication time in seconds.
    """
    print("--------------------diloco is executing-------------------------")
    sync_comm_time = 0.0
    if offload_cpu is None:
        prototype_param = next(original_model.parameters(), None)
        offload_cpu = prototype_param is not None and prototype_param.device.type == "cpu"
    with torch.no_grad():
        if outer_optimizer:
            grads_for_sync = []
            # Calculate the effective gradient (update direction) for the outer step
            for param, original_param in zip(
                model.parameters(), original_model.parameters()
            ):
                if original_param.grad is None or original_param.grad.shape != original_param.data.shape:
                    original_param.grad = torch.zeros_like(original_param.data)
                # In-place: grad = original - current
                original_param.grad.copy_(original_param.data)
                if offload_cpu:
                    p_cpu = param.detach().to("cpu", copy=True)
                    original_param.grad.sub_(p_cpu)
                else:
                    original_param.grad.sub_(param.data)
                grads_for_sync.append(original_param.grad)

            # --- Batch Communication ---
            comm_start_sync = time.time()
            if grads_for_sync:
                device = next(model.parameters()).device
                flat_grads = _flatten_dense_tensors(
                    [grad.detach() for grad in grads_for_sync]
                ).to(device, non_blocking=True)
                dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
                flat_grads.div_(world_size)
                if offload_cpu:
                    flat_grads = flat_grads.to("cpu", non_blocking=True)
                averaged_grads = _unflatten_dense_tensors(
                    flat_grads, grads_for_sync
                )
                for grad, averaged in zip(grads_for_sync, averaged_grads):
                    grad.copy_(averaged)
                del flat_grads, averaged_grads
            sync_comm_time = time.time() - comm_start_sync

            # --- Outer Optimizer Step ---
            outer_optimizer.step()
            outer_optimizer.zero_grad(set_to_none=True)

            # --- Update Worker Model ---
            # Copy the globally updated parameters from original_model back to the worker model
            for param, original_param in zip(
                model.parameters(), original_model.parameters()
            ):
                param.data.copy_(original_param.data.to(param.device))
        else:
            # --- Direct Averaging Synchronization (No Outer Optimizer) ---
            comm_start_sync = time.time()
            # Directly average the parameters of the model across all workers
            for param in model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data.div_(world_size)
            sync_comm_time = time.time() - comm_start_sync

    # --- Communication Delay Simulation ---
    if comm_delay:
        logger.info(f"Simulating communication time: {comm_delay:.4f} seconds")
        # Override measured time with simulated delay
        sync_comm_time = comm_delay
    else:
        logger.info(f"Synchronization communication time: {sync_comm_time:.4f} seconds")

    return sync_comm_time
