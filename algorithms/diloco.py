import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
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

    original_snapshot = deepcopy(model)
    if args.outer_lr != 1.0:
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
    model, original_model, outer_optimizer, world_size, logger, comm_delay=None
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
    with torch.no_grad():
        if outer_optimizer:
            # --- DiLoCo-like Synchronization ---
            grads_for_sync = []
            # Calculate the effective gradient (update direction) for the outer step
            for param, original_param in zip(
                model.parameters(), original_model.parameters()
            ):
                grad_update = original_param.data - param.data
                # Assign this difference to the .grad field of the parameters
                # in original_model so the outer_optimizer can use it.
                if original_param.grad is None:
                    original_param.grad = torch.zeros_like(original_param.data)
                original_param.grad.copy_(grad_update)
                grads_for_sync.append(
                    original_param.grad.data
                )  # Collect grads for all-reduce

            # --- Batch Communication ---
            comm_start_sync = time.time()
            if grads_for_sync:  # Proceed only if there are gradients to sync
                # Flatten all gradients into a single tensor for efficient all-reduce
                flat_grads = torch.cat([g.flatten() for g in grads_for_sync])
                # Aggregate gradients across all workers
                dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
                # Average the gradients
                flat_grads.div_(world_size)

                # --- Unflatten Gradients ---
                # Copy the averaged gradients back to the original_model's .grad attributes
                offset = 0
                for grad_sync in grads_for_sync:
                    numel = grad_sync.numel()
                    grad_sync.copy_(
                        flat_grads[offset : offset + numel].view_as(grad_sync)
                    )
                    offset += numel
            sync_comm_time = time.time() - comm_start_sync

            # --- Outer Optimizer Step ---
            outer_optimizer.step()
            outer_optimizer.zero_grad(
                set_to_none=True
            )  # Use set_to_none=True for potential memory savings

            # --- Update Worker Model ---
            # Copy the globally updated parameters from original_model back to the worker model
            for param, original_param in zip(
                model.parameters(), original_model.parameters()
            ):
                # Preserve original semantics: copy updated snapshot params back to the worker model
                param.data.copy_(original_param.data)

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
