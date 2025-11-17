import os
import glob
import shutil
import threading
import queue
from typing import Dict, Any, Optional

import torch
from torch.cuda import device_memory_used
import torch.distributed as dist

"""
Async Checkpointing Module

Public Functions:
- save_checkpoint(...): Schedules an async checkpoint write.
- load_checkpoint(...): Loads the latest checkpoint for a rank.
- get_latest_checkpoint(...): Helper to fetch newest rank file.
- shutdown_async_checkpointing(): Flushes queue & joins worker thread (call before dist.destroy_process_group()).

Usage Pattern (typical in training loop):
    save_checkpoint(...)
At the end of training (rank 0 only or all ranks):
    shutdown_async_checkpointing()

NOTE: If the training process exits without calling shutdown_async_checkpointing(),
      pending checkpoints may be lost. Integrate shutdown in your training finalization logic.
"""

# ======================================================================
# Globals for async system
# ======================================================================
_ASYNC_QUEUE: Optional[queue.Queue] = None
_WORKER_THREAD: Optional[threading.Thread] = None
_ASYNC_INITIALIZED = False
_QUEUE_MAXSIZE = 2  # Small to bound memory usage
_STOP_SENTINEL = object()
_INIT_LOCK = threading.Lock()


def _init_async_if_needed(logger: Optional[Any]):
    """Lazy-init the async queue and worker thread."""
    global _ASYNC_INITIALIZED, _ASYNC_QUEUE, _WORKER_THREAD
    if _ASYNC_INITIALIZED:
        return
    with _INIT_LOCK:
        if _ASYNC_INITIALIZED:
            return
        _ASYNC_QUEUE = queue.Queue(maxsize=_QUEUE_MAXSIZE)

        def _worker():
            while True:
                item = _ASYNC_QUEUE.get()
                if item is _STOP_SENTINEL:
                    _ASYNC_QUEUE.task_done()
                    break
                try:
                    _write_checkpoint_payload(**item)
                except Exception as e:
                    if item.get("logger"):
                        item["logger"].error(f"Async checkpoint write failed: {e}", exc_info=True)
                finally:
                    _ASYNC_QUEUE.task_done()

        _WORKER_THREAD = threading.Thread(target=_worker, name="CheckpointWriter", daemon=True)
        _WORKER_THREAD.start()
        _ASYNC_INITIALIZED = True
        if logger:
            logger.info("Async checkpoint system initialized.")


def shutdown_async_checkpointing(logger: Optional[Any] = None, wait: bool = True):
    """Flush pending writes and stop background thread."""
    global _ASYNC_INITIALIZED, _ASYNC_QUEUE, _WORKER_THREAD
    if not _ASYNC_INITIALIZED:
        return
    if logger:
        logger.info("Shutting down async checkpointing...")
    _ASYNC_QUEUE.put(_STOP_SENTINEL)
    if wait:
        _ASYNC_QUEUE.join()
    if _WORKER_THREAD and _WORKER_THREAD.is_alive():
        _WORKER_THREAD.join(timeout=10)
    _ASYNC_INITIALIZED = False
    _ASYNC_QUEUE = None
    _WORKER_THREAD = None
    if logger:
        logger.info("Async checkpointing shutdown complete.")


# ======================================================================
# Core helpers
# ======================================================================

def get_latest_checkpoint(checkpoint_dir: str,
                          logger: Optional[Any] = None,
                          rank: int = 0) -> str:
    """Return path to newest checkpoint file for given rank, or '' if none."""
    if not os.path.isdir(checkpoint_dir):
        if logger:
            logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return ""
    pattern = os.path.join(checkpoint_dir, "step_*", f"rank_{rank}.pt")
    files = glob.glob(pattern)
    if not files:
        if logger:
            logger.info(f"No checkpoints found for rank {rank} in {checkpoint_dir}")
        return ""
    files.sort(key=os.path.getmtime, reverse=True)
    latest = files[0]
    if logger:
        logger.info(f"Latest checkpoint for rank {rank}: {latest}")
    return latest


def _clean_old_checkpoints(checkpoint_dir: str, max_keep: int, logger: Optional[Any]):
    """Remove old step_* directories, keeping only newest 'max_keep'."""
    if max_keep <= 0:
        return
    step_dirs = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    if len(step_dirs) <= max_keep:
        return
    step_dirs.sort(key=os.path.getmtime, reverse=True)
    to_remove = step_dirs[max_keep:]
    for d in to_remove:
        try:
            shutil.rmtree(d, ignore_errors=True)
            if logger:
                logger.info(f"Removed old checkpoint directory: {d}")
        except Exception as e:
            if logger:
                logger.error(f"Failed removing old checkpoint {d}: {e}", exc_info=True)


def _serialize_shard_tracker(shard_tracker: Dict[int, Dict[str, Any]],
                             save_model_only: bool,
                             include_old_sent: bool) -> Dict[int, Dict[str, Any]]:
    out = {}
    for idx, info in shard_tracker.items():
        shard = {
            "sent_at_step": info.get("sent_at_step", 0),
            "next_receive_step": info.get("next_receive_step", 0),
            **({"old_sent_at_step": info.get("old_sent_at_step", 0)} if include_old_sent else {}),
            "params": [p.detach().cpu().clone() for p in info["params"]],
            "staged_params": None if info.get("staged_params") is None else
                             [p.detach().cpu().clone() for p in info["staged_params"]],
        }
        if "global_num_params" in info:
            shard["global_num_params"] = info["global_num_params"]
        if "global_num_bytes" in info:
            shard["global_num_bytes"] = info["global_num_bytes"]
        if info.get("outer_optimizer") is not None and not save_model_only:
            shard["outer_optimizer_state_dict"] = info["outer_optimizer"].state_dict()
        out[idx] = shard
    return out


def _restore_shard_tracker(loaded: Dict[int, Dict[str, Any]],
                           shard_tracker: Dict[int, Dict[str, Any]],
                           include_old_sent: bool):
    device = next(iter(shard_tracker.values()))["params"][0].device
    for idx, shard in loaded.items():
        if idx not in shard_tracker:
            continue
        tgt = shard_tracker[idx]
        tgt["sent_at_step"] = shard["sent_at_step"]
        tgt["next_receive_step"] = shard["next_receive_step"]
        if include_old_sent:
            tgt["old_sent_at_step"] = shard.get("old_sent_at_step", 0)
        for i, cpu_t in enumerate(shard["params"]):
            tgt["params"][i].data.copy_(cpu_t.to(device))
        if shard["staged_params"] is not None:
            if tgt["staged_params"] is None:
                tgt["staged_params"] = [p.to(device).clone() for p in shard["staged_params"]]
            else:
                for i, cpu_t in enumerate(shard["staged_params"]):
                    tgt["staged_params"][i].data.copy_(cpu_t.to(device))
        if "outer_optimizer_state_dict" in shard and tgt.get("outer_optimizer") is not None:
            tgt["outer_optimizer"].load_state_dict(shard["outer_optimizer_state_dict"])


# ======================================================================
# Async write implementation
# ======================================================================

def _prepare_cpu_checkpoint_payload(
    algorithm: str,
    checkpoint_dir: str,
    rank: int,
    step_dir_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    original_snapshot: Optional[torch.nn.Module],
    outer_optimizer: Optional[torch.optim.Optimizer],
    shard_tracker: Optional[Dict[int, Dict[str, Any]]],
    epoch: int,
    global_step: int,
    micro_step: int,
    comp_time_total: float,
    comm_time_total: float,
    comm_vol_total: float,
    metric_value: Optional[float],
    is_best: bool,
    save_model_only: bool,
    max_checkpoints: int,
    logger: Optional[Any]
) -> Dict[str, Any]:
    """Create a fully CPU-offloaded dict representing the checkpoint."""
    # CPU state dict for model
    cpu_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    checkpoint = {
        "algorithm": algorithm,
        "epoch": epoch,
        "global_step": global_step,
        "micro_step": micro_step,
        "comp_time_total": comp_time_total,
        "comm_time_total": comm_time_total,
        "comm_vol_total": comm_vol_total,
        "model_state_dict": cpu_model_state,
        "rank": rank,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }

    if not save_model_only:
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if scaler:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

    if algorithm == "diloco":
        if original_snapshot is not None:
            checkpoint["original_snapshot_state_dict"] = {
                k: v.detach().cpu().clone() for k, v in original_snapshot.state_dict().items()
            }
        if outer_optimizer is not None and not save_model_only:
            checkpoint["outer_optimizer_state_dict"] = outer_optimizer.state_dict()
    elif algorithm == "streaming":
        if shard_tracker is not None:
            checkpoint["shard_tracker"] = _serialize_shard_tracker(
                shard_tracker, save_model_only, include_old_sent=False
            )
    elif algorithm == "dc":
        if shard_tracker is not None:
            checkpoint["shard_tracker"] = _serialize_shard_tracker(
                shard_tracker, save_model_only, include_old_sent=True
            )
    else:
        if logger:
            logger.error(f"Unknown algorithm '{algorithm}' during checkpoint serialization.")

    return {
        "checkpoint": checkpoint,
        "checkpoint_dir": checkpoint_dir,
        "step_dir_name": step_dir_name,
        "rank": rank,
        "global_step": global_step,
        "is_best": is_best,
        "metric_value": metric_value,
        "max_checkpoints": max_checkpoints,
        "algorithm": algorithm,
        "logger": logger,
        "save_model_only": save_model_only,
    }


def _write_checkpoint_payload(checkpoint: Dict[str, Any],
                              checkpoint_dir: str,
                              step_dir_name: str,
                              rank: int,
                              global_step: int,
                              is_best: bool,
                              metric_value: Optional[float],
                              max_checkpoints: int,
                              algorithm: str,
                              logger: Optional[Any],
                              save_model_only: bool):
    """Worker thread actual write function."""
    step_dir = os.path.join(checkpoint_dir, step_dir_name)
    os.makedirs(step_dir, exist_ok=True)
    file_path = os.path.join(step_dir, f"rank_{rank}.pt")

    torch.save(checkpoint, file_path)

    # Best model handling (only rank0 needs its own copy; other ranks optional)
    if is_best and rank == 0:
        meta_path = os.path.join(checkpoint_dir, "best_step.txt")
        with open(meta_path, "w") as f:
            f.write(f"step: {global_step}\nmetric: {metric_value if metric_value is not None else 'N/A'}\n")

    # Cleanup only by rank0 after its own save (no barrier needed)
    if rank == 0 and max_checkpoints > 0:
        _clean_old_checkpoints(checkpoint_dir, max_checkpoints, logger)

    if logger:
        logger.info(f"Async checkpoint saved: {file_path}")

# ======================================================================
# Public API: save & load
# ======================================================================

def save_checkpoint(
    algorithm: str,
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    original_snapshot: Optional[torch.nn.Module] = None,
    outer_optimizer: Optional[torch.optim.Optimizer] = None,
    shard_tracker: Optional[Dict[int, Dict[str, Any]]] = None,
    epoch: int = 0,
    global_step: int = 0,
    micro_step: int = 0,
    comp_time_total: float = 0.0,
    comm_time_total: float = 0.0,
    comm_vol_total: float = 0.0,
    metric_value: Optional[float] = None,
    is_best: bool = False,
    rank: int = 0,
    save_model_only: bool = False,
    max_checkpoints: int = 3,
    logger: Optional[Any] = None,
) -> str:
    """
    Schedule an asynchronous checkpoint save.

    Returns immediately with the intended file path; actual write occurs later.
    """
    _init_async_if_needed(logger)

    step_dir_name = f"step_{global_step}"
    intended_path = os.path.join(checkpoint_dir, step_dir_name, f"rank_{rank}.pt")

    payload = _prepare_cpu_checkpoint_payload(
        algorithm=algorithm,
        checkpoint_dir=checkpoint_dir,
        rank=rank,
        step_dir_name=step_dir_name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        original_snapshot=original_snapshot,
        outer_optimizer=outer_optimizer,
        shard_tracker=shard_tracker,
        epoch=epoch,
        global_step=global_step,
        micro_step=micro_step,
        comp_time_total=comp_time_total,
        comm_time_total=comm_time_total,
        comm_vol_total=comm_vol_total,
        metric_value=metric_value,
        is_best=is_best,
        save_model_only=save_model_only,
        max_checkpoints=max_checkpoints,
        logger=logger,
    )

    # Enqueue (may block if queue is full, which still is shorter than the full write time)
    try:
        _ASYNC_QUEUE.put(payload, block=True)
        if logger:
            logger.debug(f"Checkpoint for step {global_step} (rank {rank}) enqueued for async save.")
    except Exception as e:
        if logger:
            logger.error(f"Failed to enqueue checkpoint: {e}", exc_info=True)

    return intended_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    original_snapshot: Optional[torch.nn.Module] = None,
    outer_optimizer: Optional[torch.optim.Optimizer] = None,
    shard_tracker: Optional[Dict[int, Dict[str, Any]]] = None,
    rank: int = 0,
    map_location: str = "cpu",
    load_model_only: bool = False,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load the latest or specified checkpoint synchronously.
    If 'checkpoint_path' is a directory, loads the newest rank file there.
    Returns training state dict.
    """
    # If a directory was given, resolve the latest file for this rank.
    if os.path.isdir(checkpoint_path):
        resolved = get_latest_checkpoint(checkpoint_path, logger, rank)
        if not resolved:
            return {}
        checkpoint_file = resolved
    else:
        checkpoint_file = checkpoint_path

    if not os.path.exists(checkpoint_file):
        if logger:
            logger.error(f"Checkpoint file not found: {checkpoint_file}")
        return {}

    if logger:
        logger.info(f"Rank {rank}: Loading checkpoint: {checkpoint_file}")

    ckpt = torch.load(checkpoint_file, map_location=map_location)
    algorithm = ckpt.get("algorithm", "unknown")
    if logger:
        logger.info(f"Detected algorithm in checkpoint: {algorithm}")

    # Restore model first
    model.load_state_dict(ckpt["model_state_dict"])

    if not load_model_only:
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    # RNG states
    rng_states = ckpt.get("rng_states")
    if rng_states:
        torch_state = rng_states.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch.tensor(torch_state, dtype=torch.uint8, device='cpu'))
        if torch.cuda.is_available():
            cuda_states = rng_states.get("cuda")
            if cuda_states is not None:
                cuda_list = [torch.tensor(s, dtype=torch.uint8, device='cpu') for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_list)

    # Algorithm-specific
    if algorithm == "diloco":
        if original_snapshot and "original_snapshot_state_dict" in ckpt:
            original_snapshot.load_state_dict(ckpt["original_snapshot_state_dict"])
        if outer_optimizer and "outer_optimizer_state_dict" in ckpt and not load_model_only:
            outer_optimizer.load_state_dict(ckpt["outer_optimizer_state_dict"])
    elif algorithm == "streaming":
        if shard_tracker and "shard_tracker" in ckpt:
            _restore_shard_tracker(ckpt["shard_tracker"], shard_tracker, include_old_sent=False)
    elif algorithm == "dc":
        if shard_tracker and "shard_tracker" in ckpt:
            _restore_shard_tracker(ckpt["shard_tracker"], shard_tracker, include_old_sent=True)

    training_state = {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "micro_step": ckpt.get("micro_step", 0),
        "comp_time_total": ckpt.get("comp_time_total", 0.0),
        "comm_time_total": ckpt.get("comm_time_total", 0.0),
        "comm_vol_total": ckpt.get("comm_vol_total", 0.0),
        "algorithm": algorithm,
    }
    if rank == 0:
        training_state["metric_value"] = ckpt.get("metric_value")

    if logger:
        logger.info(f"Rank {rank}: Loaded epoch={training_state['epoch']} step={training_state['global_step']}")

    return training_state
