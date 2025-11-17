import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from .streaming import init_streaming_state


def init_dc_state(model: torch.nn.Module, args: Any, logger: Any) -> Dict[str, Any]:
    """
    Initialize state for the 'dc' algorithm (DC-DiLoCo).

    This extends the streaming initialization with:
      - 'old_sent_at_step' per shard
      - 'h' = sync_interval // N (dynamic send frequency)

    Returns:
        {
            "shard_tracker": Dict[int, Dict[str, Any]],
            "num_layers": int,
            "h": int,
        }
    """
    if logger:
        logger.info(
            f"Initializing state for 'dc' algorithm with {args.num_shards} shards (pattern={args.pattern})."
        )

    # Reuse streaming layout (shard_tracker with outer_optimizer, stats)
    streaming_state = init_streaming_state(model, args, logger)
    shard_tracker = streaming_state["shard_tracker"]
    num_layers = streaming_state["num_layers"]

    # Compute per-interval dynamic send frequency
    N = args.N  # expected maximum number of transmissions within one sync_interval
    h = args.sync_interval // N

    # Add DC-specific fields
    for idx in shard_tracker.keys():
        shard_tracker[idx]["old_sent_at_step"] = 0

    if logger:
        logger.info(f"DC 参数: N={N}, h={h} (每 {h} 步选择一次分片发送)")

    return {
        "shard_tracker": shard_tracker,
        "num_layers": num_layers,
        "h": h,
    }


def sync_dc_diloco(
    model,
    shard_tracker,
    sync_shard_idx,
    world_size,
    logger,
    comm_delay,
    num_shards,
    alpha,
    dc_lambda,
):
    """
    DC-DiLoCo 分块同步与延迟补偿逻辑。

    Args:
        model: 当前工作模型
        shard_tracker: 分片跟踪器字典，包含：
            - params: θ(t_rec - H)_m,p （接收前的全局基值或外优化后的基值）
            - staged_params: θ(t_rec)_m,p （发送时刻延迟步数前的模型快照）
            - param_refs: 指向当前本地模型中对应分片参数的引用
            - sent_at_step: 上一次发送的步数
            - old_sent_at_step: 上上一次发送的步数（用于接收间隔计算）
            - next_receive_step: 预计接收步数
            - outer_optimizer: 外层优化器（可选）
            - global_num_params / global_num_bytes: 分片统计
        sync_shard_idx: 要同步的分片索引
        world_size: 总进程数
        logger: 日志记录器
        comm_delay: 模拟通信延迟（秒），None 表示真实时间
        num_shards: 分片总数
        alpha: 本地与全局融合权重（local = alpha*local + (1-alpha)*global）
        dc_lambda: 延迟补偿基础系数（后续动态调整 lambda_i）

    Returns:
        本次同步的通信耗时（或模拟延迟时间）
    """
    print("--------------------cdiloco is executing-------------------------")
    cur_shard = shard_tracker[sync_shard_idx]
    param_refs = cur_shard["param_refs"]

    # --- 1. 计算外层梯度近似增量 Δm,p = θ(t_rec - H)_m,p - θ(t_rec)_m,p ---
    with torch.no_grad():
        offload_cpu = cur_shard["params"][0].device.type == "cpu"
        sync_grads = []
        for p_old, p_rec in zip(cur_shard["params"], cur_shard["staged_params"]):
            if offload_cpu and p_rec.device.type != "cpu":
                p_rec_cpu = p_rec.detach().to("cpu", copy=True)
            else:
                p_rec_cpu = p_rec
            g = (p_old.data if hasattr(p_old, "data") else p_old).clone()
            g.sub_(p_rec_cpu.data if hasattr(p_rec_cpu, "data") else p_rec_cpu)
            sync_grads.append(g)

    # --- 2. 通信聚合 Δp = (1/M) * Σ Δm,p ---
    comm_start = time.time()
    device = next(model.parameters()).device
    if cur_shard["params"][0].device.type == "cpu":
        flat_cpu = _flatten_dense_tensors([g.detach() for g in sync_grads])
        flat_gpu = flat_cpu.to(device, non_blocking=True)
        dist.all_reduce(flat_gpu, op=dist.ReduceOp.SUM)
        flat_gpu.div_(world_size)
        averaged = _unflatten_dense_tensors(flat_gpu.cpu(), sync_grads)
        for dst, src in zip(sync_grads, averaged):
            dst.copy_(src)
        del flat_cpu, flat_gpu, averaged
    else:
        flat_gpu = _flatten_dense_tensors([g.detach() for g in sync_grads])
        dist.all_reduce(flat_gpu, op=dist.ReduceOp.SUM)
        flat_gpu.div_(world_size)
        averaged = _unflatten_dense_tensors(flat_gpu, sync_grads)
        for dst, src in zip(sync_grads, averaged):
            dst.copy_(src)
        del flat_gpu, averaged
    comm_time = time.time() - comm_start

    # --- 3. 外层优化或简单平均 (SGD outer step / averaging) ---
    if cur_shard["outer_optimizer"]:
        cur_shard["outer_optimizer"].zero_grad()
        for param, avg_delta in zip(cur_shard["params"], sync_grads):
            if param.grad is None or (hasattr(param, "grad") and param.grad is not None and param.grad.shape != param.data.shape):
                param.grad = torch.zeros_like(param.data)
            param.grad.copy_(avg_delta)
        cur_shard["outer_optimizer"].step()
    else:
        with torch.no_grad():
            for param, avg_delta in zip(cur_shard["params"], sync_grads):
                param.data.sub_(avg_delta.data)
    # 注意：此时 cur_shard['params'] 是外层更新后的参考基值 θ_outer

    receive_interval = cur_shard["sent_at_step"] - cur_shard["old_sent_at_step"]
    delay_steps = cur_shard["next_receive_step"] - cur_shard["sent_at_step"]

    # --- 4. 构造局部更新估计 g_1 与偏差 D ---
    # g_1: (staged_params - local_params) （原注释里曾尝试除以 delay_steps）
    current_local_params = param_refs
    offload_cpu = cur_shard["params"][0].device.type == "cpu"
    if offload_cpu:
        local_list = [p.detach().to("cpu", copy=True) for p in current_local_params]
        staged_list = [p.detach().to("cpu", copy=True) for p in cur_shard["staged_params"]]
        global_list = cur_shard["params"]
    else:
        local_list = [p.data for p in current_local_params]
        staged_list = [p.data for p in cur_shard["staged_params"]]
        global_list = [p.data for p in cur_shard["params"]]
    g_1 = [s.clone().sub_(l) for s, l in zip(staged_list, local_list)]
    # D: (global_params - staged_params)
    D = [g.clone().sub_(s) for g, s in zip(global_list, staged_list)]

    # --- 5. 动态延迟补偿 (Hadamard 修正) ---
    g_1_corrected = []
    dc_lambda_base = dc_lambda
    epsilon = 1e-8

    for g1, d in zip(g_1, D):
        # 分子: λ₀ * ||g₁||
        numerator = dc_lambda_base * torch.norm(g1)
        # 近似二阶项 (g1 * g1 * d) 做尺度标准化（原代码硬编码除以 4e-4）
        correction_term = (g1 * g1 * d) / 4e-4
        denominator = torch.norm(correction_term)
        dynamic_lambda = numerator / (denominator + epsilon)
        g_1_corrected.append(g1 + (dynamic_lambda * correction_term))
        logger.info(
            f"分片 {sync_shard_idx + 1} 参数梯度校正: "
            f"||g1||={torch.norm(g1):.10f}, "
            f"||g1g1D||={torch.norm(correction_term):.10f}, "
            f"lambda_base={dc_lambda_base:.10f}, dynamic_lambda={dynamic_lambda:.10f}"
        )

    del D, g_1  # 释放无用中间列表

    # --- 6. 应用校正 (目前未乘 delay_steps，保持与现实现一致) ---
    with torch.no_grad():
        for global_p, g_corr, local_p in zip(
            cur_shard["params"], g_1_corrected, param_refs
        ):
            gp = global_p.data if hasattr(global_p, "data") else global_p
            local_p.data.copy_((gp - g_corr.to(gp.device) if hasattr(g_corr, "to") else gp - g_corr).to(local_p.device))

    # --- 7. 准备下一轮状态 ---
    with torch.no_grad():
        cur_shard["staged_params"] = None

    # --- 8. 返回通信耗时或模拟值 ---
    if comm_delay:
        actual_delay = comm_delay / num_shards  # 平均分摊
        logger.info(f"分片 {sync_shard_idx + 1} 模拟通信时间: {actual_delay:.4f} 秒")
        return actual_delay
    else:
        logger.info(
            f"分片 {sync_shard_idx + 1} 通信时间 (all-reduce): {comm_time:.4f} 秒"
        )
        return comm_time
