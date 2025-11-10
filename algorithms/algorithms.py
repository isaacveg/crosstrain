import time
import torch
import torch.distributed as dist


def sync_diloco(model, original_model, outer_optimizer, world_size, logger, comm_delay=None):
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
    # print("--------------------diloco is executing-------------------------")
    sync_comm_time = 0.0
    with torch.no_grad():
        if outer_optimizer:
            # --- DiLoCo-like Synchronization ---
            grads_for_sync = []
            # Calculate the effective gradient (update direction) for the outer step
            for param, original_param in zip(model.parameters(), original_model.parameters()):
                # grad = original_param.data - param.data # Original DiLoCo direction
                grad_update = original_param.data - param.data
                # Assign this difference to the .grad field of the parameters
                # in original_model so the outer_optimizer can use it.
                # Ensure the grad attribute exists and is compatible
                if original_param.grad is None:
                    original_param.grad = torch.zeros_like(original_param.data)
                original_param.grad.copy_(grad_update)
                grads_for_sync.append(original_param.grad.data) # Collect grads for all-reduce

            # --- Batch Communication ---
            comm_start_sync = time.time()
            if grads_for_sync: # Proceed only if there are gradients to sync
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
                    # Ensure grad_sync is used to determine the view shape
                    grad_sync.copy_(flat_grads[offset:offset+numel].view_as(grad_sync))
                    offset += numel
            sync_comm_time = time.time() - comm_start_sync

            # --- Outer Optimizer Step ---
            outer_optimizer.step()
            outer_optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for potential memory savings

            # --- Update Worker Model ---
            # Copy the globally updated parameters from original_model back to the worker model
            for param, original_param in zip(model.parameters(), original_model.parameters()):
                param.data.copy_(original_param.data)

        else:
            # --- Direct Averaging Synchronization (No Outer Optimizer) ---
            comm_start_sync = time.time()
            # Directly average the parameters of the model across all workers
            # Note: With AMP, model parameters might be FP16. Averaging FP16 directly
            # can lead to precision loss. A more robust approach might involve
            # casting to FP32 before all-reduce and back, or synchronizing FP32
            # master weights if the optimizer maintains them (AdamW does).
            # For simplicity, we keep the direct FP16 averaging here.
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


def sync_streaming_diloco(model, shard_tracker, sync_shard_idx, world_size, logger,
                         comm_delay, num_shards, alpha):
    """
    分块同步模型参数 (Corrected Logic)
    model: 当前工作模型 (state at t_now)
    shard_tracker: 包含旧状态的字典
      - params: state at t_rec - H (base for outer gradient)
      - staged_params: state at t_rec (recorded delay_steps ago)
    """
    # print("--------------------sdiloco is executing-------------------------")
    cur_shard = shard_tracker[sync_shard_idx]

    # --- 1. Calculate Outer Gradient Delta: Δm,p = θ(t_rec - H)_m,p - θ(t_rec)_m,p ---
    with torch.no_grad():
        sync_grads = [p_old.data - p_rec.data
                      for p_old, p_rec in zip(cur_shard["params"], cur_shard["staged_params"])]

    # --- 2. Communicate and Average Delta: Δp = (1/M) * Σ Δm,p ---
    comm_start = time.time()
    # Batch communication - flatten gradients for this shard
    try:
        flat_grads = torch.cat([g.flatten() for g in sync_grads])
    except RuntimeError as e:
        logger.error(f"Error flattening gradients for shard {sync_shard_idx}: {e}")
        for i, g in enumerate(sync_grads):
            logger.error(f"  Grad {i} shape: {g.shape}, dtype: {g.dtype}, device: {g.device}")
        return 0.0 # Skip if flattening fails

    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    flat_grads.div_(world_size) # flat_grads now holds averaged Δp

    # Unflatten the averaged delta back into sync_grads list structure
    # sync_grads will now hold the averaged Δp, structured like the parameters
    offset = 0
    for grad in sync_grads:
        numel = grad.numel()
        if offset + numel > flat_grads.numel():
             logger.error(f"Error unflattening: offset {offset} + numel {numel} > flat_grads size {flat_grads.numel()}")
             return 0.0 # Skip if unflattening calculation is wrong
        grad.copy_(flat_grads[offset:offset + numel].view_as(grad))
        offset += numel
    comm_time = time.time() - comm_start
    del flat_grads # Free memory

    # --- 3. Apply Outer Optimization: θ_outer = OuterOpt(θ(t_rec - H)_p, Δp) ---
    if cur_shard['outer_optimizer']:
        cur_shard['outer_optimizer'].zero_grad() # Clean up gradients
        for param, avg_delta in zip(cur_shard["params"], sync_grads):
            if param.grad is None:
                param.grad = avg_delta.clone() # Create grad buffer if needed
            else:
                param.grad.copy_(avg_delta)
        # Perform the optimizer step (updates cur_shard["params"])
        cur_shard['outer_optimizer'].step()
    else: # Equivalent to simple averaging (outer_lr=1.0 means SGD with lr=1.0)
        with torch.no_grad():
            for param, avg_delta in zip(cur_shard["params"], sync_grads):
                param.data.sub_(avg_delta.data)
        # cur_shard["params"] now holds θ_outer(t_now)_p
    del sync_grads # Free memory associated with the averaged delta list

    # --- 4. Merge: θ(t_now)_m,p = α*θ(t_now)_m,p + (1-α)*θ_outer ---
    globally_updated = cur_shard["params"]
    param_refs = cur_shard["param_refs"]
    with torch.no_grad():
        for local_p, updated_p in zip(param_refs, globally_updated):
            local_p.data.mul_(alpha).add_(updated_p.data, alpha=1 - alpha)


    # --- 5. Prepare State for Next Cycle ---
    with torch.no_grad():
        # for param, staged_param in zip(cur_shard["params"], cur_shard["staged_params"]):
        #     param.data.copy_(staged_param.data)
        cur_shard["staged_params"] = None # Optional memory optimization

    # --- Logging and Return ---
    if comm_delay:
        # Simulate delay based on config, dividing total delay by shards for average effect
        actual_delay = comm_delay / num_shards
        logger.info(f"分片 {sync_shard_idx+1} 模拟通信时间: {actual_delay:.4f} 秒")
        return actual_delay
    else:
        # Return measured all-reduce time
        logger.info(f"分片 {sync_shard_idx+1} 通信时间 (all-reduce): {comm_time:.4f} 秒")
        return comm_time
    

def sync_dc_diloco(model, shard_tracker, sync_shard_idx, world_size, logger,
                         comm_delay, num_shards, alpha, dc_lambda):
    """
    分块同步模型参数 (Corrected Logic)
    model: 当前工作模型 (state at t_now)
    shard_tracker: 包含旧状态的字典
      - params: state at t_rec - H (base for outer gradient)
      - staged_params: state at t_rec (recorded delay_steps ago)
    """
    # print("--------------------cdiloco is executing-------------------------")
    cur_shard = shard_tracker[sync_shard_idx]
    param_refs = cur_shard["param_refs"]

    # --- 1. Calculate Outer Gradient Delta: Δm,p = θ(t_rec - H)_m,p - θ(t_rec)_m,p ---
    # 在 DC-S3GD 中并不是真正意义上的 grads 而是类似变化量的东西。这里先获取最新的参数
    with torch.no_grad():
        sync_grads = [p_old.data - p_rec.data
                      for p_old, p_rec in zip(cur_shard["params"], cur_shard["staged_params"])]

    # --- 2. Communicate and Average Delta: Δp = (1/M) * Σ Δm,p ---
    comm_start = time.time()
    # Batch communication - flatten gradients for this shard
    try:
        flat_grads = torch.cat([g.flatten() for g in sync_grads])
    except RuntimeError as e:
        logger.error(f"Error flattening gradients for shard {sync_shard_idx}: {e}")
        for i, g in enumerate(sync_grads):
            logger.error(f"  Grad {i} shape: {g.shape}, dtype: {g.dtype}, device: {g.device}")
        return 0.0 # Skip if flattening fails

    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    flat_grads.div_(world_size) # flat_grads now holds averaged Δp

    #CHANGE
    # grad_norm = torch.norm(flat_grads, p=2).item()
    # I_p = cur_shard['sent_at_step'] - cur_shard['old_sent_at_step']
    # normalized_grad_norm = (grad_norm * 1000) / (I_p * cur_shard['global_num_params'])
    # shard_tracker[sync_shard_idx]['grad_norm'] = normalized_grad_norm
    # logger.info(f"发送步数 {cur_shard['sent_at_step']} 分片 {sync_shard_idx+1} 的归一化梯度范数: {normalized_grad_norm:.6f} (I_p={I_p})")
    # Unflatten the averaged delta back into sync_grads list structure
    # sync_grads will now hold the averaged Δp, structured like the parameters
    offset = 0
    for grad in sync_grads:
        numel = grad.numel()
        if offset + numel > flat_grads.numel():
             logger.error(f"Error unflattening: offset {offset} + numel {numel} > flat_grads size {flat_grads.numel()}")
             return 0.0 # Skip if unflattening calculation is wrong
        grad.copy_(flat_grads[offset:offset + numel].view_as(grad))
        offset += numel
    comm_time = time.time() - comm_start
    del flat_grads # Free memory

    # --- 3. Apply Outer Optimization: θ_outer = OuterOpt(θ(t_rec - H)_p, Δp) ---
    if cur_shard['outer_optimizer']:
        cur_shard['outer_optimizer'].zero_grad() # Clean up gradients
        for param, avg_delta in zip(cur_shard["params"], sync_grads):
            if param.grad is None:
                param.grad = avg_delta.clone() # Create grad buffer if needed
            else:
                param.grad.copy_(avg_delta)
        # Perform the optimizer step (updates cur_shard["params"])
        cur_shard['outer_optimizer'].step()
    else: # Equivalent to simple averaging (outer_lr=1.0 means SGD with lr=1.0)
        with torch.no_grad():
            for param, avg_delta in zip(cur_shard["params"], sync_grads):
                param.data.sub_(avg_delta.data)
        # cur_shard["params"] now holds θ_outer(t_now)_p
    # del sync_grads # Free memory associated with the averaged delta list
    # 现在拥有：model里有 θ(t_now)_m,p，cur_shard['params']里有 global θ(t_{now-tau})_m,p
    # cur_shard['staged_params']里是θ(t_{now-tau})_m,p

    receive_interval = cur_shard['sent_at_step'] - cur_shard['old_sent_at_step']
    delay_steps = cur_shard['next_receive_step'] - cur_shard['sent_at_step']
    # 计算平均伪外梯度 g_1，为 tau 步之内的更新除以 tau 步，当前的model和staged params之间的差值
    # Correct way to calculate g_1
    current_local_params = param_refs
    # g_1 = [(staged_p.data - local_p.data) / delay_steps 
    #     for local_p, staged_p in zip(current_local_params, cur_shard['staged_params'])]
    g_1 = [(staged_p.data - local_p.data)
        for local_p, staged_p in zip(current_local_params, cur_shard['staged_params'])]
    # 计算模型平均更新差值 D，为 params 和 staged_params 之间的差值，可以认为是全局和本地之间出现的偏差
    # D = (cur_shard['params'] - cur_shard['staged_params']) / receive_interval
    # D = [(global_p.data - local_p.data) / receive_interval 
    #      for global_p, local_p in zip(cur_shard['params'], cur_shard['staged_params'])]
    D = [(global_p.data - local_p.data)
         for global_p, local_p in zip(cur_shard['params'], cur_shard['staged_params'])]
    # 利用 DC-S3GD 和 DC-ASGD 的公式进行梯度修正（修正g_1）
    # g_1_corrected = g_1 + lmbd * (g_1 Hadamard g_1) * D，lmbd * g_1 Hadamard g_1 是海森矩阵的廉价估计
    g_1_corrected = []
    #CHANGEN
    # 固定的 dc_lambda 将被替换为在循环内进行的动态计算
    # 函数参数 'dc_lambda' 应重命名为 'dc_lambda_base' 以代表 λ₀
    dc_lambda_base = dc_lambda # 假设您后续会重命名函数参数。论文建议 dc_lambda_base = 0.2 
    epsilon = 1e-8 # 用于保证数值稳定性

    for g1, d in zip(g_1, D):
        # 这个代码块实现了论文中的公式 (17) 
        
        # 分子: λ₀ * ||gᵢ||
        numerator = dc_lambda_base * torch.norm(g1)
        
        # 分母: ||gᵢ ⊙ gᵢ ⊙ Dᵢ||
        correction_term = (g1 * g1 * d) / 4e-4
        denominator = torch.norm(correction_term)
        
        
        # 计算动态lambda: λᵢ = 分子 / (分母 + ε)
        dynamic_lambda = numerator / (denominator + epsilon)
        
        # 使用新计算出的 dynamic_lambda 应用校正
        # g_1_corrected.append(g1 + (dynamic_lambda * correction_term / (learning_rate + epsilon))) 
        # 这个除以ita不太行
        g_1_corrected.append(g1 + (dynamic_lambda * correction_term))   
        # logger.info(f"分片 {sync_shard_idx+1} 参数梯度校正: ||g1||={torch.norm(g1):.10f}, "
        #              f"||g1g1D||={torch.norm(correction_term):.10f}, "
        #              f"lambda={dc_lambda_base:.10f}, {dynamic_lambda:.10f}") 
        
    del D, g_1 # Free memory

    # for g1, d in zip(g_1, D):
    #     g_1_corrected.append(g1 + (dc_lambda * (g1 * g1) * d))
    # del D, g_1 # Free memory
    
    # 原本的公式：w = w + D + ∆w，对应到我们这里，w 是 staged_params，w+D其实就是params，
    # g_1_corrected 应该乘 tau 回去应用到更新上
    # w_corrected = cur_shard['params'] + (g_1_corrected * delay_steps)
    # with torch.no_grad():
    #     for global_p, g_corr, local_p in zip(
    #             cur_shard['params'], g_1_corrected, param_refs):
    #         local_p.data.copy_(global_p.data + g_corr * delay_steps)
    with torch.no_grad():
        for global_p, g_corr, local_p in zip(
                cur_shard['params'], g_1_corrected, param_refs):
            # local_p.data.copy_(global_p.data - g_corr * delay_steps)
            local_p.data.copy_(global_p.data - g_corr)

    # --- 5. Prepare State for Next Cycle ---
    with torch.no_grad():
        # for param, staged_param in zip(cur_shard["params"], cur_shard["staged_params"]):
        #     param.data.copy_(staged_param.data)
        cur_shard["staged_params"] = None # Optional memory optimization

    # --- Logging and Return ---
    if comm_delay:
        # Simulate delay based on config, dividing total delay by shards for average effect
        actual_delay = comm_delay / num_shards
        logger.info(f"分片 {sync_shard_idx+1} 模拟通信时间: {actual_delay:.4f} 秒")
        return actual_delay
    else:
        # Return measured all-reduce time
        logger.info(f"分片 {sync_shard_idx+1} 通信时间 (all-reduce): {comm_time:.4f} 秒")
        return comm_time