import torch.nn as nn
import torch
import torch.distributed as dist


def get_layer_shards(model: nn.Module, num_shards: int, pattern: str = "sequential"):
    """
    按层将 model.parameters() 划分成 num_shards 份：
      - sequential: 连续切分
      - stride: 跳跃切分
    返回 List[List[nn.Parameter]]，长度 = num_shards
    """
    # 1. 把每一“层”按顺序收集成一个大列表
    layers = []
    # embed
    layers.append(list(model.model.embed_tokens.parameters()))
    # transformer 层
    for lyr in model.model.layers:
        layers.append(list(lyr.parameters()))
    # norm + lm_head
    layers.append(list(model.model.norm.parameters()))
    layers.append(list(model.lm_head.parameters()))
    # 2. 根据 pattern 计算每个 shard 包含哪些 layer idx
    L = len(layers)
    shards = [[] for _ in range(num_shards)]
    if pattern == "sequential":
        per = (L + num_shards - 1) // num_shards
        for i, layer_params in enumerate(layers):
            shard_id = min(i // per, num_shards - 1)
            shards[shard_id].extend(layer_params)
    elif pattern == "stride":
        for i, layer_params in enumerate(layers):
            shard_id = i % num_shards
            shards[shard_id].extend(layer_params)
    else:
        raise ValueError(f"Unknown pattern {pattern}")
    return shards, L

#CHANGE 定义select_next_shard函数
def select_next_shard(shard_tracker, H, K, current_global_step):
    # 1. 检查是否有过度陈旧的片段（I_p ≥ H）
    for shard_idx in range(K):
        t_p_b = shard_tracker[shard_idx]["sent_at_step"]
        I_p = current_global_step - t_p_b
        # print("1:-----------------------", I_p)
        if I_p >= H:
            return shard_idx # 优先选择过度陈旧的片段
    # 2. 选择 R_p 最大的片段
    #CHANGEN
    selected_idx = 0
    dict = {}
    for shard_idx in range(K):
        if shard_tracker[shard_idx]["sent_at_step"] == 0:
            return shard_idx # 优先选择还没发送过的片段
        cur_shard = shard_tracker[shard_idx]
        with torch.no_grad():
            sync_grads = [p_old.data - p_rec.data
                            for p_old, p_rec in zip(cur_shard["params"], [p.data.clone() for p in shard_tracker[shard_idx]["param_refs"]])]
        device = sync_grads[0].device
        total_norm_sq = torch.tensor(0.0, device=device, dtype=torch.float32)

        with torch.no_grad():
            for tensor in sync_grads:
                # 1. 计算单个张量的 L2 范数
                norm_of_tensor = tensor.norm(1, dtype=torch.float32)
                # 2. 平方并累加到总和中
                total_norm_sq += norm_of_tensor

        # 3. 最后对总和开方，得到最终的整体 L2 范数
        # .item() 将其从tensor转换为python浮点数
        # update_magnitude = torch.sqrt(total_norm_sq).item()
        update_magnitude = total_norm_sq.item()
        # print(f"-----------------------{current_global_step}|{cur_shard['sent_at_step']}")
        I_p = current_global_step - cur_shard["sent_at_step"]
        # print(f"分片{shard_idx + 1}|I_p{I_p}")
        current_R = update_magnitude * 10 ** 8 / (I_p * cur_shard["global_num_params"])
        dict[shard_idx] = current_R
        # print(f"分片{shard_idx + 1}|范数{current_R}")
        del sync_grads

    torch.distributed.barrier()
    dict_tensor = torch.tensor(
        [dict.get(i, -1.0) for i in range(K)],
        device=device,
        dtype=torch.float32
    )
    dist.all_reduce(dict_tensor, op=dist.ReduceOp.SUM)
    # flat_grads.div_(world_size) # flat_grads now holds averaged Δp
    # update_magnitude = sync_grads.norm(2).item()
    # print("Avg grad norms: ", dict_tensor)
    selected_idx = torch.argmax(dict_tensor).item()
    del dict_tensor
    # selected_idx = max(shard_tracker, key=lambda idx: shard_tracker[idx]['grad_norm'])
    return selected_idx

#CHANGE 定义函数计算参数数量
def print_shard_param_counts_from_shards(shards, logger):
    """
    shards: List[List[nn.Parameter]]
    get_layer_shards(...) 返回的第一个值（注意要解包）
    作用：对每个 shard 统计参数个数、字节数；在 TP 组内做 SUM，打印全局统计
    """
    if not shards:
        return
    # 用第一个参数的 device，兼容 CUDA/NPU
    dev = shards[0][0].device if shards[0] else torch.device("cpu")

    # tp_group = mpu.get_tensor_model_parallel_group()
    # tp_rank  = mpu.get_tensor_model_parallel_rank()
    # dp_rank  = mpu.get_data_parallel_rank()
    # pp_rank  = mpu.get_pipeline_model_parallel_rank()

    global_counts = []
    global_bytes  = []

    for shard in shards:
        # 本 rank 上该 shard 的参数个数/字节
        n_local = sum(p.numel() for p in shard)
        b_local = sum(p.numel() * p.element_size() for p in shard)

        n_t = torch.tensor([n_local], device=dev, dtype=torch.long)
        b_t = torch.tensor([b_local], device=dev, dtype=torch.long)

        # 在 TP 组内做 SUM 得到“全局（跨 TP 切分）”的数值
        # torch.distributed.all_reduce(n_t, group=tp_group)
        # torch.distributed.all_reduce(b_t, group=tp_group)

        global_counts.append(int(n_t.item()))
        global_bytes.append(int(b_t.item()))

    # 只在 (DP=0, TP=0, PP=0) 打印一次，避免刷屏
    # if dp_rank == 0 and tp_rank == 0 and pp_rank == 0:
    head = f"[Shard Stats] (global across TP)"
    logger.info(head)
    for i, (n, b) in enumerate(zip(global_counts, global_bytes)):
        mb = b / (1024**2)
        logger.info(f"  - Shard {i:02d}: params={n:,}  bytes={mb:,.2f} MB")
    return global_counts, global_bytes
