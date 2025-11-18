import os
import time
import logging
import random
import copy
import torch
import torch.distributed as dist
import tqdm
from torch.amp import autocast
import datetime


def approx_hessian(grads):
    """
    用梯度的平方近似Hessian矩阵对角线元素。
    接收梯度列表并返回每个参数的Hessian近似值列表。
    """
    return [(g.clone().detach() ** 2) for g in grads]  # 对每个梯度张量计算平方

def get_temp_grad(w_t, batch, device, task_type, use_amp=True): # 添加 device, task_type, use_amp 参数
    """
    计算模型参数 w_t 在 batch 上的梯度，不进行反向传播更新。
    适配了 AMP，通过 use_amp 控制是否启用 autocast。

    Args:
        w_t: 模型实例。
        batch: 输入数据批次。
        device: 计算设备。
        task_type: 任务类型 ('classification' 或 'language_modeling')。
        use_amp: 是否启用 AMP (autocast)。

    Returns:
        梯度列表 (每个元素对应模型的一个参数)。
    """
    # 将 batch 数据移动到指定设备
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    # 确保模型处于评估模式（如果需要，取决于具体用途，但通常计算临时梯度时不希望BN等层更新状态）
    original_mode = w_t.training
    w_t.eval() # 切换到评估模式

    try:
        # 使用 autocast 上下文执行前向传播和损失计算
        with autocast(enabled=use_amp):
            if task_type == "language_modeling":
                outputs = w_t(**batch)
            elif task_type == "classification":
                outputs = w_t(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"],
                              labels=batch["label"])
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
            # 损失计算也应在 autocast 内
            loss = outputs.loss

        # 计算梯度，设置 create_graph=False 因为我们不需要二阶梯度
        # retain_graph=False (默认) 因为我们只计算一次梯度，不保留计算图
        grads = torch.autograd.grad(loss, w_t.parameters(), create_graph=False)

    finally:
        # 恢复模型原始的训练模式状态
        w_t.train(original_mode)

    # 返回分离的梯度副本，确保不影响外部计算图
    return [g.detach().clone() for g in grads]

def init_distributed():
    """
    初始化分布式环境，使用 NCCL 后端 适用于单机多 GPU 。
    环境变量 LOCAL_RANK 必须通过启动命令传入。
    返回 local_rank, 当前进程 rank, 总进程数 world_size。
    """
    random.seed(2025)   # 设置随机种子
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return local_rank, rank, world_size

def setup_logging(rank, log_dir):
    """
    配置日志：每个进程记录到独立文件，同时输出到控制台。
    日志存储到 log_dir 目录下
    """
    # 创建目录结构
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("NodeLogger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f"%(asctime)s - Rank {rank} - %(levelname)s - %(message)s")

    # 确保处理程序不会重复添加
    if logger.handlers:
        logger.handlers = []

    # 文件日志
    log_path = os.path.join(log_dir, f"node_{rank}.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台日志
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 记录运行配置
    if rank == 0:
        config_path = os.path.join(log_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(f"Run dir: {log_dir}\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not specified')}\n")

    return logger

def calc_comm_delay(model, world_size, logger, simulated_bandwidth_mbps=None):
    """
    计算模型参数同步的通信时间。
    """
    if simulated_bandwidth_mbps is None:
        return None
    # 计算参数总量（字节）
    total_params_bytes = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32

    # 计算理论上的通信量（字节）: 2*(N-1)/N * 参数总量
    comm_volume_bytes = 2 * (world_size - 1) / world_size * total_params_bytes

    # 转换为MB
    comm_volume_mb = comm_volume_bytes / (1024 * 1024)

    # 模拟的带宽延迟（秒）= 数据量(MB) / 带宽(MB/s)
    simulated_delay = comm_volume_mb / (simulated_bandwidth_mbps / 8)  # 除以8将Mbps转换为MB/s

    logger.info(f"模型大小: {total_params_bytes/1024/1024:.2f}MB, Ring All Reduce通信量: {comm_volume_mb:.2f}MB")
    logger.info(f"模拟 {simulated_bandwidth_mbps}Mbps 带宽，延迟 {simulated_delay:.2f}秒")
    return simulated_delay

def evaluate(model, eval_dataloader, device, task_type,
             use_amp=True, amp_type=torch.bfloat16, eval_batch_size=None, max_eval_batches=None):
    """在验证集上评估模型性能，适配 AMP。

    Args:
        model: 要评估的模型。
        eval_dataloader: 验证数据加载器。
        device: 计算设备。
        task_type: 任务类型 ("classification" 或 "language_modeling")。
        use_amp: 是否在评估时启用 autocast (推荐)。
        max_eval_batches: 评估的最大批次数 (用于流式或大型数据集)。

    Returns:
        包含评估指标的字典 (例如: {"loss": ..., "accuracy": ...} 或 {"loss": ..., "perplexity": ...})。
    """
    model.eval() # 切换到评估模式
    total_loss = 0.0
    total_samples = 0 # 用于分类任务
    total_tokens = 0  # 用于语言模型任务
    correct_predictions = 0 # 用于分类任务

    # 确定迭代次数
    try:
        # 尝试获取数据加载器长度
        num_batches = len(eval_dataloader)
        if max_eval_batches is not None and max_eval_batches <= num_batches:
            num_batches = max_eval_batches
            print(f"评估将限制在 {max_eval_batches} 个批次。")
    except TypeError:
        # 如果是 IterableDataset (流式)，长度未知
        if max_eval_batches is None:
            print("警告: 评估数据集是流式的且未指定 max_eval_batches，评估可能不会终止。")
            # 可以设置一个默认值或引发错误
            num_batches = 50 # 设置一个默认的最大批次数以防万一
            print(f"将默认评估最多 {num_batches} 个批次。")
        else:
            num_batches = max_eval_batches
            print(f"评估流式数据集，限制在 {num_batches} 个批次。")

    eval_start_time = time.time()
    # 使用 tqdm 创建进度条
    pbar = tqdm.tqdm(total=num_batches, desc="评估中", unit="batch", leave=False)

    # 使用 inference_mode 进一步降低评估显存占用
    torch.cuda.empty_cache()
    with torch.inference_mode():
        batch_count = 0
        for batch in eval_dataloader:
            if batch_count >= num_batches:
                break # 达到最大批次数限制

            # 将数据移动到设备
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # --- 使用 autocast 进行前向传播和损失计算 ---
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_type):
                if task_type == "language_modeling":
                    outputs = model(**batch)
                    # LM 通常在模型内部计算损失，假设 batch 包含 labels
                elif task_type == "classification":
                    outputs = model(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   labels=batch["label"])
                else:
                    raise ValueError(f"未知的任务类型: {task_type}")

                loss = outputs.loss # 获取损失

            # 累积损失和指标
            batch_size = batch["input_ids"].size(0)
            if task_type == "language_modeling":
                # 计算有效 token 数量 (排除 padding)
                # 假设 labels 中 padding token 的 id 为 -100 (HuggingFace 常用做法)
                # 或者使用 attention_mask
                # num_tokens = (batch['labels'] != -100).sum().item() # 如果有 labels
                num_tokens = batch['attention_mask'].sum().item() # 使用 attention mask 更通用
                if num_tokens > 0:
                     total_loss += loss.item() * num_tokens # 按 token 数加权损失
                     total_tokens += num_tokens
                else:
                    # 处理完全是 padding 的批次（虽然少见）
                    pass # 不累加损失或 token 数
            elif task_type == "classification":
                total_loss += loss.item() * batch_size # 按样本数加权损失
                total_samples += batch_size
                predictions = outputs.logits.argmax(dim=-1)
                correct_predictions += (predictions == batch["label"]).sum().item()

            batch_count += 1
            pbar.update(1) # 更新进度条

            # 可以在这里添加定期更新 pbar 后缀信息，但可能影响性能
            # if batch_count % 10 == 0:
            #    pbar.set_postfix(...)

    pbar.close() # 关闭进度条
    eval_duration = time.time() - eval_start_time

    # 计算最终指标
    results = {}
    log_message = f"评估完成 ({batch_count}/{num_batches} 批次) 用时: {eval_duration:.2f}s - "
    if task_type == "language_modeling":
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float('inf')
            results = {"loss": avg_loss, "perplexity": perplexity}
            log_message += f"Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Tokens: {total_tokens}"
        else:
            results = {"loss": float('nan'), "perplexity": float('nan')}
            log_message += "无有效 Tokens 处理。"
    elif task_type == "classification":
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            accuracy = correct_predictions / total_samples
            results = {"loss": avg_loss, "accuracy": accuracy}
            log_message += f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Samples: {total_samples}"
        else:
            results = {"loss": float('nan'), "accuracy": float('nan')}
            log_message += "无有效样本处理。"

    print(log_message) # 打印最终评估结果
    torch.cuda.empty_cache()
    model.train() # 恢复模型到训练模式
    return results
