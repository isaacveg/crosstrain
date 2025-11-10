import time
import os
import argparse
from typing import Optional
import torch
import torch.distributed as dist
from torch.optim import AdamW
# import wandb  # Add wandb import
# from copy import deepcopy
from transformers.optimization import get_cosine_schedule_with_warmup
from utils.common import init_distributed, setup_logging, evaluate, calc_comm_delay
from utils.load_data_model import load_data_and_model
from utils.arg_utils import parse_args
from utils.checkpointing import save_checkpoint, load_checkpoint, get_latest_checkpoint, shutdown_async_checkpointing
from algorithms.diloco import sync_diloco, init_diloco_state
from algorithms.streaming import sync_streaming_diloco, init_streaming_state
from algorithms.dc_diloco import sync_dc_diloco, init_dc_state
from algorithms.shard_utils import select_next_shard


# Judge current device type
global USE_NPU
try:
    #适配昇腾NPU
    import torch_npu
    from torch_npu.npu.amp import autocast, GradScaler  #适配NPU混合精度训练
    from torch_npu.contrib import transfer_to_npu    #自动迁移
    USE_NPU = True
    print("使用昇腾NPU进行训练")
except ImportError:
    from torch import GradScaler
    USE_NPU = False
    print("使用GPU进行训练")


def train(model, dataloader, optimizer, scheduler,
          device, logger, world_size, rank, comm_delay=None,
          task_type="classification", args: Optional[argparse.Namespace] = None):
    """
    训练过程：
    - 模型以 train 模式运行
    - 实现传、算分离的分块同步策略
    - 模拟通信延迟，延迟应用更新
    - 分别统计计算和通信的时间
    """
    model.train()
    global_step = 0     # 实际步数（每accumulation_steps个小批量算一步）
    micro_step = 0      # 小批量计数
    comp_time_total = 0.0
    comm_time_total = 0.0
    comm_vol_total = 0.0
    log_comp_time = 0.0
    running_loss = 0.0      # 累积期间的总损失

    # 初始化用于单步时间统计的变量
    step_comp_time = 0.0
    step_comm_time = 0.0

    # 获取验证数据加载器
    train_dataloader, eval_dataloader = dataloader
    original_snapshot, outer_optimizer, shard_tracker = None, None, None
    # 计算模型参数字节数
    total_params_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    total_params_mb = total_params_bytes / (1024 * 1024)
    # Initialize algorithm-specific state using helpers
    num_shards = args.num_shards
    base_sync_points = None
    h = None
    if args.algorithm == 'diloco':
        state = init_diloco_state(model, args, logger)
        original_snapshot = state["original_snapshot"]
        outer_optimizer = state["outer_optimizer"]
    elif args.algorithm == 'streaming':
        state = init_streaming_state(model, args, logger)
        shard_tracker = state["shard_tracker"]
        base_sync_points = state["base_sync_points"]
        num_layers = state["num_layers"]
        if rank == 0:
            logger.info(f"模型共 {num_layers} 层，分为 {num_shards} 个分片")
    elif args.algorithm == 'dc':
        state = init_dc_state(model, args, logger)
        shard_tracker = state["shard_tracker"]
        num_layers = state["num_layers"]
        h = state["h"]
        if rank == 0:
            logger.info(f"模型共 {num_layers} 层，分为 {num_shards} 个分片")

    # 初始化 AMP GradScaler
    if args.use_amp and args.amp_type == torch.float16:
        scaler = GradScaler(enabled=True)
        logger.info("Initialized GradScaler for AMP.")
    else:
        scaler = GradScaler(enabled=False)

    # 从检查点恢复训练（如果指定）
    best_metric = float('inf')  # 用于跟踪最佳验证结果
    epoch = 0
    if args.resume and args.checkpoint_dir:
        # 查找当前rank的最新检查点
        resume_path = get_latest_checkpoint(args.checkpoint_dir, logger, rank)

        if resume_path:
            logger.info(f"Rank {rank}: 正在从检查点恢复训练: {resume_path}")
            # 加载检查点状态
            training_state = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler,
                original_snapshot, outer_optimizer, shard_tracker, rank, device, False, logger
            )
            # 检查加载的算法是否与当前配置匹配
            loaded_algorithm = training_state.get("algorithm")
            if loaded_algorithm and loaded_algorithm != args.algorithm:
                logger.warning(f"警告：尝试从一个 '{loaded_algorithm}' 算法的检查点恢复，但当前配置的算法是 '{args.algorithm}'。")
            # 更新训练状态
            if training_state:
                epoch = training_state.get("epoch", 0)
                global_step = training_state.get("global_step", 0)
                micro_step = training_state.get("micro_step", 0)
                comp_time_total = training_state.get("comp_time_total", 0.0)
                comm_time_total = training_state.get("comm_time_total", 0.0)
                comm_vol_total = training_state.get("comm_vol_total", 0.0)
                logger.info(f"Rank {rank}: 恢复训练成功，当前轮次: {epoch}, 当前步数: {global_step}")
                logger.info(f"当前训练时间： {comp_time_total}s, 当前通信总时间: {comm_time_total}s, 当前总通信量: {comm_vol_total} MB")
            else:
                logger.info(f"Rank {rank}: 检查点加载失败，将从头开始训练")
        else:
            logger.info(f"Rank {rank}: 未找到有效检查点，将从头开始训练")

    # 训练前清零梯度
    optimizer.zero_grad(set_to_none=True)

    for epoch_idx in range(epoch, args.epochs):
        epoch_step = 0  # 记录每个epoch内的step数
        logger.info(f"开始 Epoch {epoch_idx+1}/{args.epochs}")

        # 设置数据加载器的 epoch，确保恢复训练时的数据顺序正确
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch_idx)
            logger.info(f"设置 sampler epoch: {epoch_idx}")
        else:
            logger.info("当前 dataloader 没有可设置 epoch 的 sampler")

        for batch in train_dataloader:

            if global_step >= args.total_steps:
                logger.info(f"到达 ({args.total_steps}) 步数，停止训练。")
                break

            # 将 batch 数据移动到 GPU 上
            batch = {k: v.to(device) for k, v in batch.items()}
            start_comp = time.time()

            # 前向计算、损失计算、反向传播与优化
            # with autocast(device_type=device.type, enabled=args.use_amp, dtype=args.amp_type):
            # NPU 上必须用这个cuda写法不然会报错
            with torch.autocast(device_type="cuda", dtype=args.amp_type, enabled=args.use_amp):
                if task_type == "language_modeling":
                    outputs = model(**batch)
                elif task_type == "classification":
                    outputs = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["label"])
                loss = outputs.loss

            # 缩放损失以适应梯度累积
            scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            running_loss += loss.item()

            micro_step += 1
            comp_time = time.time() - start_comp
            if args.simulated_comp_time > 0:
                # 计算每个微步需要 sleep 的时间
                comp_time = args.simulated_comp_time
            comp_time_total += comp_time
            log_comp_time += comp_time
            step_comp_time += comp_time

            # log一下看看速度
            if rank == 0:
                logger.info(f"Micro step {micro_step}, Loss: {loss.item():.4f}, Time: {comp_time:.4f} seconds")

            # 只有在积累了足够的梯度后才更新参数
            if micro_step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # model.clip_grad_norm_(1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # 每隔 log_interval 步，输出一次日志
                if global_step % args.log_interval == 0:
                    # 使用累积的损失值进行记录
                    avg_loss = running_loss / args.gradient_accumulation_steps
                    # Get current learning rate and scale factor
                    current_lr = scheduler.get_last_lr()[0]
                    current_scale = scaler.get_scale()

                    logger.info(f"Epoch {epoch_idx+1}/{args.epochs}, Step {epoch_step} (Global: {global_step}) - "
                                f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, "
                                f"Scale: {current_scale:.1f}, CompTime: {log_comp_time:.4f}s")

                    # 记录到 wandb (只在主进程中)
                    if rank == 0 and args.use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/computation_time": log_comp_time,
                            "train/grad_scale": current_scale,
                            "step": global_step,
                        }, step=global_step)

                    log_comp_time = 0.0
                    running_loss = 0.0

                # 每隔 eval_interval 步，在验证集上评估
                current_metric = None
                is_best = False
                if rank==0 and global_step % args.eval_interval == 0 and eval_dataloader is not None:
                    eval_results = evaluate(model, eval_dataloader, device,
                                            task_type, args.use_amp, args.amp_type,
                                            args.eval_batch_size, args.max_eval_batches)

                    # 根据任务类型输出不同的评估指标
                    if task_type == "classification":
                        current_metric = eval_results['loss']  # 或使用 accuracy
                        logger.info(f"评估结果 [步数 {global_step}] - 验证损失: {eval_results['loss']:.4f}, 准确率: {eval_results['accuracy']:.4f}")
                    else:
                        current_metric = eval_results['loss']  # 或使用 perplexity
                        logger.info(f"评估结果 [步数 {global_step}] - 验证损失: {eval_results['loss']:.4f}, 困惑度: {eval_results['perplexity']:.4f}")

                    # 检查是否为最佳模型
                    if current_metric < best_metric:
                        best_metric = current_metric
                        is_best = True

                    if args.use_wandb:
                        wandb_log = {"eval/loss": eval_results['loss']}
                        if "accuracy" in eval_results:
                            wandb_log["eval/accuracy"] = eval_results['accuracy']
                        if "perplexity" in eval_results:
                            wandb_log["eval/perplexity"] = eval_results['perplexity']
                        wandb.log(wandb_log, step=global_step)

                # 每隔 checkpoint_interval 步，保存检查点
                if args.checkpoint_dir and global_step % args.checkpoint_interval == 0:
                    # 所有rank都保存检查点
                    logger.info(f"Rank {rank}: 保存检查点，当前步数: {global_step}")
                    save_checkpoint(
                        args.algorithm,
                        args.checkpoint_dir, model, optimizer, scheduler,
                        scaler, original_snapshot, outer_optimizer, shard_tracker,
                        epoch_idx, global_step, micro_step,
                        comp_time_total, comm_time_total,comm_vol_total,
                        current_metric, is_best, rank,
                        False, args.max_checkpoints, logger
                    )

                global_step += 1
                epoch_step += 1

                # 检查是否需要发送或接收模型分片
                if args.algorithm == 'diloco':
                    if global_step % args.sync_interval == 0:
                        comm_time = sync_diloco(
                            model, original_snapshot, outer_optimizer,
                            world_size, logger, comm_delay
                        )
                        comm_time_total += comm_time
                        step_comm_time += comm_time
                        comm_vol_total += total_params_mb

                elif args.algorithm == 'streaming':
                    for shard_idx in range(num_shards):
                        # Check if we need to receive this shard
                        if shard_tracker[shard_idx]["next_receive_step"] == global_step:
                            comm_time = sync_streaming_diloco(
                                model, shard_tracker, shard_idx, world_size, logger,
                                comm_delay, num_shards, args.alpha)

                            comm_time_total += comm_time
                            step_comm_time += comm_time
                            comm_vol_total += shard_tracker[shard_idx]["global_num_bytes"] / (1024 * 1024)  # MB
                            # 记录通信时间到 wandb (只在主进程中)
                            if rank == 0 and args.use_wandb:
                                wandb.log({
                                    "communication_time": comm_time,
                                    "sync_shard_idx": shard_idx,
                                }, step=global_step)
                        # Check if we need to record/send this shard
                        # Send current params in staged_params
                        relative_step = global_step % args.sync_interval
                        if relative_step == base_sync_points[shard_idx]:
                            shard_tracker[shard_idx]["sent_at_step"] = global_step
                            # Calculate next receive step based on current send
                            shard_tracker[shard_idx]["next_receive_step"] = global_step + args.delay_steps

                            # 使用clone()创建参数的深拷贝而不仅是引用
                            shard_tracker[shard_idx]["staged_params"] = [
                                    p.data.clone() for p in shard_tracker[shard_idx]["param_refs"]
                                ]
                            logger.info(f"分片 {shard_idx+1} 已记录并发送，当前步数: {global_step}, 将在步数 {shard_tracker[shard_idx]['next_receive_step']} 接收")

                elif args.algorithm == 'dc':
                    for shard_idx in range(num_shards):
                        # Check if we need to receive this shard
                        if shard_tracker[shard_idx]["next_receive_step"] == global_step:
                            comm_time = sync_dc_diloco(
                                model, shard_tracker, shard_idx, world_size, logger,
                                comm_delay, num_shards, args.alpha, args.dc_lambda)

                            comm_time_total += comm_time
                            step_comm_time += comm_time
                            comm_vol_total += shard_tracker[shard_idx]["global_num_bytes"] / (1024 * 1024)  # MB

                            # 记录通信时间到 wandb (只在主进程中)
                            if rank == 0 and args.use_wandb:
                                wandb.log({
                                    "communication_time": comm_time,
                                    "sync_shard_idx": shard_idx,
                                }, step=global_step)
                    if global_step % h == 0 and global_step > 0:
                        # 调用动态片段选择函数
                        selected_shard_idx = select_next_shard(shard_tracker, args.sync_interval, args.num_shards, global_step)
                        # 执行同步（复用 sync_model 函数）
                        if selected_shard_idx is not None:
                            shard_tracker[selected_shard_idx]["old_sent_at_step"] = shard_tracker[selected_shard_idx]["sent_at_step"]
                            shard_tracker[selected_shard_idx]["sent_at_step"] = global_step
                            shard_tracker[selected_shard_idx]["staged_params"] = [p.data.clone() for p in shard_tracker[selected_shard_idx]["param_refs"]]
                            shard_tracker[selected_shard_idx]["next_receive_step"] = global_step + args.delay_steps
                            logger.info(f"分片 {selected_shard_idx+1} 已记录并发送，当前步数: {global_step}, 将在步数 {shard_tracker[selected_shard_idx]['next_receive_step']} 接收")
                if rank == 0:
                    logger.info(f"当前总通信量： {comm_vol_total:.2f} MB")
                # 记录单步时间并重置计时器
                total_step_time = step_comp_time + step_comm_time
                if rank == 0:
                    logger.info(f"Step {global_step} Timing - Comp: {step_comp_time:.4f}s, Comm: {step_comm_time:.4f}s, Total: {total_step_time:.4f}s")
                    if args.use_wandb:
                        wandb.log({
                            "time/computation_step": step_comp_time,
                            "time/communication_step": step_comm_time,
                            "time/total_step": total_step_time,
                        }, step=global_step)

                # 重置单步计时器
                step_comp_time = 0.0
                step_comm_time = 0.0

        # 每个epoch结束时输出一次总结
        if rank == 0:
            logger.info(f"完成 Epoch {epoch_idx+1}/{args.epochs}, 当前步数: {global_step}")

        # 每个epoch结束保存检查点
        if args.checkpoint_dir:
            logger.info(f"Rank {rank}: 保存 Epoch {epoch_idx+1} 检查点")
            save_checkpoint(
                args.algorithm, args.checkpoint_dir, model, optimizer, scheduler,
                scaler, original_snapshot, outer_optimizer, shard_tracker,
                epoch_idx, global_step, micro_step,
                comp_time_total, comm_time_total,
                None, False, rank,
                False, args.max_checkpoints, logger
            )


        if global_step >= args.total_steps:
            if eval_dataloader is not None:
                # 进行最终评估
                final_metrics = {}
                final_eval_results = evaluate(model, eval_dataloader, device,
                                             task_type, args.use_amp, args.amp_type,
                                             args.eval_batch_size, args.max_eval_batches)
                for k, v in final_eval_results.items():
                    final_metrics[f"final/eval_{k}"] = v

                logger.info(f"最终评估结果: {final_metrics}")
                if args.use_wandb and rank == 0:
                    wandb.log(final_metrics, step=global_step)
            break

    # 汇总所有GPU的时间统计数据
    metrics = torch.tensor([comp_time_total, comm_time_total, global_step], dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    # 除以进程数得到平均值
    avg_comp_time = metrics[0].item() / world_size
    avg_comm_time = metrics[1].item() / world_size

    # 同步 global_step (应该都一样，但为安全起见)
    global_step = int(metrics[2].item() / world_size)

    # 只在主进程(rank 0)输出平均统计信息
    if rank == 0:
        logger.info(f"===== 平均统计信息（所有 {world_size} 个GPU） =====")
        logger.info(f"总计算时间(平均): {avg_comp_time:.4f} 秒")
        logger.info(f"总通信时间(平均): {avg_comm_time:.4f} 秒")
        logger.info(f"总通信量(平均): {comm_vol_total / world_size:.2f} MB")
        logger.info(f"平均计算时间: {avg_comp_time / global_step:.4f} 秒")
        logger.info(f"平均通信时间: {avg_comm_time / (global_step // args.sync_interval):.4f} 秒")
        logger.info(f"平均步长时间: {(avg_comp_time + avg_comm_time) / global_step:.4f} 秒")
        logger.info(f"计算时间占比: {avg_comp_time / (avg_comp_time + avg_comm_time):.2%}")
        logger.info(f"实际批量大小: {args.batch_size * args.gradient_accumulation_steps} (批量大小 {args.batch_size} * 梯度累积步数 {args.gradient_accumulation_steps})")


def main():
    # 解析命令行参数
    args = parse_args()

    # 初始化分布式训练相关环境
    local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("high")
    args.amp_type = torch.bfloat16 if args.amp_type == 'bf16' else torch.float16

    # 设置梯度累积步数
    if args.effective_batch_size is not None:
        per_device_batch_size = args.batch_size
        total_batch_size = args.effective_batch_size * world_size
        args.gradient_accumulation_steps = args.effective_batch_size // per_device_batch_size
        if rank == 0:
            print(f"Effective batch size: {args.effective_batch_size}")
            print(f"Per-device batch size: {per_device_batch_size}")
            print(f"Total batch size (across all devices): {total_batch_size}")
            print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")


    # 设置 logger
    logger = setup_logging(rank, args.log_dir)
    logger.info(f"初始化完成。当前 Rank: {rank}, 总进程数: {world_size}")
    logger.info(f"梯度累积步数: {args.gradient_accumulation_steps}, 实际批量大小: {args.batch_size * args.gradient_accumulation_steps * world_size} (全局)")

    # 检查是否需要恢复训练，获取WandB run ID
    loaded_wandb_run_id = None
    global_step_to_resume = None
    if args.resume and args.checkpoint_dir and rank == 0:
        latest_checkpoint_path = get_latest_checkpoint(args.checkpoint_dir, logger, rank)
        # 只有rank 0需要检查wandb ID
        if latest_checkpoint_path:
            logger.info(f"Rank 0: Found potential checkpoint for resume: {latest_checkpoint_path}")
            try:
                # 只加载检查点元数据以获取 ID，避免加载整个模型到 CPU
                ckpt_data = torch.load(latest_checkpoint_path, map_location='cpu')
                loaded_wandb_run_id = ckpt_data.get("wandb_run_id")
                global_step_to_resume = ckpt_data.get("global_step", None)
                if loaded_wandb_run_id:
                    logger.info(f"Rank 0: Found WandB run ID in checkpoint: {loaded_wandb_run_id}, global step: {global_step_to_resume}")
                else:
                    logger.warning("Rank 0: Checkpoint found, but no WandB run ID saved within it. Starting a new WandB run.")
                del ckpt_data # 释放内存
            except Exception as e:
                logger.error(f"Rank 0: Failed to load checkpoint metadata for WandB ID: {e}", exc_info=True)

    # 初始化 wandb (只在主进程中)
    if rank == 0 and args.use_wandb:
        wandb_config = vars(args)
        wandb_id_to_use = loaded_wandb_run_id if args.resume else None # 仅在 resume 时尝试使用旧 ID
        resume_status = "allow" if wandb_id_to_use else None

        # 如果没有指定实验名称，自动生成一个
        if args.wandb_name is None:
            args.wandb_name = f"streaming_{args.model_name}_{args.dataset_name}_sync{args.sync_interval}_ns{args.num_shards}_ds{args.delay_steps}"
            if args.outer_lr != 1.0:
                args.wandb_name += f"_olr{args.outer_lr}"
                if args.use_nesterov:
                    args.wandb_name += "_nesterov"
            if args.alpha != 0.5:
                args.wandb_name += f"_a{args.alpha}"
            if args.gradient_accumulation_steps > 1:
                args.wandb_name += f"_acc{args.gradient_accumulation_steps}"

        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                # id=wandb_id_to_use if wandb_id_to_use else None,
                dir='/data/yzhu/distrain/runs/streaming_diloco',
                config=wandb_config,
                resume=resume_status,
                # resume_from=f"{wandb_id_to_use}?_step={global_step_to_resume}" if (wandb_id_to_use and global_step_to_resume) else None
            )
            logger.info(f"Weights & Biases 初始化完成。项目: {args.wandb_project}, 实验: {args.wandb_name}")
            # 如果是恢复运行，且实际恢复的ID与加载的ID不同，记录一下
            if resume_status == "must" and wandb.run and wandb.run.id != loaded_wandb_run_id:
                 logger.warning(f"WandB 恢复时使用了新的 ID: {wandb.run.id} (预期 ID: {loaded_wandb_run_id})")
            # 将最终的 run id 保存回 args，以便保存到检查点
            if wandb.run: args.wandb_run_id = wandb.run.id

        except Exception as e:
            logger.error(f"WandB 初始化失败: {e}", exc_info=True)
            args.use_wandb = False # 初始化失败则禁用

    # 记录训练参数
    if rank == 0:
        logger.info("--- 最终训练参数 ---")
        for k, v in vars(args).items():
             logger.info(f"{k}: {v}")
        logger.info("--------------------")
        # 保存配置到文件
        config_path = os.path.join(args.log_dir, "config.txt")
        try:
            with open(config_path, "w") as f:
                 for k, v in vars(args).items():
                     f.write(f"{k}: {v}\n")
                 f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            logger.info(f"训练参数已保存到: {config_path}")
        except IOError as e:
             logger.error(f"无法保存参数到文件: {e}")

    # 加载数据集和对应的模型与分词器（每个节点各自加载独立副本）
    train_dataloader, eval_dataloader, tokenizer, model, task_type = load_data_and_model(
        args.dataset_name, args.model_name,
        args.batch_size, args.eval_batch_size,
        rank, world_size
    )
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型和数据加载完成。模型参数量: {num_params:,}")
    logger.info(f"预计模型大小：{num_params * (2 if args.use_amp else 4) / 1e6:.2f} MB")
    logger.info(f"任务类型: {task_type}")
    args.task_type = task_type # 将检测到的任务类型存入args

    # 配置优化器和学习率调度器
    if args.weight_decay > 0:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    # 计算通信延迟
    comm_delay = calc_comm_delay(model, world_size, logger, simulated_bandwidth_mbps=args.bandwidth)

    # 使用传入的参数进行训练，传入训练和验证数据加载器
    train_start_time = time.time()
    try:
        train(model, (train_dataloader, eval_dataloader), optimizer,
              scheduler, device, logger, world_size, rank, comm_delay,
              task_type, args)
    except Exception as e:
         logger.error(f"训练主函数捕获到错误: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        train_duration = time.time() - train_start_time
        logger.info(f"总训练时长: {train_duration:.2f} 秒 ({train_duration/3600:.2f} 小时)")
        if rank == 0:
            # Append end time to config file
            try:
                with open(os.path.join(args.log_dir, "config.txt"), "a") as f:
                    f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total duration: {train_duration:.2f} seconds\n")
            except IOError as e:
                logger.error(f"Could not write end time to config file: {e}")

            if args.use_wandb and wandb.run is not None:
                logger.info("Finishing WandB run...")
                wandb.finish()

        logger.info("Destroying process group...")
        shutdown_async_checkpointing(logger)
        dist.destroy_process_group()
        logger.info("Process group destroyed.")

if __name__ == "__main__":
    main()
