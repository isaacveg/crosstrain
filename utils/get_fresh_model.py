from transformers import AutoConfig, AutoModelForCausalLM

# 从预训练模型路径加载配置（不加载权重）
model_path = "ibm-granite/granite-3.0-1b-a400m-base"  # 或 instruct 版本
config = AutoConfig.from_pretrained(model_path)

# 从配置创建随机初始化的模型
model = AutoModelForCausalLM.from_config(config)

# 保存模型权重到本地目录，命名为granite-3.0-1b-a400m-fresh
model.save_pretrained("granite-3.0-1b-a400m-fresh")
