# init a fresh model from ibm-granite/granite-3.0-1b-a400m-base
import os
os.environ['HF_HOME'] = '/data/hfhub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def init_granite_weights(model, init_std=0.1):
    """
    Initialize model parameters to match Granite 3.0 initialization.

    Args:
        model: The model to initialize.
        init_std: Standard deviation for normal distribution initialization.
    """
    for name, param in model.named_parameters():
        if 'bias' in name:
            # Initialize biases to 0
            # Note: there's no bias in the Granite model architecture
            nn.init.zeros_(param)
        elif 'weight' in name:
            # Initialize weights from a normal distribution with mean=0, std=init_std
            nn.init.normal_(param, mean=0.0, std=init_std)
        elif isinstance(param, nn.LayerNorm):
            # Initialize LayerNorm weights (Gamma) to 1, and Bias (Beta) to 0
            if 'weight' in name:
                nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

model_save_path = '/data/hfhub/granite-3.0-1b-a400m-fresh'
model_name = "ibm-granite/granite-3.0-1b-a400m-base"
# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_save_path)
# Save model
model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name))
init_granite_weights(model)
model.config.save_pretrained(model_save_path)
model.save_pretrained(model_save_path, safe_serialization=True)

# list files in the model save path
print("Model and tokenizer saved to:", model_save_path)
print("Files in the model save path:")
for root, dirs, files in os.walk(model_save_path):
    for file in files:
        print(os.path.join(root, file))
print("Done.")
