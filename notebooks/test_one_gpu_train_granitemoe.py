import os
import sys
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

# Make project root importable (so we can import from utils)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.common import evaluate  # noqa: E402


def main():
    # Environment variables (same as in the notebook)
    os.environ["HF_HOME"] = "/data/hfhub"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # Basic config (mirroring the notebook)
    total_batch_size = 256
    per_device_batch_size = 8
    accumulation_steps = total_batch_size // per_device_batch_size

    device = torch.device("cuda:3")

    # Model & tokenizer
    model_path = "/data/hfhub/granite-3.0-1b-a400m-fresh/"
    print("Loading Granite MoE model from", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Dataset (C4 streaming, same setup as notebook)
    root = "/data/hfhub/datasets/c4"
    train_glob = os.path.join(root, "en", "c4-train.*.json.gz")
    val_glob = os.path.join(root, "en", "c4-validation.00000-of-00008.json.gz")

    ds = load_dataset(
        "json",
        data_files={"train": train_glob, "validation": val_glob},
        streaming=True,
    )

    # Shuffle stream (same as notebook)
    ds = ds.shuffle(seed=2025)

    block_size = 1024

    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=block_size)
        return outputs

    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=per_device_batch_size,
        pin_memory=True,
        num_workers=4,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=32,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=4000,
    )

    model.train()
    cnt = 0
    step = 0
    running_loss = 0.0
    step_time = 0.0

    print("Starting training loop...")

    for batch in train_dataloader:
        t0 = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)

        loss = outputs.loss
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        running_loss += loss.item()
        cnt += 1

        comp_time = time.time() - t0
        step_time += comp_time

        if cnt % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            print(
                f"Completed step {step}, "
                f"Avg Loss: {running_loss / accumulation_steps:.4f}, "
                f"Step time: {step_time:.4f} , "
                f"Avg step time: {step_time / accumulation_steps:.4f} sec",
                flush=True,
            )

            running_loss = 0.0
            step_time = 0.0

            scheduler.step()

            if step % 100 == 0:
                evaluate(
                    model,
                    eval_dataloader,
                    device,
                    "language_modeling",
                    use_amp=True,
                    amp_type=torch.bfloat16,
                    eval_batch_size=32,
                    max_eval_batches=50,
                )

            if step >= 4000:
                print("Reached max training steps (4000). Stopping training.")
                break

    print("Training finished.")


if __name__ == "__main__":
    main()
