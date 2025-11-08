import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, DataCollatorForLanguageModeling


def load_sst2_data(tokenizer, batch_size, eval_batch_size, rank, world_size):
    """Load SST-2 dataset for sentiment classification"""
    train_dataset = load_dataset("glue", "sst2", split="train")
    eval_dataset = load_dataset("glue", "sst2", split="validation")

    block_size = 128

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=block_size)

    # Preprocess train and validation sets
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Create distributed samplers and dataloaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    return train_dataloader, eval_dataloader

def load_c4_data(dataset_name, tokenizer, batch_size, eval_batch_size, rank, world_size):
    """Load C4-en dataset for language modeling"""
    # Load only training data, no validation set
    ds = load_dataset(f"{dataset_name}", streaming=True,
                      data_files={
                "train": "en/c4-train.*.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            })
    ds = ds.shuffle(seed=2025)

    block_size = 1024   # DiLoCo 论文
    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=block_size)
        return outputs

    tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"]).with_format("torch")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = split_dataset_by_node(tokenized_datasets["train"], world_size=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=batch_size, pin_memory = True, num_workers = 4)

    eval_dataloader = DataLoader(
        dataset=tokenized_datasets["validation"],
        collate_fn=data_collator,
        batch_size=eval_batch_size,
        pin_memory = True,
        num_workers = 4
    )

    # Return train dataloader and None for eval_dataloader, as we'll compute ppl differently
    return train_dataloader, eval_dataloader

def load_data_and_model(dataset_name, model_name, batch_size, eval_batch_size, rank, world_size):
    """
    Load dataset and model.
    Return Trainloader, Evalloader, tokenizer, model objects.
    """
    if dataset_name == "sst2" and model_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        train_dataloader, eval_dataloader = load_sst2_data(tokenizer, batch_size, eval_batch_size, rank, world_size)
        return train_dataloader, eval_dataloader, tokenizer, model, 'classification'

    elif 'c4' in dataset_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it
        model = AutoModelForCausalLM.from_pretrained(model_name)
        train_dataloader, eval_dataloader = load_c4_data(dataset_name, tokenizer, batch_size, eval_batch_size, rank, world_size)
        return train_dataloader, eval_dataloader, tokenizer, model, 'language_modeling'

    else:
        raise ValueError(f"Unsupported combination of dataset and model: {dataset_name}, {model_name}")
