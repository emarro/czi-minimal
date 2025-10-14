"""Unified CLI for Caduceus minimal training example."""

import logging
import os
import random
from pathlib import Path
from typing import cast

import fire
import torch
from caduceus import CaduceusConfig, CaduceusForMaskedLM, CaduceusTokenizer
from composer import Trainer
from composer.callbacks import (
    LRMonitor,
    SpeedMonitor,
    CheckpointSaver,
    RuntimeEstimator,
    MemoryMonitor,
)
from composer.loggers import WandBLogger
from composer.models import HuggingFaceModel
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer.utils import dist, reproducibility
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling
from utils import load_config, setup_logging


logger = logging.getLogger(__name__)


def generate_dna_sequence(max_length: int, fixed_length: bool = True) -> str:
    """Generate a random DNA sequence of specified or random length.

    Args:
        max_length: Maximum length of DNA sequence to generate
        fixed_length: If True, generate sequence of exactly max_length.
                     If False, generate sequence of random length up to max_length.

    Returns:
        A string of random DNA bases (A, C, G, T)
    """
    if fixed_length:
        length = max_length
    else:
        length = random.randint(1, max_length)
    return "".join(random.choice("ACGT") for _ in range(length))


def create_training_dataset(config_path: str = "config.yaml") -> None:
    """Create a dataset of random DNA sequences using parameters from config.yaml."""

    # Load config
    config = load_config(config_path)
    data_gen_config = config.data_generation

    # Extract parameters from config
    output_dir = data_gen_config.output_dir
    num_sequences = data_gen_config.num_sequences
    max_seq_length = data_gen_config.max_seq_length
    fixed_length = data_gen_config.fixed_length
    train_split = data_gen_config.train_split

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate split sizes
    num_train = int(num_sequences * train_split)
    num_val = num_sequences - num_train

    # Generate and save training data
    logger.info(f"Generating {num_train} training sequences...")
    with open(output_path / "train.txt", "w") as f:
        for _ in range(num_train):
            sequence = generate_dna_sequence(max_seq_length, fixed_length)
            f.write(sequence + "\n")

    # Generate and save validation data
    logger.info(f"Generating {num_val} validation sequences...")
    with open(output_path / "val.txt", "w") as f:
        for _ in range(num_val):
            sequence = generate_dna_sequence(max_seq_length, fixed_length)
            f.write(sequence + "\n")

    logger.info(f"Dataset created in {output_path}")
    logger.info(f"Training sequences: {num_train}")
    logger.info(f"Validation sequences: {num_val}")
    logger.info(
        f"{'Fixed' if fixed_length else 'Variable'} sequence length{'s' if not fixed_length else ''} up to {max_seq_length}"
    )


def build_model(cfg: DictConfig):
    """Build Caduceus model from config."""
    model_config = cfg.model.get("model_config", {})
    model_config = CaduceusConfig(**model_config)
    model = CaduceusForMaskedLM(model_config)
    tokenizer = CaduceusTokenizer(model_max_length=cfg.max_seq_len)

    # Debug info
    logger.info("\n=== Model Configuration ===")
    logger.info(
        f"Total number of parameters: {sum([x.numel() for x in model.parameters()])}"
    )
    logger.info(
        f"Total number of trainable parameters: {sum([x.numel() for x in model.parameters() if x.requires_grad])}"
    )
    logger.info(f"Model config: {model_config}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer.get_vocab())}")
    logger.info(f"Model vocab size: {model.config.vocab_size}")
    logger.info(f"Tokenizer vocab: {tokenizer.get_vocab()}")
    logger.info("=========================\n")

    return HuggingFaceModel(model, tokenizer)


def build_dataloader(cfg: DictConfig, tokenizer, batch_size: int, split: str = "train"):
    """Build data loader for masked language modeling."""

    # Load dataset
    dataset_name_or_path = (
        cfg.data_local if cfg.data_local is not None else cfg.data_remote
    )
    dataset = load_dataset(
        dataset_name_or_path,
        # data_files={split: os.path.join(cfg.data_local, f"{split}.txt")},
        split=split,
    )

    class TokenizedDataset(Dataset):
        def __init__(self, dataset, tokenizer, max_length, repeat_weight=0.1):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.seq_idx = "seq"
            self.repeat_weight = repeat_weight

            logger.info("\n=== Dataset Information ===")
            logger.info(f"Dataset size: {len(dataset)}")
            logger.info(f"Tokenizer vocabulary: {tokenizer.get_vocab()}")
            logger.info(f"Max sequence length: {max_length}")

            # Print first few sequences
            logger.info("\n=== Example Sequences ===")
            for i in range(min(3, len(dataset))):
                logger.info(f"Example {i} text: {dataset[i][self.seq_idx][:50]}...")
            logger.info("========================\n")

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            encoding = self.tokenizer(
                item[self.seq_idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=True,
            )
            is_lowercase = torch.tensor(
                [x.islower() for x in item[self.seq_idx]],
                device=encoding["input_ids"].device,
            )
            # print(item[self.seq_idx])
            # print(is_lowercase)
            # print(is_lowercase.shape)
            # raise Exception("Debug")
            # Remove the batch dimension since DataLoader will add it
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}

            # Create labels for masked language modeling
            labels = encoding["input_ids"].clone()

            # Get DNA token IDs directly from tokenizer's character set;
            # see https://github.com/kuleshov-group/llmlib/issues/8 for more on
            # why this is necessary and how it might be improved.
            dna_token_ids = {
                int(self.tokenizer.get_vocab()[c]) for c in self.tokenizer.characters
            }
            valid_dna_tokens = torch.tensor(
                [int(token_id) in dna_token_ids for token_id in labels]
            )
            labels[
                ~valid_dna_tokens
            ] = -100  # Mask out everything that's not a DNA token

            # Print detailed info for first few batches
            if idx < 3:
                logger.info(f"\n=== Example {idx} Details ===")
                logger.info(f"Raw text length: {len(item[self.seq_idx])}")
                logger.info(f"Raw text: {item[self.seq_idx][:50]}...")
                logger.info(f"Input IDs length: {len(encoding['input_ids'])}")
                logger.info(f"Input IDs: {encoding['input_ids'][:50]}...")
                logger.info(f"DNA token mask: {valid_dna_tokens[:50]}...")
                logger.info(f"Labels: {labels[:50]}")
                logger.info(
                    f"Number of DNA tokens to predict: {valid_dna_tokens.sum()}"
                )
                logger.info("========================\n")

            # Add labels to the encoding
            encoding["labels"] = labels
            repeat_loss = self.repeat_weight
            # Repeat regions are reweighted to repeat_loss. 1 otherwise.
            loss_weights = is_lowercase * (1 - repeat_loss) + 1
            encoding["loss_weights"] = loss_weights

            return encoding

    tokenized_dataset = TokenizedDataset(
        dataset, tokenizer, cfg.max_seq_len, cfg.repeat_weight
    )
    sampler = dist.get_sampler(tokenized_dataset, shuffle=(split == "train"))
    collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=cfg.mlm_probability
    )

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        num_workers=cfg.train_loader.num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
    )


def run_training(config_path: str = "config.yaml"):
    """Train the model using the specified config."""
    logger.info("Starting training...")

    # Load config
    cfg = OmegaConf.load(config_path)
    cfg = cast(DictConfig, cfg)

    # Set seed for reproducibility
    reproducibility.seed_all(cfg.seed)

    # Initialize distributed training
    if not dist.is_initialized():
        dist.initialize_dist()

    # Build model
    logger.info("Building model...")
    model = build_model(cfg)

    # Build optimizer
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # Build scheduler
    scheduler = CosineAnnealingWithWarmupScheduler(
        t_warmup=cfg.scheduler.t_warmup, alpha_f=cfg.scheduler.alpha_f
    )

    # Build callbacks
    callbacks = [
        LRMonitor(),
        SpeedMonitor(window_size=100),
        CheckpointSaver(weights_only=True),
        RuntimeEstimator(),
        MemoryMonitor(),
    ]

    # Build loggers
    loggers = []
    if "wandb" in cfg.get("loggers", {}):
        loggers.append(WandBLogger(**cfg.loggers.wandb))

    # Build data loaders
    logger.info("Building data loaders...")
    train_loader = build_dataloader(
        cfg,
        model.tokenizer,
        cfg.global_train_batch_size // dist.get_world_size(),
        split="train",
    )

    val_loader = None
    if cfg.data_local is not None:
        if os.path.exists(os.path.join(cfg.data_local, "val.txt")):
            val_loader = build_dataloader(
                cfg,
                model.tokenizer,
                cfg.global_train_batch_size // dist.get_world_size(),
                split="val",
            )
    else:
        val_loader = build_dataloader(
            cfg,
            model.tokenizer,
            cfg.global_train_batch_size // dist.get_world_size(),
            split="validation",
        )

    # Create trainer; see
    # https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.Trainer.html
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.get("eval_interval", "1000ba"),
        callbacks=callbacks,
        loggers=loggers,
        precision=cfg.precision,
        device_train_microbatch_size=cfg.device_train_microbatch_size,
        save_folder=cfg.get("save_folder"),
        save_interval=cfg.get("save_interval", "1000ba"),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep", -1),
        run_name=cfg.get("run_name", "caduceus-train"),
    )

    # Start training
    trainer.fit()


if __name__ == "__main__":
    logger = setup_logging()
    fire.Fire(
        {
            "create_training_dataset": create_training_dataset,
            "run_training": run_training,
        }
    )
