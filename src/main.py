"""Unified CLI for Caduceus minimal training example."""

import logging
import os
from typing import cast, Optional, Any

import hydra_setup  # register resolvers for hydra

import hydra
import torch
from composer import Trainer
from composer.callbacks import (
    LRMonitor,
    SpeedMonitor,
    CheckpointSaver,
    RuntimeEstimator,
    MemoryMonitor,
)
from composer.core import Evaluator
from composer.loggers import WandBLogger
from composer.utils import dist, reproducibility
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DataCollatorForLanguageModeling,
)


from composer.models.tasks import ComposerClassifier
from composer.profiler import JSONTraceHandler, cyclic_schedule
from composer.profiler.profiler import Profiler

from callbacks.flop_counter import FlopMonitor, BPredMonitor
from callbacks.visualizer import IGVCallBack
from callbacks.logger import ChrChunker
from callbacks.hf_saver import HuggingFaceCompatibleCheckpointing


logger = logging.getLogger(__name__)


def build_dataloader(
    cfg: DictConfig,
    tokenizer,
    batch_size: int,
    max_seq_len: int,
    mlm: bool,
    default_target_ratio: Optional[int] = None,
    split: str = "train",
    eval_only: bool = False,
    mask_seq: bool = False,
):
    """Build data loader for masked language modeling."""

    # Load dataset
    dataset_name_or_path = (
        cfg.data_local if cfg.data_local is not None else cfg.data_remote
    )
    if eval_only:
        dataset_name_or_path = cfg.data_remote
        # split naming error
        split = "train"
        mask_seq = mask_seq
    dataset = load_dataset(
        dataset_name_or_path,
        # data_files={split: os.path.join(cfg.data_local, f"{split}.txt")},
        split=split,
    )

    cutoff = cfg.get("cutoff", None)
    # Filter sequences that remain with lots of "N"s
    if cutoff is not None:
        logger.info(f"Dataset length: {len(dataset)}")
        dataset = dataset.filter(
            lambda batch: batch["seq"].count("N") < len(batch["seq"]) * cutoff
        )  # keep if percentage of N is less than cutoff
        logger.info(
            f"Dataset length after filter with cutoff {cutoff * 100}%: {len(dataset)}"
        )

    class TokenizedDataset(Dataset):
        def __init__(
            self,
            dataset,
            tokenizer,
            max_length,
            repeat_weight=0.1,
            mask_seq=False,
            default_target_ratio=None,
            mlm=True,
        ):
            """
            Datasets that wraps tokenization
            dataset: HF dataset
            tokenizer: HF tokenizer
            max_length: int max length of seq (will truncate)
            repeat_weight: float the value for which to downweight repetitive (soft-masked) portions of a seq
            mask_seq: bool Whether to mask sequences at middle token (if ref and alt are in dataset)
            default_target_ratio: The default target ratio used by HNet (N in their paper) (if any)
            """
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.seq_idx = "seq"
            self.repeat_weight = repeat_weight
            self.mask_seq = mask_seq
            self.default_target_ratio = default_target_ratio
            self.mlm = mlm

            logger.info("\n=== Dataset Information ===")
            logger.info(f"Dataset size: {len(dataset)}")
            logger.info(f"Tokenizer vocabulary: {tokenizer.get_vocab()}")
            logger.info(f"Max sequence length: {max_length}")

            # Print first few sequences
            # logger.info("\n=== Example Sequences ===")
            # for i in range(min(3, len(dataset))):
            #    logger.info(f"Example {i} text: {dataset[i][self.seq_idx][:50]}...")
            #    logger.info(f"Example {i} ds: {self[i]}")
            # logger.info("========================\n")

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            sequence = item[self.seq_idx]
            ref_id, alt_id = None, None
            var_idx = None
            if self.mask_seq:
                var_idx = len(sequence) // 2
                seq_bp = sequence[var_idx]
                if item["ref"] != seq_bp:
                    # var_idx -= 1
                    sequence = sequence[::-1]
                    seq_bp = sequence[var_idx]
                assert item["ref"] == seq_bp, (
                    f"Masking in eval dataloader failed, found {seq_bp} when we expected {item['ref']} around {sequence[var_idx - 5 : var_idx + 5]}"
                )
                assert item["alt"] != item["ref"], (
                    f"Error, found REF bp {item['ref']} and ALT bp {item['alt']} to be the same"
                )
                # sequence[var_idx] = int(
                #    self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                # )
                if self.mlm:
                    sequence = (
                        sequence[:var_idx]
                        + self.tokenizer.mask_token
                        + sequence[var_idx + 1 :]
                    )
                ref_id = tokenizer(item["ref"], return_tensors="pt")["input_ids"][:, 0]
                alt_id = tokenizer(item["alt"], return_tensors="pt")["input_ids"][:, 0]

            encoding = self.tokenizer(
                sequence,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
            if ref_id is not None:
                encoding["ref_id"] = ref_id
                encoding["alt_id"] = alt_id
            for k, v in item.items():
                if type(v) is not str:
                    encoding[k] = torch.tensor(v)
                elif k == self.seq_idx:  # ignore seq
                    continue
                elif k == "chrom":
                    if v in [str(x) for x in range(1, 30)]:
                        encoding[k] = torch.tensor(int(v))
                    else:
                        encoding[k] = torch.tensor(-1)
                elif k == "annotation_mask":
                    encoding[k] = torch.tensor(v)  # [L] boolean mask
                    # if k not in tokenizer.aux_mappings:  # cast assembly/chr to int
                    # tokenizer.aux_mappings[k] = Vocab()
                    # print(tokenizer.aux_mappings)
                # encoding[k] = torch.tensor(tokenizer.aux_mappings[k][v])

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
            labels = encoding["input_ids"].detach().clone()

            if self.mlm:
                # Get DNA token IDs directly from tokenizer's character set;
                # see https://github.com/kuleshov-group/llmlib/issues/8 for more on
                # why this is necessary and how it might be improved.
                dna_token_ids = {
                    int(self.tokenizer.get_vocab()[c])
                    for c in self.tokenizer.characters
                }
                valid_dna_tokens = torch.tensor(
                    [int(token_id) in dna_token_ids for token_id in labels]
                )
                labels[
                    ~valid_dna_tokens
                ] = -100  # Mask out everything that's not a DNA token
            else:
                # We're in AR and need to shift inputs and labels ourselves
                encoding["input_ids"] = encoding["input_ids"][:-1]

            # Print detailed info for first few batches
            # if idx < 0:
            # logger.info(f"\n=== Example {idx} Details ===")
            # logger.info(f"Raw text length: {len(item[self.seq_idx])}")
            # logger.info(f"Raw text: {item[self.seq_idx][:50]}...")
            # logger.info(f"Input IDs length: {len(encoding['input_ids'])}")
            # logger.info(f"Input IDs: {encoding['input_ids'][:50]}...")
            # logger.info(f"DNA token mask: {valid_dna_tokens[:50]}...")
            # logger.info(f"Labels: {labels[:50]}")
            # logger.info(
            #    f"Number of DNA tokens to predict: {valid_dna_tokens.sum()}"
            # )
            # logger.info("========================\n")

            # Add labels to the encoding
            encoding["labels"] = labels if self.mlm else labels[1:]
            repeat_loss = self.repeat_weight
            # Repeat regions are reweighted to repeat_loss. 1 otherwise.
            loss_weights = (is_lowercase * (repeat_loss - 1)) + 1
            encoding["loss_weights"] = loss_weights if self.mlm else loss_weights[1:]
            if self.default_target_ratio is not None:
                encoding["target_ratio"] = torch.tensor(self.default_target_ratio)

            return encoding

    tokenized_dataset = TokenizedDataset(
        dataset,
        tokenizer,
        max_seq_len if cfg.seq_len is None else cfg.seq_len,
        cfg.repeat_weight,
        mask_seq=mask_seq,
        default_target_ratio=default_target_ratio,
        mlm=mlm,
    )
    sampler = dist.get_sampler(tokenized_dataset, shuffle=(split == "train"))
    collate_fn = (
        DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=cfg.mlm_probability if not mask_seq else 0.0,
            mask_replace_prob=cfg.mask_replace_prob,
            random_replace_prob=cfg.random_replace_prob,
        )
        if (mlm and not eval_only)
        else None
    )  # Collator overwrites the labels from __getitem__, disable if not mlm!

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
        pin_memory=True,
    )


@hydra.main(version_base=None, config_path="config", config_name="train")
def run_training(cfg: DictConfig) -> None:
    """Train the model using the specified config."""
    logger.info("Starting training...")
    print(cfg)

    # Load config
    # cfg = OmegaConf.load(config_path)
    cfg = cast(DictConfig, cfg)

    # Set seed for reproducibility
    reproducibility.seed_all(cfg.trainer.seed)

    # Initialize distributed training
    if not dist.is_initialized():
        dist.initialize_dist()

    # Build model
    logger.info("Building model...")
    model = hydra.utils.instantiate(cfg.model)
    num_params = sum([x.numel() for x in model.parameters()])
    num_trainable_params = sum(
        [x.numel() for x in model.parameters() if x.requires_grad]
    )
    logger.info(f"Num params: {num_params:,}")
    logger.info(f"Num trainable params: {num_trainable_params:,}")

    # Build optimizer
    optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        model.parameters(),
        # lr=cfg.optimizer.lr,
        # betas=cfg.optimizer.betas,
        # eps=cfg.optimizer.eps,
        # weight_decay=cfg.optimizer.weight_decay,
    )

    # Build scheduler
    scheduler = hydra.utils.instantiate(cfg.scheduler)

    # Build callbacks
    callbacks = [
        LRMonitor(),
        SpeedMonitor(window_size=100),
        # CheckpointSaver(
        #    weights_only=False,
        #    folder=cfg.trainer.get("save_folder"),
        #    save_interval=cfg.trainer.get("save_interval", "1000ba"),
        #    num_checkpoints_to_keep=cfg.trainer.get("save_num_checkpoints_to_keep", -1),
        #    overwrite=cfg.trainer.get("save_overwrite", False),
        # ),
        HuggingFaceCompatibleCheckpointing(
            disable_hf=cfg.callbacks.get("disable_hf"),
            save_local=cfg.callbacks.get("save_local"),
            save_to_hub=cfg.callbacks.get("save_to_hub"),
            hub_repo_id=cfg.callbacks.get("hub_repo_id"),
            private=cfg.callbacks.get("private", True),
            weights_only=True,
            folder=cfg.callbacks.get("save_folder"),
            save_interval=cfg.callbacks.get("save_interval", "1000ba"),
            num_checkpoints_to_keep=cfg.trainer.get("save_num_checkpoints_to_keep", -1),
            overwrite=cfg.trainer.get("save_overwrite", False),
        ),
        RuntimeEstimator(),
        MemoryMonitor(),
        FlopMonitor(),
    ]
    if cfg.model.get("log_bpreds", False):
        callbacks.append(BPredMonitor())

    # Build loggers
    loggers = []
    if "wandb" in cfg.get("loggers", {}):
        api_key = cfg.loggers.wandb.api_key
        if api_key is None and "WANDB_API_KEY" not in os.environ:
            raise Exception(
                "WANDB logger instantiated by not API key was provided, make sure .env is set up properly"
            )
        if "WANDB_API_KEY" not in os.environ:
            os.environ["WANDB_API_KEY"] = api_key
        import wandb

        try:
            wandb.login()
        except Exception as e:
            print(
                f"Logging in with key {os.environ['WANDB_API_KEY']} failed, error {e}"
            )
            raise Exception(e)
        dict_cfg: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
        dict_cfg["num_params"] = num_params
        dict_cfg["num_trainable_params"] = num_trainable_params
        loggers.append(
            WandBLogger(
                project=cfg.loggers.wandb.project,
                entity=cfg.loggers.wandb.entity,
                tags=cfg.loggers.wandb.tags
                if cfg.loggers.wandb.tags is not None
                else None,
                init_kwargs={
                    "config": dict_cfg,
                    "config_exclude_keys": [
                        "loggers"
                    ],  # dont include loggers in config, might leak api keys
                },
            )
        )
        # loggers[-1].log_hyperparameters(cfg)

    # Build data loaders
    logger.info("Building data loaders...")
    train_loader = build_dataloader(
        cfg.dataset,
        model.tokenizer,
        cfg.trainer.global_train_batch_size // dist.get_world_size(),
        split="train",
        max_seq_len=cfg.model.max_seq_len,
        mlm=cfg.model.mlm,
        default_target_ratio=cfg.model.get("default_target_ratio", None),
    )
    val_loader = None
    val_loader = build_dataloader(
        cfg.dataset,
        model.tokenizer,
        cfg.trainer.global_train_batch_size // dist.get_world_size(),
        split="validation",
        max_seq_len=cfg.model.max_seq_len,
        mlm=cfg.model.mlm,
        default_target_ratio=cfg.model.get("default_target_ratio", None),
    )

    val_loader = Evaluator(
        label="eval_split",
        dataloader=val_loader,
        metric_names=["EvalLoss", "ARLoss", "RatioLoss", "Accuracy"],
        eval_interval=cfg.dataset.eval_interval,
        device_eval_microbatch_size=cfg.trainer.device_train_microbatch_size,
    )
    eval_dataloaders = [val_loader]
    if cfg.eval_dataset is not None:
        zeroshot_val_loader = build_dataloader(
            cfg.eval_dataset,
            model.tokenizer,
            cfg.trainer.global_train_batch_size // dist.get_world_size(),
            split="validation",
            eval_only=True,
            mask_seq=True,
            max_seq_len=cfg.model.max_seq_len,
            mlm=cfg.model.mlm,
            default_target_ratio=None,
        )

        zeroshot_val_loader = Evaluator(
            label=cfg.eval_dataset.label,
            dataloader=zeroshot_val_loader,
            metric_names=[cfg.eval_dataset.target],
            eval_interval=cfg.eval_dataset.eval_interval,
            device_eval_microbatch_size=cfg.trainer.device_train_microbatch_size,
        )
        if cfg.model.get("log_bpreds", False):
            callbacks.append(
                IGVCallBack(target_eval_label="maize_allele_freq", log_only_N=200)
            )
        eval_dataloaders = [val_loader, zeroshot_val_loader]

    if cfg.model.get("log_bpreds", False) and cfg.maize_dataset is not None:
        maize_val_loader = build_dataloader(
            cfg.maize_dataset,
            model.tokenizer,
            cfg.trainer.global_train_batch_size // dist.get_world_size(),
            split="train",
            eval_only=True,
            mask_seq=False,
            max_seq_len=cfg.model.max_seq_len,
            mlm=cfg.model.mlm,
            default_target_ratio=None,
        )

        maize_val_loader = Evaluator(
            label="maize_chr1",
            dataloader=maize_val_loader,
            eval_interval=cfg.maize_dataset.eval_interval,
            metric_names=[],
            device_eval_microbatch_size=cfg.trainer.device_train_microbatch_size,
        )
        eval_dataloaders.append(maize_val_loader)
        if not os.path.exists(cfg.maize_dataset.save_dir):
            os.makedirs(cfg.maize_dataset.save_dir)

        callbacks.append(
            ChrChunker(
                target_eval_label="maize_chr1", save_dir=cfg.maize_dataset.save_dir
            )
        )

    # Create trainer; see
    # https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.Trainer.html
    print(f"Eval interval: {cfg.trainer.eval_interval}")
    composer_trace_dir = "composer_profiler"
    torch_trace_dir = "torch_profiler"

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_dataloaders,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.trainer.max_duration,
        eval_interval=cfg.trainer.eval_interval,
        callbacks=callbacks,
        loggers=loggers,
        precision=cfg.trainer.precision,
        device_train_microbatch_size=cfg.trainer.device_train_microbatch_size,
        # save_folder=cfg.trainer.get("save_folder"),
        # ave_interval=cfg.trainer.get("save_interval", "1000ba"),
        # ave_num_checkpoints_to_keep=cfg.trainer.get(
        #   "save_num_checkpoints_to_keep", -1
        # ,
        run_name=cfg.run_name,
        autoresume=cfg.trainer.autoresume,
        # profiler=Profiler(
        #    trace_handlers=[
        #        JSONTraceHandler(folder=composer_trace_dir, overwrite=True)
        # ],
        #    schedule=cyclic_schedule(
        #        wait=1,
        #        warmup=1,
        #        active=3,
        #        repeat=1,
        #    ),
        #    torch_prof_folder=torch_trace_dir,
        #    torch_prof_overwrite=True,
        #    torch_prof_memory_filename=None,
        #    torch_prof_with_stack=True,
        # ),
    )

    # Start training
    trainer.fit(reset_time=cfg.trainer.get("reset_time", False))


if __name__ == "__main__":
    run_training()
