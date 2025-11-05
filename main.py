"""Unified CLI for Caduceus minimal training example."""

import logging
import os
import random
from pathlib import Path
from typing import cast, Optional, Any

import hydra_setup  # register resolvers for hydra

import fire
import hydra
import torch
from caduceus import CaduceusConfig, CaduceusForMaskedLM, CaduceusTokenizer
from collections import namedtuple
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
from composer.models import HuggingFaceModel
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer.utils import dist, reproducibility
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchmetrics import PearsonCorrCoef
from torchmetrics.aggregation import MeanMetric, RunningMean
from torchmetrics.classification import MulticlassAccuracy
from transformers import (
    DataCollatorForLanguageModeling,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import (
    AttnConfig,
    SSMConfig,
    HNetConfig,
)
from hnet.utils.tokenizers import ByteTokenizer


logger = logging.getLogger(__name__)


class ComposerWrapper(HuggingFaceModel):
    def __init__(self, *args, mlm=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_pcc = PearsonCorrCoef()
        self.val_loss = MeanMetric()
        self.val_loss.tag = ""
        self.train_ar_loss = RunningMean()
        self.train_ar_loss.tag = "ar"
        self.val_ar_loss = MeanMetric()
        self.val_ar_loss.tag = "ar"
        self.train_ratio_loss = RunningMean()
        self.train_ratio_loss.tag = "ratio"
        self.val_ratio_loss = MeanMetric()
        self.val_ratio_loss.tag = "ratio"

        self.train_acc = MulticlassAccuracy(average="micro", ignore_index=-100)
        self.train_acc.tag = "acc"

        self.val_acc = MulticlassAccuracy(average="micro", ignore_index=-100)
        self.val_acc.tag = "acc"

        self.mlm = mlm

    def eval_forward(self, batch, outputs=None):
        alt_outputs = None  # hacky placeholder for second fwd pass for ALT seqs in VEP
        if outputs:
            return outputs
        if "ref_id" in batch and not self.mlm:
            # if we're doning a VEP task, it makes life much easier to do the ALT fwd pass here and shove it into the output tuple
            ref_bp_id = batch["ref_id"]  # [B]
            ref_ids = batch["input_ids"].detach().clone()
            B, seq_len = ref_ids.shape
            var_idx = seq_len // 2
            # update (possibly masked) token with ref
            ref_ids[:, var_idx] = ref_bp_id

            assert torch.all(ref_ids[:, var_idx] == ref_bp_id), (
                f"REF bps from batch do not match the input ids, IDS: {ref_ids[:, var_idx - 1 : var_idx + 2]}, ref_bps: {ref_bp_id}"
            )
            alt_bp_id = batch["alt_id"]  # [B]
            alt_ids = batch["input_ids"].detach().clone()
            B, seq_len = alt_ids.shape
            # update (possibly masked) token with alt
            alt_ids[:, var_idx] = alt_bp_id
            assert torch.all(ref_bp_id != alt_bp_id), (
                f"REF and ALT bps are the same bp for some batches. REF: {ref_bp_id[ref_bp_id == alt_bp_id]} ALT: {alt_bp_id[ref_bp_id == alt_bp_id]}"
            )
            assert torch.all(ref_ids[:, :var_idx] == alt_ids[:, :var_idx]), (
                "Not all ids before the variant site match"
            )
            assert torch.all(ref_ids[:, var_idx + 1 :] == alt_ids[:, var_idx + 1 :]), (
                "Not all ids after the variant site match"
            )
            assert torch.all(ref_ids[:, var_idx] != alt_ids[:, var_idx]), (
                f"Some variants have the same ALT and REF bp, REF: {ref_ids[:, var_idx][ref_ids[:, var_idx] == alt_ids[:, var_idx]]} ALT: {alt_ids[:, var_idx][ref_ids[:, var_idx] == alt_ids[:, var_idx]]}"
            )

            alt_outputs = self.model(input_ids=alt_ids)

            new_batch = {
                "input_ids": ref_ids,
                "labels": batch["labels"],
                "loss_weights": batch["loss_weights"],
            }

            if "target_ratio" in batch:
                new_batch["target_ratio"] = batch["target_ratio"]

            batch = new_batch

        outputs = self.model(**batch)

        if alt_outputs is not None:
            # hacky way to add the alternate input to the output tuple posthoc
            outputs = outputs._asdict()
            outputs["alt_outputs"] = alt_outputs
            new_namedtuple = namedtuple("CausalLMOutputsforZS", outputs.keys())
            outputs = new_namedtuple(**outputs)
        return outputs

    def update_metric(self, batch, outputs, metric) -> None:
        """
        Update metric by returning as a socre the (log) ref/alt probabilities
        Args:
            batch: dict[str, Tensor] the input batch.
            outputs: MaskedLMOutput['logits': Tensor(batch, seq_len, vocab_len), 'loss': float]
            metric: torchmetrics.Metric the metric we're updating
        """
        # TODO: Redo by shoving all the evals for each split into a collection class?
        if (
            len(batch.keys()) == 5 or "ref_id" not in batch
        ):  # not in the zero-shot eval task
            val = None
            if metric.tag == "ar":
                val = outputs.ar_loss if not self.mlm else None
            elif metric.tag == "ratio":
                val = outputs.ratio_loss if not self.mlm else None
            elif metric.tag == "acc":
                B, L, V = outputs.logits.shape
                preds = outputs.logits.softmax(dim=-1).argmax(dim=-1)
                labels = batch["labels"]
                if self.mlm:
                    # if MLM only count
                    preds[labels == -100] = 0
                    labels[labels == -100] = 0
                metric.update(preds.view(-1), labels.view(-1))
                return
            else:
                val = outputs.loss
            if val is not None:
                metric.update(value=val)
            return
        probs = outputs.logits.softmax(dim=-1)
        batch_size, seq_len, vocab_len = outputs.logits.shape

        if self.mlm:
            ref_bp = batch["ref_id"]  # [batch_size]
            ref_prob = torch.gather(
                probs[:, (seq_len // 2) - 1, :], dim=1, index=ref_bp.unsqueeze(1)
            ).squeeze(1)

            alt_bp = batch["alt_id"]  # [batch_size]
            alt_prob = torch.gather(
                probs[:, (seq_len // 2) - 1, :], dim=1, index=alt_bp.unsqueeze(1)
            ).squeeze(1)

            assert len(probs.shape) == 3, (
                f"Expected probs of shape [batch, seq_len, vocba_len], found {probs.shape}"
            )
            assert (probs < 0).sum() == 0, (
                f"Found probabilities less than 0 in probs: {probs[probs < 0]}"
            )
            assert (probs > 1).sum() == 0, (
                f"Found probabilities greater than 1 in probs: {probs[probs > 1]}"
            )
            assert ((torch.abs(probs.sum(dim=-1)) - 1) > 1e-5).sum() == 0, (
                f"Probabilities in probs do not normalize to 1: {probs[(torch.abs(probs.sum(dim=-1)) - 1) > 1e-5]}"
            )
            score = torch.log(alt_prob / ref_prob)
        else:
            alt_probs = outputs.alt_outputs.logits.softmax(dim=-1)
            ref_bp = batch["ref_id"]  # [B]
            alt_bp = batch["alt_id"]  # [B]
            input_ids = batch["input_ids"]  # [B, L]
            var_idx = seq_len // 2

            assert torch.all(input_ids[:, var_idx] == ref_bp), (
                f"REF bps from batch do not match the input ids, IDS: {input_ids[:, var_idx - 1 : var_idx + 2]}, ref_bps: {ref_bp}"
            )
            assert torch.all(ref_bp != alt_bp), (
                "Not all REF and  ALT bps are different, error in pre-processing"
            )

            ref_log_probs = torch.log(
                torch.gather(probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(2)
            )
            alt_ids = input_ids.detach().clone()  # [B, L]
            alt_ids[:, var_idx] = alt_bp
            assert torch.all(input_ids[:, :var_idx] == alt_ids[:, :var_idx]), (
                "Not all ids before the variant site match"
            )
            assert torch.all(
                input_ids[:, var_idx + 1 :] == alt_ids[:, var_idx + 1 :]
            ), "Not all ids after the variant site match"
            ref_check = input_ids[:, var_idx]
            alt_check = alt_ids[:, var_idx]
            not_match = ref_check != alt_check

            assert torch.all(not_match), (
                f"Some REF and ALT sequences have the same BP at the variant site, REFs: {ref_check[~not_match]} ({ref_check[~not_match].shape}), ALTs: {alt_check[~not_match]} ({alt_check[~not_match].shape})"
            )

            alt_log_probs = torch.log(
                torch.gather(alt_probs, dim=-1, index=alt_ids.unsqueeze(-1)).squeeze(2)
            )
            ref_pll = ref_log_probs.mean(dim=-1)
            alt_pll = alt_log_probs.mean(dim=-1)
            score = alt_pll - ref_pll

        maf = batch["MAF"]  # the 'label' [batch_size]

        metric.update(preds=score, target=maf)

    def get_metrics(self, is_train=False):
        if is_train:
            return {
                "ARLoss": self.train_ar_loss,
                "RatioLoss": self.train_ratio_loss,
                "Accuracy": self.train_acc,
            }
        return {
            "PearsonCorrCoef": self.val_pcc,
            "EvalLoss": self.val_loss,
            "ARLoss": self.val_ar_loss,
            "RatioLoss": self.val_ratio_loss,
            "Accuracy": self.val_acc,
        }


def build_model(cfg: DictConfig):
    """Build Caduceus model from config."""
    # TODO: Redo this whole thing, switch to hydra for cfg mngmnt and create unified instiation
    if cfg.from_pretrained:
        model_config = AutoConfig.from_pretrained(
            cfg.pretrained_name_or_path, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_name_or_path, trust_remote_code=True
        )
        tokenizer.characters = "atcg"
        if cfg.from_scratch:
            model = AutoModelForMaskedLM.from_config(
                model_config, trust_remote_code=True
            )
        else:
            model = AutoModelForMaskedLM.from_pretrained(
                cfg.pretrained_name_or_path, trust_remote_code=True
            )
    else:
        if cfg.hnet_model:
            model_config = cfg.get("model")
            model_config = OmegaConf.to_container(model_config, resolve=True)
            # attn_cfg = AttnConfig(**model_config.get("attn_cfg"))
            # ssm_cfg = SSMConfig(**model_config.get("ssm_cfg"))
            hnet_cfg = HNetConfig(**model_config)
            # Create model
            model = HNetForCausalLM(hnet_cfg, dtype=torch.bfloat16)
            # Use existing tokenizer instead of byte tokenizer (dna is already in bytes)
            # tokenizer = ByteTokenizer()
            tokenizer = CaduceusTokenizer(model_max_length=cfg.max_seq_len)

        else:
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
    print(f"Total number of parameters: {sum([x.numel() for x in model.parameters()])}")
    print(
        f"Total number of trainable parameters: {sum([x.numel() for x in model.parameters() if x.requires_grad])}"
    )

    logger.info(f"Model config: {model_config}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer.get_vocab())}")
    logger.info(f"Model vocab size: {model.config.vocab_size}")
    logger.info(f"Tokenizer vocab: {tokenizer.get_vocab()}")
    logger.info("=========================\n")

    # return HuggingFaceModel(model, tokenizer, eval_metrics=None)
    return ComposerWrapper(model, tokenizer, mlm=cfg.mlm)


def build_dataloader(
    cfg: DictConfig,
    tokenizer,
    batch_size: int,
    max_seq_len: int,
    mlm: bool,
    default_target_ratio: Optional[int] = None,
    split: str = "train",
    eval_only: bool = False,
):
    """Build data loader for masked language modeling."""

    # Load dataset
    dataset_name_or_path = (
        cfg.data_local if cfg.data_local is not None else cfg.data_remote
    )
    mask_seq = False
    if eval_only:
        dataset_name_or_path = cfg.data_remote
        # split naming error
        split = "train"
        mask_seq = True
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
                var_idx = (len(sequence) // 2) - 1
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
        max_seq_len,
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
    # model = build_model(cfg)
    num_params = sum([x.numel() for x in model.parameters()])
    num_trainable_params = sum(
        [x.numel() for x in model.parameters() if x.requires_grad]
    )

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
        CheckpointSaver(
            weights_only=False,
            folder=cfg.trainer.get("save_folder"),
            save_interval=cfg.trainer.get("save_interval", "1000ba"),
            num_checkpoints_to_keep=cfg.trainer.get("save_num_checkpoints_to_keep", -1),
        ),
        RuntimeEstimator(),
        MemoryMonitor(),
    ]

    # Build loggers
    loggers = []
    if "wandb" in cfg.get("loggers", {}):
        from dotenv import load_dotenv

        load_dotenv(cfg.paths.env_path)
        api_key = cfg.loggers.wandb.api_key
        if api_key is None and "WANDB_API_KEY" not in os.environ:
            raise Exception(
                "WANDB logger instantiated by not API key was provided, make sure .env is set up properly"
            )
        os.environ["WANDB_API_KEY"] = api_key
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
    )
    eval_dataloaders = [val_loader]
    if cfg.eval_dataset is not None:
        zeroshot_val_loader = build_dataloader(
            cfg.eval_dataset,
            model.tokenizer,
            cfg.trainer.global_train_batch_size // dist.get_world_size(),
            split="validation",
            eval_only=True,
            max_seq_len=cfg.model.max_seq_len,
            mlm=cfg.model.mlm,
            default_target_ratio=None,
        )
        zeroshot_val_loader = Evaluator(
            label="maize_allele_freq",
            dataloader=zeroshot_val_loader,
            metric_names=["PearsonCorrCoef"],
        )
        eval_dataloaders = [val_loader, zeroshot_val_loader]

    # Create trainer; see
    # https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.Trainer.html
    print(f"Eval interval: {cfg.trainer.eval_interval}")
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
        autoresume=True,
    )

    # Start training
    trainer.fit()


if __name__ == "__main__":
    run_training()
