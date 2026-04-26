import logging
import torch

import numpy as np

from composer.utils import dist
from datasets import load_dataset
from dataclasses import dataclass
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DataCollatorForLanguageModeling,
)
from typing import cast, Optional, Any

logger = logging.getLogger(__name__)


def get_utf8_char_length(first_byte: int) -> int:
    """
    Determines the number of bytes a UTF-8 character occupies based on its first byte.

    Args:
        first_byte: The integer value (0-255) of the first byte of a potential UTF-8 character.

    Returns:
        The number of bytes the character occupies (1, 2, 3, or 4).
        Returns 0 if the byte is a UTF-8 continuation byte (10xxxxxx), which is not a valid start byte.
    """
    if (first_byte & 0x80) == 0x00:  # 0xxxxxxx (ASCII)
        return 1
    elif (first_byte & 0xE0) == 0xC0:  # 110xxxxx (2-byte character)
        return 2
    elif (first_byte & 0xF0) == 0xE0:  # 1110xxxx (3-byte character)
        return 3
    elif (first_byte & 0xF8) == 0xF0:  # 11110xxx (4-byte character)
        return 4
    else:  # 10xxxxxx (Continuation byte, not a start byte)
        return 0


def chunk_by_max_length(
    batch: dict[str, np.ndarray],
    max_seq_len: int,
    overlap: int = 0,
    add_bos_to_chunks: bool = True,
    add_eos_to_chunks: bool = True,
    pad_to_max_length: bool = True,
    tokenizer=None,
) -> dict[str, list]:
    """
    Chunks a list of tokenized sequences, applying overlap and handling BOS/EOS/padding for each chunk.
    Expects tokenized_output_list to contain dictionaries with 'input_ids' (numpy arrays).
    Written by Gemini, decodes to and from orig sequence multiple times to handle multibyte tokens. Can be
    optimized by likely not worth the effort unless pre-processing is highly compute constrained.
    """
    if tokenizer is None:
        raise ValueError(
            "Tokenizer must be provided for special token indices and padding."
        )

    # Recalculate effective content max length for chunks
    # Note: This uses the tokenizer's current special token setup, assuming it matches 'add_bos_to_chunks' etc.
    # In a real scenario, you'd calculate based on the actual BOS/EOS settings for the chunks.
    space_for_special_tokens = (1 if add_bos_to_chunks else 0) + (
        1 if add_eos_to_chunks else 0
    )
    effective_content_max_length_for_chunk = max_seq_len - space_for_special_tokens

    if effective_content_max_length_for_chunk < 0:
        raise ValueError(
            f"max_seq_len ({max_seq_len}) is too small to accommodate "
            f"{'' if not add_bos_to_chunks else 'BOS '}{'' if not add_eos_to_chunks else 'EOS '}tokens for chunking. "
            f"Requires at least {space_for_special_tokens} spots."
        )

    if overlap >= effective_content_max_length_for_chunk:
        raise ValueError(
            f"Overlap ({overlap}) must be strictly less than effective content max length for chunk "
            f"({effective_content_max_length_for_chunk})."
        )
    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    all_chunks = []
    input_ids = batch["input_ids"]

    for item in input_ids:
        # Original full sequence, assuming it's already encoded by the tokenizer
        # We need to remove any BOS/EOS added by the initial tokenizer for chunking.
        if type(item) is np.ndarray:
            raw_ids = item.tolist()
        elif type(item) is list:
            raw_ids = item
        else:
            raise Exception(
                f"Type {type(item)} of {item} not handled, chunking may be incorrect"
            )
        # Filter out BOS/EOS if they were added during the initial tokenization
        content_ids = [
            idx
            for idx in raw_ids
            if idx not in [tokenizer.bos_idx, tokenizer.eos_idx, tokenizer.pad_idx]
        ]

        if len(content_ids) <= effective_content_max_length_for_chunk:
            # If the original content fits within one chunk (after removing initial BOS/EOS), just process it.
            # We re-tokenize it to ensure correct BOS/EOS/padding for the final chunk.
            encoded_single = tokenizer.encode(
                [tokenizer.decode(content_ids, skip_special_tokens=True)],
                add_bos=add_bos_to_chunks,
                add_eos=add_eos_to_chunks,
                padding=pad_to_max_length,
                max_length=max_seq_len,
            )[0]["input_ids"]
            all_chunks.append(encoded_single)
            continue

        # Apply chunking with overlap
        start_idx = 0
        stride = effective_content_max_length_for_chunk - overlap
        if stride <= 0:
            stride = 1  # Should be caught by validation above, but as a safeguard

        while start_idx < len(content_ids):
            end_idx = min(
                start_idx + effective_content_max_length_for_chunk, len(content_ids)
            )
            # catch our overlap accidently put us to start in a multi-byte token
            # advance until we're out of it
            # slightly less context, respects our max token len bound
            while get_utf8_char_length(content_ids[start_idx]) == 0:
                start_idx += 1
                if start_idx >= len(content_ids):  # out last content was a partial MBC
                    break
            if start_idx >= len(content_ids):
                break
            # if our last byte is within a multi byte char, reverse until we find it's start
            offset = 0
            while get_utf8_char_length(content_ids[end_idx - 1]) == 0:
                end_idx -= 1
                offset += 1
            # if out end_idx is at a multi-byte token that would put us over the seq len
            if get_utf8_char_length(content_ids[end_idx - 1]) + end_idx > min(
                start_idx + effective_content_max_length_for_chunk, len(content_ids)
            ):
                # go back one byte and leave it for the next chunk
                end_idx -= 1
                offset += 1
            else:
                # multibyte token fits, reset out end position
                # print(
                #    f"Orig range: {start_idx}: {
                #        min(
                #            start_idx + effective_content_max_length_for_chunk,
                #            len(content_ids),
                #        )
                #    }"
                # )
                # print(
                #    f"Token at position {end_idx - 1} fits in {get_utf8_char_length(content_ids[end_idx - 1])} bytes, updating to that end with a seq of len {len(content_ids)}"
                # )
                # print(
                #    f"New end bytes: {content_ids[end_idx - 1 : end_idx - 1 + get_utf8_char_length(content_ids[end_idx - 1])]}"
                # )
                # print(
                #    tokenizer.decode(
                #        content_ids[
                #            end_idx - 1 : end_idx
                #            - 1
                #            + get_utf8_char_length(content_ids[end_idx - 1])
                #        ]
                #    )
                # )
                end_idx += get_utf8_char_length(content_ids[end_idx - 1]) - 1
                offset = 0

            chunk_content = content_ids[start_idx:end_idx]

            # Convert chunk_content back to string to let the tokenizer re-add BOS/EOS and pad
            decoded_chunk_content = tokenizer.decode(
                chunk_content, skip_special_tokens=True
            )
            encoded_chunk = tokenizer.encode(
                [decoded_chunk_content],
                add_bos=add_bos_to_chunks,
                add_eos=add_eos_to_chunks,
                padding=pad_to_max_length,
                max_length=max_seq_len,
            )[0]["input_ids"]
            all_chunks.append(encoded_chunk)

            if end_idx == len(content_ids):
                break

            start_idx += stride - offset

    return {"input_ids": all_chunks}


def build_dataloader(
    cfg: Optional[DictConfig] = None,
    tokenizer=None,
    batch_size: Optional[int] = 1,
    max_seq_len: Optional[int] = 512,
    mlm: Optional[bool] = True,
    default_target_ratio: Optional[int] = None,
    split: str = "train",
    eval_only: bool = False,
    mask_seq: bool = False,
    k: Optional[int] = None,  # only for kmer tok
    **kwargs,
):
    """Build data loader for masked language modeling.
    Args:
    - cfg:  dataset cfg (from yaml)
    - tokenizer:  the model tokenizer
    - batch_size:  device batch size to use
    - max_seq_len:  maximum seq len (pad to this length)
    - mlm:  whether to use MLM or NTP collator
    - default_target_ratio:  compression factor expected by HNet
    - split: the split to parse (train, evaluation, test)
    - eval_only:  a dataset only used for eval (e.g. default to train and masking seq) #TODO: refactor out
    - mask_seq: whether to mask out var idxes if a ref and alt token are given (for DNA VEP)
    - k: the k to tokenize sequence into (for use with k-mer tokenizers only)
    """
    print(f"Using config {cfg}")
    print(kwargs)
    print(tokenizer)
    print(mlm)
    print(split)

    # Load dataset
    dataset_name_or_path = (
        cfg.data_local if cfg.data_local is not None else cfg.data_remote
    )
    if eval_only:
        dataset_name_or_path = cfg.data_remote
        # split naming error
        split = "train"
        mask_seq = mask_seq
    val = False
    if split == "validation":
        split = "train"
        val = True
    dataset = load_dataset(
        dataset_name_or_path,
        # data_files={split: os.path.join(cfg.data_local, f"{split}.txt")},
        split=split,
    )
    val_split = len(dataset) // 10
    if val:
        dataset = dataset.select(list(range(0, val_split)))
    else:
        dataset = dataset.select(list(range(val_split, len(dataset))))

    def anon(ex):
        dicts = tokenizer.encode(
            ex["text"], add_bos=False, add_eos=False, padding=False
        )  # list of dicts
        ret = {"input_ids": [x["input_ids"] for x in dicts]}
        return ret

    tokenized_not_chunked = dataset.map(
        anon,
        batched=True,
        batch_size=4,
    )  # each sequence fully tokenized into bytes, no special toks (added in after chunking), [N, max(L1, \ldots, LN)]

    chunked = tokenized_not_chunked.map(
        chunk_by_max_length,
        fn_kwargs={
            "max_seq_len": max_seq_len,
            "overlap": cfg.overlap,
            "add_bos_to_chunks": True,
            "add_eos_to_chunks": True,
            "pad_to_max_length": True,
            "tokenizer": tokenizer,
        },
        batched=True,
        batch_size=2048,  # Smaller if chunking factor is high
        num_proc=8,  # Should set to num cores
        remove_columns=tokenized_not_chunked.column_names,
    )

    class LanguageModelingDataset(Dataset):
        def __init__(self, tokenized_data, default_target_ratio):
            self.tokenized_data = tokenized_data
            self.default_target_ratio = default_target_ratio

        def __len__(self):
            return len(self.tokenized_data)

        def __getitem__(self, idx):
            # The 'input_ids' are currently numpy arrays; convert them to torch tensors.
            item = self.tokenized_data[idx]
            # Use torch.long for input_ids as they are typically used as indices in embeddings.
            encoding = {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in item.items()
            }
            if self.default_target_ratio is not None:
                encoding["target_ratio"] = torch.tensor(self.default_target_ratio)
            return encoding

    tokenized_dataset = LanguageModelingDataset(
        tokenized_data=chunked, default_target_ratio=default_target_ratio
    )
    sampler = dist.get_sampler(tokenized_dataset, shuffle=(not val))

    def custom_data_collator(
        features,
        mlm_probability,
        mask_replace_prob,
        random_replace_prob,
        tokenizer,
        mlm=False,
        default_target_ratio=None,
    ):
        batch = tokenizer.pad(features, return_tensors="pt")

        if mlm:
            # Original MLM logic from DataCollatorForLanguageModeling
            inputs, labels = (
                DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=True,
                    mlm_probability=mlm_probability,
                    mask_replace_prob=mask_replace_prob,
                    random_replace_prob=random_replace_prob,
                    return_tensors="pt",
                )
                .torch_call(features)
                .values()
            )
        else:
            # Causal Language Modeling (NTP) with manual label shifting
            labels = batch["input_ids"].clone()
            # Shift labels to the left; the first token is the target for the second token, etc.
            # The last token has no target, so it's masked out (-100).
            labels[:, :-1] = batch["input_ids"][:, 1:]
            labels[
                :, -1
            ] = -100  # Mask out the last token as there's no next token to predict
            inputs = batch["input_ids"]
        ret = {
            "input_ids": inputs,
            "labels": labels,
            "loss_weights": torch.ones_like(labels),  # for text ignore repeat_weight
        }
        if default_target_ratio is not None:
            ret["target_ratio"] = torch.tensor(
                [default_target_ratio] * len(inputs),
                device=inputs.device,
                dtype=torch.float,
            )

        return ret

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        num_workers=0,  # cfg.num_workers,
        collate_fn=lambda features: custom_data_collator(
            features=features,
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=cfg.mlm_probability,
            mask_replace_prob=cfg.mask_replace_prob,
            random_replace_prob=cfg.random_replace_prob,
            default_target_ratio=default_target_ratio,
        ),
        sampler=sampler,
        pin_memory=True,
    )


@dataclass
class TestConfig:
    data_local: Optional[str] = None
    data_remote: Optional[str] = "codelion/fineweb-edu-1B"
    overlap: int = 16
    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.10
