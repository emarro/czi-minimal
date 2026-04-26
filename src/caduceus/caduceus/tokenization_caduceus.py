"""Character tokenizer for Hugging Face."""

import torch

import numpy as np

from typing import List, Optional, Dict, Sequence, Tuple

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from itertools import product

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import (
    ByteLevel,
    CharDelimiterSplit,
    Split,
    PreTokenizer,
    Whitespace,
)


class CaduceusTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]
    _auto_map = {
        "AutoTokenizer": ["caduceus_tokenization_caduceus.CaduceusTokenizer", None]
    }  # second entry should be fasttokenizer version

    def __init__(
        self,
        model_max_length: int,
        characters: Sequence[str] = ("A", "C", "G", "T", "N"),
        complement_map=None,
        bos_token="[BOS]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        """Character tokenizer for Hugging Face transformers.

        Adapted from https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf/blob/main/tokenization_hyena.py
        Args:
            model_max_length (int): Model maximum sequence length.
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following is a list of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            complement_map (Optional[Dict[str, str]]): Dictionary with string complements for each character.
        """
        if complement_map is None:
            complement_map = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        self.characters = characters
        self.model_max_length = model_max_length

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        padding_side = kwargs.pop("padding_side", "left")

        self._complement_map = {}
        for k, v in self._vocab_str_to_int.items():
            complement_id = (
                self._vocab_str_to_int[complement_map[k]]
                if k in complement_map.keys()
                else v
            )
            self._complement_map[self._vocab_str_to_int[k]] = complement_id

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    @property
    def complement_map(self) -> Dict[int, int]:
        return self._complement_map

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text.upper())  # Convert all base pairs to uppercase

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(
            tokens
        )  # Note: this operation has lost info about which base pairs were originally lowercase

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        # cls = [self.cls_token_id]
        result = token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    # Fixed vocabulary with no vocab file
    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple:
        return ()


class DNAByteSplit:
    def __init__(self, k):
        self.k = k

    def split_on_bytes(self, normalized):
        s = normalized.upper()  # .get()
        i = 0
        agg = []
        # Split into DNA and special Tokens (wrapped in [])
        while i < len(s):
            if s[i] != "[":
                if i + self.k > len(s):
                    agg.append((s[i], (i, i + 1)))
                    i += 1
                elif "[" in s[i : i + self.k] or "]" in s[i : i + self.k]:
                    agg.append((s[i], (i, i + 1)))
                    i += 1
                else:
                    agg.append((s[i : i + self.k], (i, i + self.k)))
                    i += self.k
            else:
                spec_token = ""
                start_idx = i
                while s[i] != "]" and i < len(s):
                    spec_token = spec_token + s[i]
                    i += 1
                if i == len(s):
                    raise Exception(f"Error processing string {s}, found unmatched '['")
                else:
                    spec_token = spec_token + "]"
                    agg.append((spec_token, (start_idx, i + 1)))
                    i += 1
        # aggregate DNA into k-mers if possible, otherwise leave single BP
        return agg
        return [(c, (i, i + 1)) for i, c in enumerate(s)]


class KMerTokenizer(PreTrainedTokenizerFast):
    """
    Caduceus Tokenizer for K-Mers.

    """

    model_input_names = ["input_ids"]
    _auto_map = {
        "AutoTokenizer": ["caduceus_tokenization_caduceus.KMerTokenizer", None]
    }  # second entry should be fasttokenizer version

    def __init__(
        self,
        k: int,
        model_max_length: int,
        characters: Sequence[str] = ("A", "C", "G", "T", "N"),
        complement_map=None,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        if complement_map is None:
            complement_map = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        k_mers = characters
        for k_idx in range(1, k):
            k_mers = tuple(["".join(x) for x in product(k_mers, characters)])
        for k_mer in k_mers:
            if len(k_mer) == k:
                complement_map[k_mer] = "".join(
                    [complement_map[char] for char in k_mer]
                )
        self.custom_pretok = DNAByteSplit(k=k)
        # print(f"Characters: {characters}")
        # print(f"KMers: {list(k_mers)}")
        # print(f"Complement map: {complement_map}")
        new_characters = k_mers + characters
        new_vocab_list = [
            bos_token,
            eos_token,
            sep_token,
            mask_token,
            cls_token,
            pad_token,
            unk_token,
        ] + list(new_characters)
        # print(new_vocab_list)
        new_vocab = {k: idx for idx, k in enumerate(new_vocab_list)}
        # print(new_vocab)
        tok = Tokenizer(
            model=WordPiece(
                new_vocab,
                unk_token=unk_token,
                max_input_chars_per_word=100_000,  # k + 6,
                continuing_subword_prefix="",
            )
        )
        tok.pre_tokenizer = Whitespace()
        self.characters = new_vocab_list
        self._vocab_str_to_int = new_vocab
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        padding_side = kwargs.pop("padding_side", "left")
        self._complement_map = {}
        for k, v in self._vocab_str_to_int.items():
            complement_id = (
                self._vocab_str_to_int[complement_map[k]]
                if k in complement_map.keys()
                else v
            )
            self._complement_map[self._vocab_str_to_int[k]] = complement_id

        super().__init__(
            tokenizer_object=tok,
            unk_token=unk_token,
            mask_token=mask_token,
            pad_token=pad_token,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    @property
    def complement_map(self) -> Dict[int, int]:
        return self._complement_map

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text.upper())  # Convert all base pairs to uppercase

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(
            tokens
        )  # Note: this operation has lost info about which base pairs were originally lowercase

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        if not isinstance(text, (list, tuple)):  # not batched, batch it
            text = [text]
        return_dict = {"input_ids": []}
        input_ids = []
        offset_mapping = []
        for seq in text:
            # print(f"Input text {text}")
            split_tokens = self.custom_pretok.split_on_bytes(seq)
            # print(f"Split tokens: {split_tokens}")
            inner_input_ids = []
            inner_offset_mapping = []
            for x, span in split_tokens:
                inner_input_ids.append(self._vocab_str_to_int[x])
                inner_offset_mapping.append(span)
            input_ids.append(inner_input_ids)
            offset_mapping.append(inner_offset_mapping)

        device = (
            kwargs["device"]
            if "device" in kwargs
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        return_dict["input_ids"] = torch.tensor(
            input_ids,
            dtype=torch.long,
        )
        if return_offsets_mapping:
            return_dict["offset_mapping"] = torch.tensor(
                offset_mapping,
                dtype=torch.long,
            )
        return return_dict
        full_seq = [(self._vocab_str_to_int[x], span) for x, span in split_tokens]
        # full_seq = super().__call__(split_tokens, **kwargs)
        print(f"Full Tokenized: {full_seq}")
        assert False
        return full_seq


class ByteTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]
    _auto_map = {
        "AutoTokenizer": ["caduceus_tokenization_caduceus.ByteTokenizer", None]
    }  # second entry should be fasttokenizer version

    def __init__(self, model_max_length=8192, padding_side="right"):
        self.bos_idx = 254
        self.eos_idx = 255
        self.pad_idx = 256
        self.mask_idx = 257
        self.cls_idx = 258
        self.sep_idx = 259
        self.unk_idx = 260
        self.dtype = (
            np.uint16
        )  # Changed from np.uint8 to np.uint16 to accommodate special tokens

        vocab = {}
        for i in range(254):
            vocab[chr(i)] = i

        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"

        vocab[self.bos_token] = self.bos_idx
        vocab[self.eos_token] = self.eos_idx
        vocab[self.pad_token] = self.pad_idx
        vocab[self.mask_token] = self.mask_idx
        vocab[self.cls_token] = self.cls_idx  # Add to vocab
        vocab[self.sep_token] = self.sep_idx  # Add to vocab
        vocab[self.unk_token] = self.unk_idx  # Add to vocab
        self._vocab = vocab

        super().__init__(
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,  # Pass the unk_token here
            pad_token=self.pad_token,
            cls_token=self.cls_token,  # Pass the cls_token here
            sep_token=self.sep_token,  # Pass the sep_token here
            model_max_length=model_max_length,  # Added argument
            padding_side=padding_side,  # Added argument
        )

    @property
    def vocab_size(self):
        return 261  # 0-253 for regular bytes, rest for special toks

    # Fixed vocabulary with no vocab file

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple:
        return ()

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def _tokenize(self, text: str) -> list[str]:
        tokens = []
        for b in text.encode("utf-8"):
            tokens.append(chr(b))
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        if token == self.bos_token:
            return self.bos_idx
        if token == self.eos_token:
            return self.eos_idx
        if token == self.pad_token:
            return self.pad_idx
        if token == self.mask_token:
            return self.mask_idx
        if token == self.cls_token:
            return self.cls_idx  # Add to vocab
        if token == self.sep_token:
            return self.sep_idx  # Add to vocab
        if token == self.unk_token:
            return self.unk_idx  # Add to vocab
        if len(token) == 1:
            byte_id = ord(token)
            if 0 <= byte_id < 254:
                return byte_id
        raise KeyError(f"Token '{token}' not in vocabulary.")

    def _convert_id_to_token(self, index: int) -> str:
        if index == self.bos_idx:
            return self.bos_token
        if index == self.eos_idx:
            return self.eos_token
        if index == self.pad_idx:
            return self.pad_token
        if index == self.mask_idx:
            return self.mask_token
        if index == self.cls_idx:
            return self.cls_token  # Add to vocab
        if index == self.sep_idx:
            return self.sep_token  # Add to vocab
        if index == self.unk_idx:
            return self.unk_token  # Add to vocab
        if 0 <= index < 254:
            return chr(index)
        # If it's a special token ID, but not explicitly handled as a byte char range
        raise ValueError(f"ID '{index}' not in vocabulary.")

    def __len__(self):
        return self.vocab_size

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(
        self,
        seqs: list[str],
        add_bos: bool = False,
        add_eos: bool = False,
        padding: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> list[dict[str, np.ndarray]]:
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])

            text_byte_ids = list(bytearray(text_byte))

            if padding:
                current_max_length = (
                    max_length if max_length is not None else self.model_max_length
                )
                if len(text_byte_ids) < current_max_length:
                    # Pad to the right (default for `padding_side`) or left if specified
                    if self.padding_side == "right":
                        text_byte_ids = text_byte_ids + [self.pad_idx] * (
                            current_max_length - len(text_byte_ids)
                        )
                    elif self.padding_side == "left":
                        text_byte_ids = [self.pad_idx] * (
                            current_max_length - len(text_byte_ids)
                        ) + text_byte_ids
                elif len(text_byte_ids) > current_max_length:
                    # Truncate if too long
                    if self.padding_side == "right":
                        text_byte_ids = text_byte_ids[:current_max_length]
                    elif self.padding_side == "left":
                        text_byte_ids = text_byte_ids[-current_max_length:]

            input_ids_array = np.array(text_byte_ids, dtype=self.dtype)

            if return_tensors == "pt":
                total_outputs.append({"input_ids": torch.tensor(input_ids_array)})
            else:
                total_outputs.append({"input_ids": input_ids_array})

        return total_outputs

    def decode(
        self,
        tokens: np.ndarray | list[int],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        # Define all special token IDs for filtering
        all_special_ids = [
            self.bos_idx,
            self.eos_idx,
            self.pad_idx,
            self.mask_idx,
            self.cls_idx,
            self.sep_idx,
            self.unk_idx,
        ]

        if skip_special_tokens:
            # Filter out all special tokens and then decode the remaining byte IDs
            filtered_tokens = [t for t in tokens if t not in all_special_ids]
            return bytearray(filtered_tokens).decode("utf-8", **kwargs)
        else:
            # When not skipping special tokens, we want to represent them as strings
            # while correctly decoding actual byte sequences.
            decoded_parts = []
            current_byte_segment = []

            for token_id in tokens:
                if token_id in all_special_ids:
                    # If a special token is encountered, decode any accumulated byte segment
                    if current_byte_segment:
                        decoded_parts.append(
                            bytearray(current_byte_segment).decode("utf-8", **kwargs)
                        )
                        current_byte_segment = []  # Reset for the next byte segment
                    # Append the string representation of the special token
                    decoded_parts.append(self._convert_id_to_token(token_id))
                elif 0 <= token_id < 254:  # This is a valid UTF-8 byte ID
                    current_byte_segment.append(token_id)
                # Any other case (like ID > 253 but not in all_special_ids) implies an unhandled token,
                # but with our current setup, all IDs >= 254 are indeed special tokens.

            # Decode any remaining byte segment at the end of the list
            if current_byte_segment:
                decoded_parts.append(
                    bytearray(current_byte_segment).decode("utf-8", **kwargs)
                )

            return "".join(decoded_parts)
