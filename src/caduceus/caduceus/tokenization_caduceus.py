"""Character tokenizer for Hugging Face."""

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
    def pre_tokenize(self, pretok):
        def split_on_bytes(_, normalized):
            s = normalized.get()
            i = 0
            agg = []
            while i < len(s):
                if s[i] != "[":
                    agg.append((s[i], (i, i + 1)))
                    i += 1
                else:
                    spec_token = ""
                    start_idx = i
                    while s[i] != "]" and i < len(s):
                        spec_token = spec_token + s[i]
                        i += 1
                    if i == len(s):
                        raise Exception(
                            f"Error processing string {s}, found unmatched '['"
                        )
                    else:
                        spec_token = spec_token + "]"
                        agg.append((spec_token, (start_idx, i + 1)))
            return agg
            return [(c, (i, i + 1)) for i, c in enumerate(s)]

        pretok.split(split_on_bytes)


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
