from dataclasses import dataclass, field
from typing import List, Union
from transformers import PretrainedConfig


class AttnConfig:
    def __init__(
        self,
        num_heads: List = list(),
        rotary_emb_dim: List = list(),
        window_size: List = list(),
    ):
        self.num_heads = num_heads
        self.rotary_emb_dim = rotary_emb_dim
        self.window_size = window_size


class SSMConfig:
    def __init__(
        self,
        d_conv: int = 4,
        expand: int = 2,
        d_state: int = 128,
        chunk_size: int = 256,
    ):
        self.d_conv = d_conv
        self.expand = expand
        self.d_state = d_state
        self.chunk_size = chunk_size


# @dataclass
# class HNetConfig(PretrainedConfig):
#    arch_layout: List[Union[str, List]] = field(default_factory=list)
#    d_model: List[int] = field(default_factory=list)
#    # intermediate dimension for the FFNs (0 indicates no FFN)
#    d_intermediate: List[int] = field(default_factory=list)
#    vocab_size: int = 256
#    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
#    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
#    tie_embeddings: bool = False
#    pad_token_id: int = -100  # pad id for loss fn
#    ratio_loss_weight: float = 0.03  # alpha in Hnet Paper
#    use_return_dict: bool = False
#    log_bpreds: bool = True
#    auto_map = {
#        "AutoConfig": "hnet.hnet.models.confg_hnet.HNetConfig",
#        "AutoModel": "hnet.hnet.models.mixer_seq.HNetForCausalLM",
#        "AutoModelForCausalLM": "hnet.hnet.models.mixer_seq.HNetForCausalLM",
#        "AutoModelForMaskedLM": "hnet.hnet.models.mixer_seq.HNetForCausalLM",
#    }
#
#    def __post_init__(self):
#        super().__init__()


class HNetConfig(PretrainedConfig):
    auto_map = {
        "AutoConfig": "models_config_hnet.HNetConfig",
        "AutoModel": "models_mixer_seq.HNetForCausalLM",
        "AutoModelForCausalLM": "models_mixer_seq.HNetForCausalLM",
        "AutoModelForMaskedLM": "models_mixer_seq.HNetForCausalLM",
    }

    def __init__(
        self,
        arch_layout: List[
            Union[str, List]
        ] = list(),  # remove dataclass fields, see transformers PR 678
        d_model: List[int] = list(),
        # intermediate dimension for the FFNs (0 indicates no FFN)
        d_intermediate: List[int] = list(),
        vocab_size: int = 256,
        ssm_cfg: SSMConfig = SSMConfig(),
        attn_cfg: AttnConfig = AttnConfig(),
        tie_embeddings: bool = False,
        pad_token_id: int = -100,  # pad id for loss fn
        ratio_loss_weight: float = 0.03,  # alpha in Hnet Paper
        use_return_dict: bool = False,
        log_bpreds: bool = True,
        selection: str = "cos",
        # auto_map={
        #    "AutoConfig": "hnet.hnet.models.confg_hnet.HNetConfig",
        #    "AutoModel": "hnet.hnet.models.mixer_seq.HNetForCausalLM",
        #    "AutoModelForCausalLM": "hnet.hnet.models.mixer_seq.HNetForCausalLM",
        #    "AutoModelForMaskedLM": "hnet.hnet.models.mixer_seq.HNetForCausalLM",
        # },
        *args,
        **kwargs,
    ):
        if "return_dict" in kwargs:
            del kwargs["return_dict"]
        super().__init__(
            *args, return_dict=use_return_dict, pad_token_id=pad_token_id, **kwargs
        )
        self.arch_layout = arch_layout
        self.d_model = (
            d_model  # intermediate dimension for the FFNs (0 indicates no FFN)
        )
        self.d_intermediate = d_intermediate
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.attn_cfg = attn_cfg
        self.tie_embeddings = tie_embeddings
        self.ratio_loss_weight = ratio_loss_weight
        self.log_bpreds = log_bpreds
        self.selection = selection
        # self.auto_map = auto_map

    def __repr__(self):
        return (
            f"HNetConfig({super().__repr__()}, " + f"arch_layout = {self.arch_layout})"
            f"d_model = {self.d_model})"
            f"d_intermediate = {self.d_intermediate})"
            f"vocab_size = {self.vocab_size})"
            f"ssm_cfg = {self.ssm_cfg})"
            f"attn_cfg = {self.attn_cfg})"
            f"tie_embeddings = {self.tie_embeddings})"
            f"pad_token_id = {self.pad_token_id})"
            f"ratio_loss_weight = {self.ratio_loss_weight})"
            f"log_bpreds = {self.log_bpreds})"
            f"auto_map = {self.auto_map})"
        )
