import torch

from caduceus import CaduceusTokenizer
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import (
    AttnConfig,
    SSMConfig,
    HNetConfig,
)
from hnet.utils.tokenizers import ByteTokenizer
from omegaconf import OmegaConf
from wrappers.composer import ComposerWrapper


def build_model(**model_config):
    # model_config = cfg.get("model")
    # model_config = OmegaConf.to_container(cfg, resolve=True)
    # attn_cfg = AttnConfig(**model_config.get("attn_cfg"))
    # ssm_cfg = SSMConfig(**model_config.get("ssm_cfg"))
    ignore_keys = ["max_seq_len", "mlm"]
    hnet_cfg = HNetConfig(
        **{x: v for x, v in model_config.items() if x not in ignore_keys}
    )
    # Create model
    model = HNetForCausalLM(hnet_cfg, dtype=torch.bfloat16)
    # Use existing tokenizer instead of byte tokenizer (dna is already in bytes)
    # tokenizer = ByteTokenizer()
    tokenizer = CaduceusTokenizer(model_max_length=model_config["max_seq_len"])
    return ComposerWrapper(model, tokenizer, mlm=model_config["mlm"])
