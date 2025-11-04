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


def build_model(cfg):
    # model_config = cfg.get("model")
    model_config = OmegaConf.to_container(cfg, resolve=True)
    # attn_cfg = AttnConfig(**model_config.get("attn_cfg"))
    # ssm_cfg = SSMConfig(**model_config.get("ssm_cfg"))
    hnet_cfg = HNetConfig(**model_config)
    # Create model
    model = HNetForCausalLM(hnet_cfg, dtype=torch.bfloat16)
    # Use existing tokenizer instead of byte tokenizer (dna is already in bytes)
    # tokenizer = ByteTokenizer()
    tokenizer = CaduceusTokenizer(model_max_length=cfg.max_seq_len)
    return ComposerWrapper(model, tokenizer, mlm=cfg.mlm)
