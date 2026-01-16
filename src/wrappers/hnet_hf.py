import torch
from caduceus.caduceus import CaduceusTokenizer
from hnet.hnet.models.mixer_seq import HNetForCausalLM
from hnet.hnet.models.config_hnet import (
    AttnConfig,
    SSMConfig,
    HNetConfig,
)
from hnet.hnet.utils.tokenizers import ByteTokenizer
from omegaconf import OmegaConf
from wrappers.composer import ComposerWrapper
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForCausalLM,
)

# class AutoModelFromPretrained(nn.module):
#    def __init__(self, automodel_cls, pretrained_model_name_or_path, trust_remote_code, num_lay


def build_model(**model_config):
    # model_config = cfg.get("model")
    # model_config = OmegaConf.to_container(cfg, resolve=True)
    # attn_cfg = AttnConfig(**model_config.get("attn_cfg"))
    # ssm_cfg = SSMConfig(**model_config.get("ssm_cfg"))
    ignore_keys = ["max_seq_len", "mlm", "default_target_ratio", "_modelstr_"]
    hnet_cfg = HNetConfig(
        **{x: v for x, v in model_config.items() if x not in ignore_keys}
    )
    hnet_cfg.auto_map = {
        "AutoConfig": "hnet.hnet.models.confg_hnet.HNetConfig",
        "AutoModel": "hnet.hnet.models.mixer_seq.HNetForCausalLM",
        "AutoModelForCausalLM": "hnet.hnet.models.mixer_seq.HNetForCausalLM",
        "AutoModelForMaskedLM": "hnet.hnet.models.mixer_seq.HNetForCausalLM",
    }
    # Create model
    model = HNetForCausalLM(hnet_cfg, dtype=torch.bfloat16)
    # Use existing tokenizer instead of byte tokenizer (dna is already in bytes)
    # tokenizer = ByteTokenizer()
    tokenizer = CaduceusTokenizer(model_max_length=model_config["max_seq_len"])
    return ComposerWrapper(model, tokenizer, mlm=model_config["mlm"])
