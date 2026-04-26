import torch
from caduceus.caduceus import CaduceusTokenizer, KMerTokenizer, ByteTokenizer
from hnet.hnet.models.mixer_seq import HNetForCausalLM
from hnet.hnet.models.config_hnet import (
    AttnConfig,
    SSMConfig,
    HNetConfig,
)
from omegaconf import OmegaConf
from wrappers.composer import ComposerWrapper


def build_model(**model_config):
    # model_config = cfg.get("model")
    # model_config = OmegaConf.to_container(cfg, resolve=True)
    # attn_cfg = AttnConfig(**model_config.get("attn_cfg"))
    # ssm_cfg = SSMConfig(**model_config.get("ssm_cfg"))
    ignore_keys = ["max_seq_len", "mlm", "default_target_ratio", "_modelstr_"]
    use_kmer = model_config.get("use_kmer", False)
    tok_type = model_config.get("tokenizer", None)
    tokenizer = None
    if tok_type is None or tok_type == "dna":
        tokenizer = CaduceusTokenizer(model_max_length=model_config["max_seq_len"])
    elif tok_type == "byte":
        tokenizer = ByteTokenizer(model_max_length=model_config["max_seq_len"])
    elif tok_type == "kmer":
        tokenizer = KMerTokenizer(
            k=model_config["k"], model_max_length=model_config["max_seq_len"]
        )
    else:
        raise Exception(f"Unkown tokenizer type {tok_type}")
    if model_config["vocab_size"] < len(tokenizer):
        model_config["vocab_size"] = len(tokenizer)

    print(model_config)
    hnet_cfg = HNetConfig(
        **{x: v for x, v in model_config.items() if x not in ignore_keys},
    )
    # Create model
    model = HNetForCausalLM(hnet_cfg, dtype=torch.bfloat16)
    # Use existing tokenizer instead of byte tokenizer (dna is already in bytes)
    # tokenizer = ByteTokenizer()
    return ComposerWrapper(model, tokenizer, mlm=model_config["mlm"])
