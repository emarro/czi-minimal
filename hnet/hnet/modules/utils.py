from dataclasses import asdict

import torch
from omegaconf import OmegaConf


def get_seq_idx(cu_seqlens, device=None):
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()

    return seq_idx


def get_stage_cfg(cfg, stage_idx):
    def dictify(cfg):
        if type(cfg) is dict:
            return cfg
        elif OmegaConf.is_dict(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
        return asdict(cfg)

    return {
        k: v[stage_idx] if isinstance(v, list) else v for k, v in dictify(cfg).items()
    }


def apply_optimization_params(
    param: torch.Tensor,
    **kwargs,
) -> None:
    """
    Annotates a parameter with optimization parameters.

    Specifically, updates the parameter's `_optim` attribute with the given kwargs.
    """

    if hasattr(param, "_optim"):
        param._optim.update(kwargs)
    else:
        param._optim = kwargs


class FlopsCounter:
    def __init__(self, device):
        self.flops_used = torch.tensor(0.0, device=device)
        self.reset()

    def add_flops(self, flops: torch.FloatTensor):
        self.flops_used += flops

    def get_flops(self):
        return self.flops_used

    def reset(self):
        self.flops_used = self.flops_used * 0.0
