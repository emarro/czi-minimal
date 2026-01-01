# Base code imported from
# https://github.com/state-spaces/mamba
from functools import partial
from typing import Optional

from torch import nn, Tensor, cat

from flash_attn.ops.triton.layer_norm import RMSNorm
from mamba_ssm.modules.mamba2 import Mamba2

# TODO: this is just the bimamba wrapper, still need RCPS
from caduceus import BiMambaWrapper as Caduceus
from caduceus import CaduceusConfig

from .mha import CausalMHA
from .mlp import SwiGLU


class Mamba2Wrapper(Mamba2):
    """
    Mamba2 wrapper class that has the same inference interface as the CausalMHA class.
    """

    def __init__(self, *args, flops_counter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.flops_counter = flops_counter

    def forward(self, *args, num_tokens, **kwargs):
        if self.flops_counter is not None:
            num_flops = (
                6
                * int(num_tokens.sum().item())
                * self.d_model
                * self.expand
                * self.d_state
            )
            self.flops_counter.add_flops(num_flops)
        return super().forward(*args, **kwargs)

    def step(self, hidden_states, inference_params):
        # Don't use _get_states_from_cache because we want to assert that they exist
        conv_state, ssm_state = inference_params.key_value_memory_dict[
            self.layer_idx
        ]  # init class of Mamba2 accepts layer_idx
        result, conv_state, ssm_state = super().step(
            hidden_states, conv_state, ssm_state
        )

        # Update the state cache in-place
        inference_params.key_value_memory_dict[self.layer_idx][0].copy_(conv_state)
        inference_params.key_value_memory_dict[self.layer_idx][1].copy_(ssm_state)
        return result


class CaduceusWrapper(Caduceus):
    """
    Mamba2 wrapper class that has the same inference interface as the CausalMHA class.
    """

    def __init__(self, *args, flops_counter=None, **kwargs):
        cad_cfg = kwargs["config"]
        try:
            super().__init__(*args, **kwargs)
            self.d_model = cad_cfg.d_model
            self.expand = cad_cfg.layer_cfg.mamba_cfg.ssm_cfg["expand"]
            self.d_state = cad_cfg.layer_cfg.mamba_cfg.ssm_cfg["d_state"]
        except TypeError as e:
            # alternate processing depending on caducues version, TODO: standardize
            self.d_model = cad_cfg.d_model
            self.expand = cad_cfg.layer_cfg["mamba_cfg"]["ssm_cfg"]["expand"]
            self.d_state = cad_cfg.layer_cfg["mamba_cfg"]["ssm_cfg"]["d_state"]
            super().__init__(
                d_model=self.d_model,
                bidirectional=cad_cfg.bidirectional,
                bidirectional_strategy=cad_cfg.bidirectional_strategy,
                **cad_cfg.layer_cfg["mamba_cfg"]["ssm_cfg"],
            )
        self.flops_counter = flops_counter

    def forward(self, *args, num_tokens, **kwargs):
        if self.flops_counter is not None:
            num_flops = (
                6
                * int(num_tokens.sum().item())
                * self.d_model
                * self.expand
                * self.d_state
            ) * 2  # Approximate as just 2 passes of caducues
            self.flops_counter.add_flops(num_flops)
        return super().forward(*args, **kwargs)

    def step(self, hidden_states, inference_params):
        # Don't use _get_states_from_cache because we want to assert that they exist
        conv_state, ssm_state = inference_params.key_value_memory_dict[
            self.layer_idx
        ]  # init class of Mamba2 accepts layer_idx
        result, conv_state, ssm_state = super().step(
            hidden_states, conv_state, ssm_state
        )

        # Update the state cache in-place
        inference_params.key_value_memory_dict[self.layer_idx][0].copy_(conv_state)
        inference_params.key_value_memory_dict[self.layer_idx][1].copy_(ssm_state)
        return result


def create_block(
    arch,
    d_model,
    d_intermediate=None,
    ssm_cfg=dict(),
    attn_cfg=dict(),
    norm_epsilon=1e-5,
    layer_idx=None,
    residual_in_fp32=True,
    device=None,
    dtype=None,
    flops_counter=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}

    # Mixer
    if arch in ("t", "T"):
        mixer_cls = partial(
            CausalMHA,
            **attn_cfg,
            **factory_kwargs,
            layer_idx=layer_idx,
            flops_counter=flops_counter,
        )
    elif arch in ("m", "M"):
        mixer_cls = partial(
            Mamba2Wrapper,
            **ssm_cfg,
            **factory_kwargs,
            layer_idx=layer_idx,
            flops_counter=flops_counter,
        )
    elif arch in ("c", "C"):
        caduceus_config = {
            "bidirectional": True,
            "bidirectional_strategy": "add",
            "d_model": d_model,
            "layer_cfg": {"mamba_cfg": {"version": "v2", "ssm_cfg": ssm_cfg}},
        }
        caduceus_config = CaduceusConfig(**caduceus_config)
        mixer_cls = partial(
            CaduceusWrapper,
            **factory_kwargs,
            config=caduceus_config,
            layer_idx=layer_idx,
            flops_counter=flops_counter,
        )

    else:
        raise NotImplementedError

    # MLP
    if arch in ("T", "M", "C"):
        mlp_cls = partial(
            SwiGLU,
            d_intermediate=d_intermediate,
            **factory_kwargs,
        )
    elif arch in ("t", "m", "c"):
        mlp_cls = nn.Identity
    else:
        raise NotImplementedError

    # Normalization
    norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)
    if arch in ("c", "C"):
        block = RCPSBlock(
            d_model,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            residual_in_fp32=residual_in_fp32,
            flops_counter=flops_counter,
        )
    else:
        block = Block(
            d_model,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            residual_in_fp32=residual_in_fp32,
            flops_counter=flops_counter,
        )
    return block


class Block(nn.Module):
    def __init__(
        self,
        d_model,
        mixer_cls=None,
        mlp_cls=None,
        norm_cls=None,
        residual_in_fp32=True,
        flops_counter=None,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm1 = norm_cls(d_model)
        self.mixer = mixer_cls(d_model)
        self.flops_counter = flops_counter
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(d_model)
            self.mlp = mlp_cls(d_model)
        else:
            self.mlp = None

        assert RMSNorm is not None, "Triton is not installed"
        assert isinstance(self.norm1, RMSNorm), "Only RMSNorm is supported"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        mixer_kwargs=None,
        num_tokens=None,
    ):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )

        if mixer_kwargs is None:
            mixer_kwargs = {}
        hidden_states = self.mixer(
            hidden_states,
            inference_params=inference_params,
            num_tokens=num_tokens,
            **mixer_kwargs,
        )

        if self.mlp is not None:
            hidden_states, residual = self.norm2(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            hidden_states = self.mlp(hidden_states)
        ###########################################
        # ------------- Update FLOPs --------------#
        ###########################################
        if self.flops_counter is not None:
            # Add FLOPs for mlps, defer mixer FLOPs to inner mixer layers
            norm_1_flops = 0.0  # ignore
            norm_2_flops = norm_1_flops  # ignore
            if isinstance(self.mlp, SwiGLU):
                mlp_flops = (
                    2
                    * int(num_tokens.sum().item())
                    * (3 * self.mlp.d_model * self.mlp.d_intermediate)
                )  # in_dim = d_model, out_dim = ffw_dim
                gate_flops = 5 * int(num_tokens.sum().item()) * self.mlp.d_model

                self.flops_counter.add_flops(
                    mlp_flops + gate_flops + norm_1_flops + norm_2_flops
                )
            # self.flops_counter.add_flops(
            #    2
            #    * int(num_tokens.sum().item())
            #    * sum([x.numel() for x in self.parameters()])
            # )

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def step(self, hidden_states, inference_params, residual=None):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )
        hidden_states = self.mixer.step(hidden_states, inference_params)
        if self.mlp is not None:
            hidden_states, residual = self.norm2(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class RCPSBlock(Block):
    """
    RCPS version of a block (taken from Caduceus)
    """

    def __init__(
        self,
        d_model,
        mixer_cls=None,
        mlp_cls=None,
        norm_cls=None,
        residual_in_fp32=True,
        flops_counter=None,
    ):
        super().__init__(
            d_model, mixer_cls, mlp_cls, norm_cls, residual_in_fp32, flops_counter
        )
        self.norm1 = norm_cls(d_model // 2)
        self.mixer = mixer_cls(d_model)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(d_model // 2)
            self.mlp = mlp_cls(d_model)
        else:
            self.mlp = None

        assert RMSNorm is not None, "Triton is not installed"
        assert isinstance(self.norm1, RMSNorm), "Only RMSNorm is supported"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        mixer_kwargs=None,
        num_tokens=None,
    ):
        ##################################################
        # ----------- Split Norm 1 into fwd and rc -------#
        ##################################################
        hidden_states_fwd, residual_fwd = self.norm1(
            hidden_states[..., hidden_states.shape[-1] // 2 :],
            residual=residual[..., hidden_states.shape[-1] // 2 :]
            if residual is not None
            else None,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )

        hidden_states_rc, residual_rc = self.norm1(
            hidden_states[..., : hidden_states.shape[-1] // 2].flip(dims=[-2, -1]),
            residual=residual[..., : hidden_states.shape[-1] // 2].flip(dims=[-2, -1])
            if residual is not None
            else None,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )
        hidden_states = cat(
            [hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1
        )
        residual = cat([residual_fwd, residual_rc.flip(dims=[-2, -1])], dim=-1)

        ###################################################
        # ------------ Run the Mixer ---------------------#
        ###################################################

        if mixer_kwargs is None:
            mixer_kwargs = {}
        hidden_states = self.mixer(
            hidden_states,
            inference_params=inference_params,
            num_tokens=num_tokens,
            **mixer_kwargs,
        )

        ##################################################
        # ----------- Split Norm 2 into fwd and rc ------#
        ##################################################
        if self.mlp is not None:
            hidden_states_fwd, residual_fwd = self.norm2(
                hidden_states[..., hidden_states.shape[-1] // 2 :],
                residual=residual[..., hidden_states.shape[-1] // 2 :]
                if residual is not None
                else None,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )

            hidden_states_rc, residual_rc = self.norm2(
                hidden_states[..., : hidden_states.shape[-1] // 2].flip(dims=[-2, -1]),
                residual=residual[..., : hidden_states.shape[-1] // 2].flip(
                    dims=[-2, -1]
                )
                if residual is not None
                else None,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            hidden_states = cat(
                [hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1
            )
            residual = cat([residual_fwd, residual_rc.flip(dims=[-2, -1])], dim=-1)

        ###########################################
        # ------------- Update FLOPs --------------#
        ###########################################
        if self.flops_counter is not None:
            # Add FLOPs for mlps, defer mixer FLOPs to inner mixer layers
            norm_1_flops = 0.0  # ignore
            norm_2_flops = norm_1_flops  # ignore
            # TODO: Account for different types of MLPs? (Always SwiGLU for now, requires larger changes)
            if isinstance(self.mlp, SwiGLU):
                mlp_flops = (
                    2
                    * int(num_tokens.sum().item())
                    * (3 * self.mlp.d_model * self.mlp.d_intermediate)
                )  # in_dim = d_model, out_dim = ffw_dim
                gate_flops = 5 * int(num_tokens.sum().item()) * self.mlp.d_model

                self.flops_counter.add_flops(
                    mlp_flops + gate_flops + norm_1_flops + norm_2_flops
                )
            # self.flops_counter.add_flops(
            #    2
            #    * int(num_tokens.sum().item())
            #    * sum([x.numel() for x in self.parameters()])
            # )

        return hidden_states, residual

    def step(self, hidden_states, inference_params, residual=None):
        raise NotImplementedError("RCPS version of step not implemented")
