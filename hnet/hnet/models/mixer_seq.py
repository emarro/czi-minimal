from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.utils.generation import GenerationMixin

from .hnet import HNet, HNetState
from .config_hnet import HNetConfig

from hnet.modules.dc import RoutingModuleOutput
from hnet.modules.utils import apply_optimization_params
from hnet.modules.utils import FlopsCounter


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    bpred_output: list[RoutingModuleOutput]
    inference_params: HNetState
    loss: torch.FloatTensor
    ar_loss: torch.FloatTensor
    ratio_loss: torch.FloatTensor
    total_flops: torch.FloatTensor


def cross_entropy(
    logits: torch.Tensor, labels: torch.LongTensor, pad_token_id: int = -100
) -> torch.FloatTensor:
    """Cross entropy loss.
    Inputs:
        logits: torch.FloatTensor [batch, seq_len, vocab_size
        y: torch.LongTensor [batch, seq_len]
        ignore_index: int
    """
    logits = logits.view(-1, logits.shape[-1])  # [batch*seq_len, vocab_size]
    y = labels.view(-1)  # [batch*seq_len]
    return F.cross_entropy(
        logits, y, ignore_index=pad_token_id, reduction="none"
    )  # [b * l]


def weighted_cross_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    loss_weights: torch.FloatTensor,
    pad_token_id: int = -100,
) -> torch.FloatTensor:
    """Weighted cross entropy loss (discounts certain tokens, e.g., repeated base pairs in genome).
    Inputs:
        logits: torch.FloatTensor [batch, seq_len, vocab_size
        y: torch.LongTensor [batch, seq_len]
        loss_weights: torch.FloatTensor [batch, seq_len]
        ignore_index: int
    """
    logits = logits.view(-1, logits.shape[-1])  # [batch * seq_len, vocab_size]
    y = labels.view(-1)  # [batch*seq_len]
    ce = F.cross_entropy(
        logits, y, ignore_index=pad_token_id, reduction="none"
    )  # [batch * seq_len]
    loss_weights = loss_weights.view(-1)  # [batch*seq_len]
    loss_weights[y == pad_token_id] = 0.0
    # TODO: Follows GPN implementation, but should we remove weight normalization?
    return ce * (loss_weights / loss_weights.sum())  # .sum()  # [1]


class HNetForCausalLM(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: HNetConfig,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config

        vocab_size = self.config.vocab_size
        d_embed = self.config.d_model[0]
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        # Flop counter to estimate the FLOPs per forward pass
        self.flops_counter = FlopsCounter(device)

        # We consider the HNet as a map (B, L, D[0]) -> (B, L, D[0])
        # Thus, the embedding is defined outside of the HNet.
        self.embeddings = nn.Embedding(vocab_size, d_embed, **factory_kwargs)

        self.backbone = HNet(
            config=config,
            # We pass in the stage_idx as an HNet needs to know what
            # depth of the hierarchy it is in.
            stage_idx=0,
            # Pass flops_counter so all inner stages can update
            flops_counter=self.flops_counter,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False, **factory_kwargs)
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def init_weights(self, initializer_range: float = 0.02) -> None:
        """
        Initializes the weights of the model.
        """
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=initializer_range)
        # embeddings are initialized differently from linears
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0)
        self.backbone._init_weights(initializer_range)

    def apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        """
        Applies the learning rate multipliers to the parameters of the model.
        NOTE: Must be ran before creating parameter groups, see hnet.utils.train.group_params for an example on how to run parameter groups.

        Inputs:
            lr_multiplier: A list of learning rate multipliers, one for each stage of the hierarchy, with the outer stages first (e.g. [3.0, 1.7, 0.9]).
        """
        for param in self.embeddings.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        for param in self.lm_head.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        self.backbone._apply_lr_multiplier(lr_multiplier)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        target_ratio: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inference_params: Optional[dict] = None,
        num_last_tokens: int = 0,
        **mixer_kwargs,
    ) -> CausalLMOutput:
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.embeddings(input_ids)

        # Add FLOPs for embedding

        if self.flops_counter is not None:
            self.flops_counter.reset()  # Reset from prev fwd pass
            # The embedding layer may have a projection in it too
            embedding_params = sum(p.numel() for p in self.embeddings.parameters())
            self.flops_counter.add_flops(2 * int(input_ids.numel()) * embedding_params)

        B, L, D = hidden_states.shape

        assert position_ids is None, (
            "Position ids are not supported for HNet due to the subsampling hierarchical structure"
        )
        # TODO: Ask June appouting packing (we can assume all seqs same length and therefore packing during training), do we need to do anything else?
        if mask is None:
            # Absent a mask, we assume we are running in packed mode
            assert inference_params is None, (
                "Inference params are not supported in packed mode"
            )
            hidden_states = hidden_states.flatten(0, 1)
            cu_seqlens = torch.arange(B + 1, device=hidden_states.device) * L
            max_seqlen = torch.tensor(L, dtype=torch.int, device=hidden_states.device)
        else:
            cu_seqlens = None
            max_seqlen = None

        num_tokens = torch.tensor(
            [L] * B, device=hidden_states.device
        )  # number of tokens for each seq in batch
        hidden_states, bpred_output = self.backbone(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params,
            num_tokens=num_tokens,
            **mixer_kwargs,
        )

        hidden_states = hidden_states.view(B, L, D)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)

        # Add the FLOPs for the LM head
        if not isinstance(self.lm_head, nn.Identity):
            self.flops_counter.add_flops(
                2
                * int(input_ids.numel())
                * self.lm_head.in_features
                * self.lm_head.out_features
            )
        total_flops = self.flops_counter.get_flops()

        loss = None
        ar_loss = None
        ratio_loss_sum = None
        unreduced_ar_loss = None
        if labels is not None:
            # Standard AR loss (or weighted version of ar loss)
            if loss_weights is not None:
                ar_loss = weighted_cross_entropy(
                    logits=lm_logits,
                    labels=labels,
                    loss_weights=loss_weights,
                    pad_token_id=self.config.pad_token_id,
                )
                unreduced_ar_loss = ar_loss  # [B * L]
                ar_loss = ar_loss.sum()
            else:
                ar_loss = cross_entropy(
                    logits=lm_logits,
                    labels=labels,
                    pad_token_id=self.config.pad_token_id,
                )
                unreduced_ar_loss = ar_loss  # [B * L]
                ar_loss = ar_loss.sum()
            loss = ar_loss
            # TODO: target_ratio should be a list (allow diff ratio per stage), currently fixed
            if target_ratio is not None:
                ratio_loss_sum = 0.0
                for bpred_stage in bpred_output:
                    # Calculate the ratio_loss for each stage
                    boundary_mask = bpred_stage.boundary_mask  #  [B * seq_len]
                    boundary_probs = bpred_stage.boundary_prob  # [B * seq_len, 2]
                    boundary_probs = boundary_probs[:, 1]  # [B *seq_len]

                    # NOTE: According to June we should flatten instead of taking the batchmean. No real effect on DNA (where L is constant)
                    # Leaving old logic for possible future exprimentation with varying target_ratio per batch
                    # boundary_mask = boundary_mask.reshape(B, L)  # [B, seq_len]
                    # boundary_probs = boundary_probs.reshape(B, L)  # [B, seq_len]
                    # f_loss = torch.sum(boundary_mask, dim=-1) * (1 / L)  # [1]
                    # g_loss = torch.sum(boundary_probs, dim=-1) * (1 / L)  # [1]
                    # cast boundary mask to same dypte as boundary_probs (mean doesn't work with bool types)
                    f_loss = torch.mean(
                        boundary_mask.to(boundary_probs.dtype), dim=-1
                    )  # [1]
                    g_loss = torch.mean(boundary_probs, dim=-1)  # [1]

                    stage_ratio_loss = (target_ratio / (target_ratio - 1)) * (
                        (target_ratio - 1) * f_loss * g_loss
                        + (1 - f_loss) * (1 - g_loss)
                    )  # [1]
                    ratio_loss_sum += stage_ratio_loss.mean()  # [1]
                # L = L_ar + \alpha * \sum_{stages} {L_ratio}
                loss = ar_loss + (self.config.ratio_loss_weight * ratio_loss_sum)

        CausalLMOutput = namedtuple(
            "CausalLMOutput",
            [
                "loss",
                "unreduced_loss",
                "logits",
                "bpred_output",
                "inference_params",
                "ar_loss",
                "ratio_loss",
                "total_flops",
            ],
        )
        return CausalLMOutput(
            loss=loss,
            unreduced_loss=unreduced_ar_loss,
            logits=lm_logits,
            bpred_output=bpred_output,
            inference_params=inference_params,
            ar_loss=ar_loss,
            ratio_loss=ratio_loss_sum,
            total_flops=total_flops,
        )

    def step(self, input_ids, inference_params):
        B = input_ids.shape[0]
        assert B == 1, (
            "HNetForCausalLM step currently only supports batch size 1 -- need to handle different-size lengths for each sample"
        )

        hidden_states = self.embeddings(input_ids)

        hidden_states, bpred_output = self.backbone.step(
            hidden_states, inference_params
        )
        logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=logits, bpred_output=bpred_output, inference_params=inference_params
        )
