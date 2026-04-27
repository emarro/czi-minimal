from dataclasses import dataclass

import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from hnet.hnet.modules.utils import get_seq_idx

from .torch_struct import SemiMarkov as SemiMarkovCRF
from .torch_struct import LinearChain as LinearChainCRF


@triton.jit
def _batched_dp_kernel(
    adj_ptr,
    F_ptr,
    bp_ptr,
    N: tl.constexpr,
    stride_adj_b,
    stride_adj_i,
    stride_adj_j,
    stride_f_b,
    stride_f_n,
    stride_bp_b,
    stride_bp_n,
    BLOCK_SIZE: tl.constexpr,
):
    # TODO: Make adj triu
    # TODO: make adj sparse (govered by \alpha in main loop) (needs some reworking of core algo)
    # TODO: experiment with additionally blocking adj with some horizon k
    batch_idx = tl.program_id(0)

    batch_adj_ptr = adj_ptr + batch_idx * stride_adj_b
    batch_f_ptr = F_ptr + batch_idx * stride_f_b
    batch_bp_ptr = bp_ptr + batch_idx * stride_bp_b

    offsets = tl.arange(0, BLOCK_SIZE)

    # Hoist: load f_vals once
    f_vals = tl.load(batch_f_ptr + offsets, mask=offsets < N, other=float("inf")).to(
        tl.float32
    )

    # Hoist: start adj pointer at j=1 column
    adj_j_ptr = batch_adj_ptr + stride_adj_j

    for j in tl.range(1, N + 1):
        mask = offsets < j

        costs = tl.load(
            adj_j_ptr + offsets * stride_adj_i, mask=mask, other=float("inf")
        ).to(tl.float32)
        candidates = f_vals + costs

        best_val = tl.min(candidates, axis=0)
        best_i = tl.argmin(candidates, axis=0)

        # Update f_vals in-register rather than store/reload
        f_vals = tl.where(offsets == j, best_val, f_vals)

        tl.store(batch_bp_ptr + j, best_i.to(tl.int64))

        adj_j_ptr += stride_adj_j  # pointer advance instead of multiply

    # Single bulk store at end
    tl.store(batch_f_ptr + offsets, f_vals, mask=offsets < N + 1)


def optimal_selection_triton(S, alpha, beta_base):
    if S.dim() == 2:
        S = S.unsqueeze(0)

    B, N, _ = S.shape
    device = S.device
    dtype = torch.float32  # Changed to float32 for consistency

    S = S.to(
        torch.bfloat16
    )  # Explicitly cast S to bfloat16 to match other implementations

    # 1. Pre-calculation (Vectorized)
    # V is calculated in original S dtype, then cast for P
    V = torch.relu(alpha - S)

    # Integral Image in fp32 for precision
    P = torch.zeros(
        (B, N + 1, N + 1), device=device, dtype=torch.float32
    )  # Changed to float32
    P[:, 1:, 1:] = torch.cumsum(
        torch.cumsum(V.float(), dim=1), dim=2
    )  # V.float() to ensure correct type if S is bfloat16

    P_diag = torch.diagonal(P, dim1=1, dim2=2)
    cost_matrix = 0.5 * (P_diag.unsqueeze(1) - 2 * P + P_diag.unsqueeze(2)).to(
        dtype
    )  # Ensure cost_matrix is float32

    indices = torch.arange(N + 1, device=device).to(dtype)
    lengths = indices.view(1, -1) - indices.view(-1, 1)
    # Clamp lengths to avoid log(0)
    adaptive_beta = (beta_base * torch.log(torch.relu(lengths) + 1.0)).to(
        dtype
    )  # Ensure adaptive_beta is float32

    adj_matrix = (cost_matrix + adaptive_beta.unsqueeze(0)).to(dtype).contiguous()

    # 2. DP Buffers
    F = torch.full(
        (B, N + 1), float("inf"), device=device, dtype=dtype
    )  # Changed to float32
    F[:, 0] = 0.0
    backpointers = torch.zeros((B, N + 1), dtype=torch.long, device=device)

    # 3. Execution
    BLOCK_SIZE = triton.next_power_of_2(N + 1)

    _batched_dp_kernel[(B,)](
        adj_matrix,
        F,
        backpointers,
        N,
        adj_matrix.stride(0),
        adj_matrix.stride(1),
        adj_matrix.stride(2),
        F.stride(0),
        F.stride(1),
        backpointers.stride(0),
        backpointers.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return backpointers


def backpointers_to_bounds(backpointers):
    # 4. CPU-based reconstruction
    print(f"Backpointers: {backpointers} ({backpointers.shape})")
    if len(backpointers.shape) == 1:
        backpointers = backpointers.unsqueeze(0)
    B, N = backpointers.shape
    all_communities = []
    bp_cpu = backpointers.cpu().numpy()
    # print(f"Backpointers: {bp_cpu}") # Added print for debugging
    for b in range(B):
        communities = []
        curr = N - 1
        while curr > 0:
            prev = bp_cpu[b, curr]
            communities.append((int(prev), curr - 1))
            curr = prev
        all_communities.append(communities[::-1])
    all_communities = np.array(all_communities)

    return all_communities if B > 1 else all_communities[0]


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


@dataclass
class RoutingModuleState:
    """
    The state of the routing module.

    Contains
        - [has_seen_tokens] (batch_size,) bool tensor. Whether that batch element has processed any tokens yet.
        - [last_hidden_state] (batch_size, d_model) tensor. The last hidden state of the batch element (used for boundary prediction).
    """

    has_seen_tokens: torch.Tensor  # (batch_size,)
    last_hidden_state: torch.Tensor  # (batch_size, d_model)


@dataclass
class DeChunkState:
    """
    The state of the dechunk.

    Contains
        - [last_value] (batch_size, d_model) tensor. The last value of the batch element (used for the EMA).
    """

    last_value: torch.Tensor  # (batch_size, d_model)


class RoutingModule(nn.Module):
    def __init__(
        self,
        d_model,
        selection="cos",
        alpha=None,
        beta=None,
        device=None,
        dtype=None,
        vocab_size=256,
    ):
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.selection = selection
        if selection == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(self.d_model, 1, bias=False),
                # nn.GELU(),
                # nn.Linear(self.d_model * 2, 1),
                nn.Sigmoid(),
            )
            # self.mlp = nn.Sequential(
            #    nn.Linear(self.d_model, d_model * 2, bias=False),
            #    nn.GELU(),
            #    nn.Linear(self.d_model * 2, 1, bias=False),
            #    nn.Sigmoid(),
            # )

        elif selection == "cos":
            self.q_proj_layer = nn.Linear(
                d_model, d_model, bias=False, **factory_kwargs
            )
            self.k_proj_layer = nn.Linear(
                d_model, d_model, bias=False, **factory_kwargs
            )
            with torch.no_grad():
                self.q_proj_layer.weight.copy_(torch.eye(d_model))
                self.k_proj_layer.weight.copy_(torch.eye(d_model))
            self.q_proj_layer.weight._no_reinit = True
            self.k_proj_layer.weight._no_reinit = True
        elif selection == "os":
            assert alpha is not None and beta is not None, (
                f"Got {alpha} for alpha and {beta} for beta, invalid"
            )
            self.alpha = alpha
            self.beta = beta
        elif "shannon" in selection:  # either shannon-N or shannon-B
            # NOTE: not well defined for the MLM case yet, assumes AR
            # shannon-N, adapative N based on the NLL of the given seq (NLL \propto bits we need to losslessly compress input)
            # shannon-B, pick (expected) number of bits we want per chunk and chunk seq s.t. our bounds line up with this
            self.bits_per_chunk = 2  # shannon-B param, put bounds where we expect to need [bits_per_chunk] bits for the prev toks
            vocab_size = vocab_size  # lookup vocab size in cfg later
            # LM head for the encoder, lets us get NLL and compression rates for the given seq (under this encoder)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
            # if we're only using entropy for N calcs, still need the MLP for bound preds
            if selection == "shannon-N":
                self.mlp = nn.Sequential(
                    nn.Linear(self.d_model, 1, bias=False),
                    # nn.GELU(),
                    # nn.Linear(self.d_model * 2, 1),
                    nn.Sigmoid(),
                )
            elif selection == "shannon-semi-markov-CRF":
                self.L = 8  # maximum sequence length
                self.C = 2  # number of differnt segments (2 for binary)
                # self.convs = nn.ModuleList(
                #    [
                #        nn.Conv1d(
                #            self.d_model, self.d_model, kernel_size=ell, padding=ell - 1
                #        )
                #        for ell in range(1, self.L + 1)
                #    ]
                # )  # L convolutions, all independent, all parallelisable
                # approximate with one big kernel of max length? Leakage exists but a problem in practice?
                # self.convs = nn.Conv1d(
                #    in_channels=self.d_model,
                #    out_channels=self.L * self.d_model,
                #    kernel_size=self.L,
                #    padding=self.L - 1,
                # )

                self.W_emit = nn.Linear(
                    self.d_model, self.C, bias=False
                )  # scores for each token
                self.phi_dur = nn.Parameter(
                    torch.zeros(1, 1, self.L, 1, self.C)
                )  # scores for different seq lengths
                self.W_trans = nn.Parameter(
                    torch.zeros(1, 1, 1, self.C, self.C)
                )  # fixed tranisition probs

        else:
            raise Exception(f"Unrecognized selection mechanism {selection}")

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(self, hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        assert (mask is not None) or (cu_seqlens is not None), (
            "Either mask or cu_seqlens must be provided"
        )

        if inference_params is not None:
            assert mask is not None, (
                "Mask must be provided if inference_params is provided"
            )
            assert (~inference_params.has_seen_tokens).all(), (
                "Cannot have seen tokens when inference_params is not provided"
            )

        if cu_seqlens is not None:
            # We are in packed mode, so hidden_states is (T, D). Make it (B, T, D)
            hidden_states = hidden_states.unsqueeze(0)  # [1, B*T, D]
            # B = hidden_states.shape[1] // cu_seqlens[1]
            # hidden_states = hidden_states.reshape(B, hidden_states.shape[1] // B, -1)
        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        B, L, D = hidden_states.shape
        boundary_prob = torch.zeros((B, L), device=hidden_states.device)

        if self.selection == "cos":
            cos_sim = torch.einsum(
                "b l d, b l d -> b l",
                F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
                F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
            )
            boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
            boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)
        elif self.selection == "mlp" or self.selection == "shannon-N":
            boundary_prob = self.mlp(hidden_states)[:, 1:, 0]  # [B,L, 1]
            boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)
        elif self.selection == "os":
            cos_sim = torch.einsum(
                "b i d, b j d -> b i j",
                F.normalize(hidden_states, dim=-1),
                F.normalize(hidden_states, dim=-1),
            )
            backpointers = optimal_selection_triton(cos_sim, self.alpha, self.beta)
            boundary_prob.scatter_(1, backpointers, 1.0)
        elif self.selection == "lc-CRF":
            pass
        elif self.selection == "shannon-semi-markov-CRF":
            S = torch.cumsum(hidden_states, dim=1)  # [B, L, D]
            spans = []
            for ell in range(1, self.L + 1):
                # sum over [v-ell+1, v] = S[v] - S[v-ell]
                right = S  # [B, T, D]
                left = F.pad(S[:, :-1, :], (0, 0, ell - 1, 0))  # shift right by ell
                mean = (right - left) / ell  # [B, T, D]
                spans.append(mean)
            spans = torch.stack(spans, dim=2)  # [B, T, L, D]
            token_emit = self.W_emit(spans)  # [B, T, L, 2]
            phi_emit = token_emit.unsqueeze(-2)  # [B T, L, 1, 2]
            phi = phi_emit + self.phi_dur + self.W_trans
            log_pots = torch.log(phi)  # [B, T, L, 2, 2]
            dist = SemiMarkovCRF(log_pots)
            marginals = dist.marginals  # [B, T, L, 2, 2]
            marginals = marginals.sum(2).sum(-1)  # [B, T, 2]
            boundary_prob = marginals[..., 0]

        elif self.selection == "shannon-B":  # use entropy to call bounds
            logits = self.lm_head(hidden_states)
            log_probs = F.log_softmax(logits, dim=-1)
            print(
                f"Log prob in shannon-B exp: {torch.exp(log_probs)},  ({log_probs.shape})"
            )
            raise NotImplementedError

        if cu_seqlens is not None:
            boundary_prob = boundary_prob.squeeze(0)
            boundary_prob[cu_seqlens[:-1]] = PAD_PROB

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)

        boundary_mask = selected_idx == 1  # (shape hidden_states.shape[:-1])
        if mask is not None:
            # No invalid tokens can be selected
            boundary_mask = boundary_mask & mask

        if inference_params is not None:
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask,
                    hidden_states[
                        torch.arange(
                            hidden_states.shape[0], device=hidden_states.device
                        ),
                        last_mask,
                    ],
                    inference_params.last_hidden_state,
                )
            )

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # (shape hidden_states.shape[:-1], 1)

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (shape hidden_states.shape[:-1], 2)
            boundary_mask=boundary_mask,  # (shape hidden_states.shape[:-1])
            selected_probs=selected_probs,  # (shape hidden_states.shape[:-1], 1)
        )

    def step(self, hidden_states, inference_params):
        # hidden_states is (B, 1, D)
        hidden_states = hidden_states.squeeze(1)
        cos_sim = torch.einsum(
            "b d, b d -> b",
            F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        inference_params.last_hidden_state.copy_(hidden_states)
        boundary_prob = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            torch.ones_like(boundary_prob),
        )
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        inference_params.has_seen_tokens.copy_(
            torch.ones_like(inference_params.has_seen_tokens)
        )
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (B, 2)
            boundary_mask=boundary_prob[..., 1] > 0.5,  # (B,)
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),  # (B, 1)
        )


class ChunkLayer(nn.Module):
    def forward(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        assert (mask is not None) or (cu_seqlens is not None), (
            "Either mask or cu_seqlens must be provided"
        )

        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(
                boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
            )
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(dim=-1)
            next_max_seqlen = int(num_tokens.max())

            device = hidden_states.device
            L = hidden_states.shape[1]
            token_idx = (
                torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            next_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                    -1, -1, hidden_states.shape[-1]
                ),
            )

            next_mask = (
                torch.arange(next_max_seqlen, device=device)[None, :]
                < num_tokens[:, None]
            )
            next_max_seqlen = None

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask

    def step(self, hidden_states, boundary_mask):
        return hidden_states[boundary_mask]


class DeChunkLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dtype=torch.bfloat16,
        block_size=256,
        headdim=32,
    ):
        super().__init__()
        self.d_model = d_model

        # Just for Mamba2 kernel.
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        assert d_model % self.headdim == 0
        self.nheads = d_model // self.headdim

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkState(
            last_value=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        cu_seqlens=None,
        inference_params=None,
        mask=None,
    ):
        if inference_params is not None:
            assert mask is not None, (
                "Mask must be provided if inference_params is provided"
            )
            assert boundary_mask[:, 0].all(), (
                "First token must be a boundary if running prefill"
            )

        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))

        if cu_seqlens is not None:
            p = p[boundary_mask].unsqueeze(0)
            seq_idx = get_seq_idx(cu_seqlens, device=hidden_states.device)
        else:
            B, L = boundary_mask.shape
            seq_idx = None

            token_idx = (
                torch.arange(L, device=hidden_states.device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            p = torch.gather(
                p, dim=1, index=seq_sorted_indices[:, : hidden_states.shape[1]]
            )  # (B, M)

        original_dtype = hidden_states.dtype
        # Reuse Mamba2 kernel for EMA Deaggregator.
        dt = torch.log(1 / (1 - p)).to(self.dtype)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = -torch.ones(
            (self.nheads,), device=hidden_states.device, dtype=torch.float32
        )
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size,
            seq_idx=seq_idx,
        )
        out = rearrange(out, "b l h p -> b l (h p)")

        if cu_seqlens is not None:
            out = out.squeeze(0)
            plug_back_idx = boundary_mask.cumsum(dim=0) - 1
            out = torch.gather(
                out, dim=0, index=plug_back_idx.unsqueeze(-1).expand(-1, self.d_model)
            )
        else:
            plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
            out = torch.gather(
                out,
                dim=1,
                index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
            )

        if inference_params is not None:
            inference_params.last_value.copy_(out[:, -1])

        return out.to(original_dtype)

    def step(self, hidden_states, boundary_mask, boundary_prob, inference_params):
        # hidden_states is (B', 1, D), where B' = boundary_mask.sum()
        # boundary_mask is (B,) and boundary_prob is (B, 2)

        B = boundary_mask.shape[0]
        # B_selected = hidden_states.shape[0]
        D = hidden_states.shape[-1]

        p = torch.zeros(B, device=hidden_states.device, dtype=hidden_states.dtype)
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(
            min=1e-4, max=1 - (1e-4)
        )

        current_hidden_states = torch.zeros(
            B, D, device=hidden_states.device, dtype=hidden_states.dtype
        )
        current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

        result = p * current_hidden_states + (1 - p) * inference_params.last_value
        inference_params.last_value.copy_(result)

        return result.unsqueeze(1)
