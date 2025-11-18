import torch


from collections import deque
from composer.core import Callback, State
from composer.loggers import Logger
from composer.models.base import ComposerModel
from composer.utils import dist

from io import BytesIO
from PIL import Image

from typing import Any, cast

import numpy as np
import matplotlib.pyplot as plt
import wandb


class FlopMonitor(Callback):
    def __init__(self, *args: Any, log_every: int = 10, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.total_train_flops = 0.0
        self.log_every = log_every
        # track flops in windows
        self.history_flops: deque[float] = deque(maxlen=log_every + 1)
        # track timesteps
        self.history_wct: deque[float] = deque(maxlen=log_every + 1)

        self.in_train: bool = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_train_flops": self.total_train_flops,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.total_train_flops = state["total_train_flops"]

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused

    def batch_end(self, state: State, logger: Logger) -> None:
        """
        Update FLOP count at end of every train batch
        """
        self.history_wct.append(state.timestamp.total_wct.total_seconds())
        batch_flops = state.outputs.total_flops  # flops used in batch

        # batch_flops is calculated on each GPU, sync across all GPUs
        flops_per_batch_tensor = state.device.tensor_to_device(
            torch.tensor(batch_flops, dtype=torch.float),
        )
        dist.all_reduce(flops_per_batch_tensor, reduce_operation="SUM")
        batch_flops = flops_per_batch_tensor.item()
        batch_flops = 3 * batch_flops  # assume bkwd pass is 2x fwd pass
        self.total_train_flops += batch_flops
        logger.log_metrics({"flop_counter/totaL-train_flops": self.total_train_flops})
        # print(f"Logging {self.total_train_flops / 1e12:,} TFlOPs")

        self.history_flops.append(batch_flops)
        if len(self.history_flops) == self.history_flops.maxlen:
            world_size = dist.get_world_size()
            elapsed_batches = len(self.history_flops) - 1
            elapsed_flops = sum(self.history_flops) - self.history_flops[0]
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            batches_per_sec = elapsed_batches / elapsed_wct
            flops_per_sec = elapsed_flops / elapsed_wct
            dev_batches_per_sec = batches_per_sec / world_size
            dev_flops_per_sec = flops_per_sec / world_size
            logger.log_metrics(
                {
                    "flop_counter/flops_per_sec": flops_per_sec,
                    "flop_counter/device/batches_per_sec": dev_flops_per_sec,
                }
            )


def calc_compression_ratio(num_tokens, mask):
    """
    Calculcate the compression (number of tokens aggregated into a chunk)
    returns:
        the percentage of the sequence that was compressed
        0 = no compression of the sequence
        ~1 = nearly all of the sequence was compressed into a single token
    """
    num_chunks = mask.to(torch.int).sum(dim=-1)  # [B]
    return 1.0 - (num_chunks / num_tokens)  # [B]


def calc_bpic(mask):
    """
    Reference implementation, assumes unpacked ([B*L]) input
    Calculates the size of each inner chunk in [mask]
    Assumes that each chunk ends at every True in mask (including the True token)
    e.g. [1,0,1,1] are chunks of size [1, 2, 1]
    Args:
      mask: torch.Tensor[bool]: [B, L] - the mask defining boundaries of inner chunks
    Returns:
      a list of tensors containing
    """
    # TODO: Rework to assume packed masked so we don't need to do the slow iteration
    # NOTE: Doing this requires keeping track of num chunk for batch since this can be ragged (not hard, just not done atm)
    if len(mask.shape) == 1:
        mask = mask.unsqueeze(0)
    B, L = mask.shape
    mask[:, L - 1] = True  # always force last tok to be a boundary if not already
    results = []
    for b_idx in range(B):
        mask_b = mask[b_idx]
        intermediate = (torch.cumsum((~mask_b).to(torch.int), dim=0) + 1) * mask_b
        intermediate = intermediate[intermediate > 0]
        result = torch.cat(
            (intermediate[0:1], (intermediate[1:] - intermediate[:-1]) + 1)
        )
        results.append(result)
    return torch.cat(results, dim=0)  # [num_chunks]


def calc_bpic_effecient(mask):
    """
    Calculates the size of each inner chunk in [mask]
    Assumes that each chunk ends at every True in mask (including the True token)
    e.g. [1,0,1,1] are chunks of size [1, 2, 1]
    Args:
      mask: torch.Tensor[bool]: [B, L] - the mask defining boundaries of inner chunks
    Returns:
      a tensor of size [num_chunks] (num_chunks = mask.sum()) containing the size of each chunk
    """
    # TODO: calc chunks stats per seq (doable given L)
    if len(mask.shape) == 1:
        mask = mask.unsqueeze(0)
    B, L = mask.shape
    mask[:, L - 1] = True  # always force last tok to be a boundary if not already
    mask = mask.reshape(-1)  # [B * L]
    intermediate = (
        torch.cumsum((~mask).to(torch.int), dim=0) + 1
    )  # Size of each chunk is at boundaries
    only_cum_size = intermediate * mask  # mask out everything except total chunk size
    size_mask = only_cum_size > 0
    # NOTE: without knowing next_num_tokens here, we no longer know chunk size per seq
    cum_sizes = torch.cat(
        [torch.tensor([1], device=mask.device), only_cum_size[size_mask]], dim=0
    )  # remove all non chunk sizes, left pad for next step
    chunk_sizes = cum_sizes[1:] - cum_sizes[:-1] + 1
    assert chunk_sizes.sum() == B * L, (
        f"Chunks do not add up to num of input tokens {B * L}, only counted {chunk_sizes.sum()} tokens in input"
    )
    return chunk_sizes  # [num_chunks]


class BPredMonitor(Callback):
    def __init__(self, *args: Any, log_every: int = 10, **kwargs: Any) -> None:
        super().__init__(*args, log_every=100, **kwargs)
        self.history_bpic: deque[torch.Tensor] = deque(maxlen=log_every + 1)
        self.history_eval_bpic: list[torch.Tensor] = list()
        self.history_eval_cr: list[torch.Tensor] = list()
        self.in_train: bool = True

    def batch_end(self, state: State, logger: Logger) -> None:
        # print(state.outputs)
        # print(state.outputs.logits.shape)
        B, L, _ = state.outputs.logits.shape
        bpred_out = state.outputs.bpred_output  # list of bpred outputs
        # print(bpred_out)
        # TODO: remove reshape, it's slow AF but using it to visualize indiv batches
        mask = bpred_out[0].boundary_mask.reshape(B, L)  # [B, L]
        # print(mask)
        # print(mask.shape)
        num_tokens = torch.tensor([L] * B, device=mask.device, dtype=torch.long)
        compression_ratio = calc_compression_ratio(
            num_tokens=num_tokens, mask=mask
        ).float()
        bpics = calc_bpic_effecient(mask).detach()  # .clone()
        max_bpics = bpics.max()  # [x.max().item() for x in bpics]
        min_bpics = bpics.min()  # [x.min().item() for x in bpics]
        mean_bpics = bpics.float().mean()  # [x.float().mean().item() for x in bpics]
        log = {
            "Boundaries/Train/Mean Seq Perc chunked in batch (1=Single Big Chunk)": compression_ratio.mean().item(),
            "Boundaries/Train/Max Seq Perc chunked in batch": compression_ratio.max().item(),
            "Boundaries/Train/Min Seq Perc chunked in batch": compression_ratio.min().item(),
            "Boundaries/Train/Max BPIC in batch": max_bpics.item(),
            "Boundaries/Train/Min BPIC in batch": min_bpics.item(),
            "Boundaries/Train/Mean BPIC in batch": mean_bpics.item(),
        }
        # for k, v in log.items():
        #    print(f"{k}: {v:3f}")
        # TODO: change naming in logger
        logger.log_metrics(log)
        self.history_bpic.append(bpics)
        # print(f"BPIC history of len {len(self.history_bpic)}: {self.history_bpic}")
        if len(self.history_bpic) == self.history_bpic.maxlen:
            full_bpics = torch.cat(list(self.history_bpic), dim=0)  # [num_total_chunks]
            log = {
                "Boundaries/Train/Mean Seq Perc chunked windowed (1=Single Big Chunk)": compression_ratio.mean().item(),
                "Boundaries/Train/Max Seq Perc chunked windowed": compression_ratio.max().item(),
                "Boundaries/Train/Min Seq Perc chunked windowed": compression_ratio.min().item(),
                "Boundaries/Train/Max BPIC windowed": full_bpics.max().item(),
                "Boundaries/Train/Min BPIC windowed": full_bpics.min().item(),
                "Boundaries/Train/Mean BPIC windowed": full_bpics.float().mean().item(),
            }
            # for k, v in log.items():
            #    print(f"{k}: {v:3f}")

    def eval_start(self, state: State, logger: Logger):
        self.in_train = False
        self.history_eval_bpic = list()
        self.history_eval_cr = list()

    def eval_batch_end(self, state: State, logger: Logger):
        B, L, _ = state.outputs.logits.shape
        bpred_out = state.outputs.bpred_output  # list of bpred outputs
        # TODO: remove reshape, it's slow AF but using it to visualize indiv batches
        mask = bpred_out[0].boundary_mask.reshape(B, L)  # [B, L]
        num_tokens = torch.tensor([L] * B, device=mask.device, dtype=torch.long)
        compression_ratio = (
            calc_compression_ratio(num_tokens=num_tokens, mask=mask)
            .float()
            .detach()
            .clone()
        )  # [B]
        bpics = calc_bpic_effecient(mask).detach().clone()
        self.history_eval_cr.append(compression_ratio)
        self.history_eval_bpic.append(bpics)

    def eval_end(self, state: State, logger: Logger):
        self.in_train = True
        compression_ratios = torch.cat(self.history_eval_cr, dim=0)  # [num_seqs]
        bpics = torch.cat(self.history_eval_bpic, dim=0)  # [num_chunks]
        log = {
            f"Boundaries/{state.dataloader_label} Eval/Mean Seq Perc chunked (1=Single Big Chunk)": compression_ratios.float()
            .mean()
            .item(),
            f"Boundaries/{state.dataloader_label} Eval/Max Seq Perc": compression_ratios.max().item(),
            f"Boundaries/{state.dataloader_label} Eval/Min Seq Perc": compression_ratios.min().item(),
            f"Boundaries/{state.dataloader_label} Eval/Max BPIC": bpics.max().item(),
            f"Boundaries/{state.dataloader_label} Eval/Min BPIC": bpics.min().item(),
            f"Boundaries/{state.dataloader_label} Eval/Mean BPIC": bpics.float()
            .mean()
            .item(),
        }
        logger.log_metrics(log)
        # print(
        #    f"Eval metrics on {bpics.size(0)} chunks, L=511 equals to {bpics.sum().item() // 511:,} sequences"
        # )
        # for k, v in log.items():
        #    print(f"{k}: {v:3f}")
        # TODO: Log a histogram of the compression ratios and off the bpics
        # TODO: Log images of the boundaries (maybe better left to a different callback or dedicated Evaluator & datasset?)
        fig, ax = plt.subplots(figsize=(6, 4))
        values = bpics.detach().cpu().numpy()
        ax.hist(values, color="blue", alpha=0.7)
        ax.set_title("Histogram of BpPIC sizes")
        ax.set_xlabel("Size")
        ax.set_ylabel("Count")
        plt.tight_layout()
        row = [wandb.Image(fig)]
        # fig.canvas.draw()  # render the figure
        ## Save figure to a BytesIO buffer in RGB format
        # buf = BytesIO()
        # fig.savefig(buf, format="png", bbox_inches="tight")
        # buf.seek(0)
        ## Read buffer with PIL and convert to NumPy array
        # img_array = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8).transpose(
        #    0, 2, 1
        # )
        # img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        # img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # HWC
        plt.close(fig)
        # Log figure to WandB
        logger.log_table(
            name=f"{state.dataloader_label}BpPIC_histogram",
            columns=["Hist"],
            rows=[row],
            step=int(state.timestamp.batch),
        )
        # Close figure to free memory
