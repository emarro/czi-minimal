import torch


from collections import deque
from composer.core import Callback, State
from composer.loggers import Logger
from composer.models.base import ComposerModel
from composer.utils import dist

from typing import Any, cast


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
        self.history_wct.append(state.timestamp.total_wct.total_seconds())
        batch_flops = state.outputs.total_flops  # flops used in batch
        if self.in_train:
            batch_flops = 3 * batch_flops  # assume bkwd pass is 2x fwd pass
            self.total_train_flops += batch_flops
            logger.log_metrics(
                {"flop_counter/totaL-train_flops": self.total_train_flops}
            )
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

    def eval_start(self, state: State, logger: Logger):
        self.in_train = False

    def eval_all_end(self, state: State, logger: Logger):
        self.in_train = True


def calc_compression_ratio(num_tokens, mask):
    num_chunks = mask.to(torch.int).sum(dim=-1)  # [B]
    return num_chunks / num_tokens  # [n]


def calc_bpic(mask):
    """
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
    return result


class BPredMonitor(Callback):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

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
        compression_ratio = calc_compression_ratio(num_tokens=num_tokens, mask=mask)
        bpics = calc_bpic(mask)
        max_bpics = [x.max().item() for x in bpics]
        min_bpics = [x.min().item() for x in bpics]
        mean_bpics = [x.float().mean().item() for x in bpics]
        log = {
            "Boundaries/Compression ratio": compression_ratio.float().mean().item(),
            "Boundaries/Max BPIC in batch": max(max_bpics),
            "Boundaries/Min BPIC in batch": min(min_bpics),
            "Boundaries/Mean BPIC in batch": sum(mean_bpics) / len(mean_bpics),
        }
        for k, v in log.items():
            print(f"{k}: {v:3f}")
            # TODO: change naming in logger
        logger.log_metrics(log)
