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
