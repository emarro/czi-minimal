import torch
from composer.core import Callback, State
from composer.models.base import ComposerModel
from composer.loggers import Logger
from composer.utils import dist
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional


import os


class ChrChunker(Callback):
    def __init__(self, target_eval_label: str, save_dir: os.PathLike):
        super().__init__()
        self.target_eval_label = target_eval_label
        self.save_dir = save_dir
        self.buffer = []

    def eval_batch_end(self, state, logger):
        if state.dataloader_label != self.target_eval_label:
            return

        outputs = state.outputs
        chrom = state.batch_get_item("chrom")
        start = state.batch_get_item("start")
        # Grad and reshape outputs to be per batch (to line up with chrom and start)
        B, L, _ = state.outputs.logits.shape
        bpred_out = state.outputs.bpred_output  # list of bpred outputs
        # TODO: remove reshape, it's slow AF but using it to visualize indiv batches
        mask = bpred_out[0].boundary_mask.reshape(B, L)  # [B, L]
        # print(f"Mask shape : {mask.shape}")
        probs = bpred_out[0].boundary_prob[..., 1].reshape(B, L).float()  # [B, L]
        # print(f"Probs shape : {probs.shape}")
        losses = state.outputs.unreduced_loss.reshape(B, L)  # [B, L]
        for batch_idx in range(B):
            row = {
                "chrom": chrom[batch_idx].item(),
                "start": start[batch_idx].item(),
                "bmask": mask[batch_idx].cpu().detach().numpy(),
                "bprobs": probs[batch_idx].cpu().detach().numpy(),
                "loss": losses[batch_idx].cpu().detach().numpy(),
            }
            self.buffer.append(row)

    def eval_end(self, state, logger):
        if state.dataloader_label != self.target_eval_label:
            return
        rank = dist.get_global_rank()
        step = state.timestamp.batch
        df = pd.DataFrame(self.buffer)
        filename = f"step{step}_rank{rank}_outputs.parquet"
        filepath = os.path.join(self.save_dir, filename)
        table = pa.Table.from_pylist(self.buffer)
        pq.write_table(table, filepath)
        # df.to_csv(filepath)
