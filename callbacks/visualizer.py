import torch
from PIL import Image, ImageDraw, ImageFont
from composer.core import Callback, State
from composer.loggers import Logger
import matplotlib.pyplot as plt
import wandb
import numpy as np
from typing import Optional


import os


class IGVCallBack(Callback):
    def __init__(
        self,
        *args,
        target_eval_label: Optional[str] = None,
        log_only_N: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            target_eval_label: str - only log from this evaluator (if none log all)
            log_only_N: int - only log the first N seqs every batch (if not None)
        """
        super().__init__(*args, **kwargs)
        self.target_eval_label = target_eval_label
        self.log_only_N = log_only_N
        self.logged_count = 0

        self.columns = ["chrom", "start", "stop", "igv_boundary_img"]
        # self.table = wandb.Table(columns=columns)
        self.rows = []  # List[List[any]], rows of data for table
        # assumed tracks to plot
        self.binary_tracks = ["softmask", "annot", "bmask"]
        self.continuous_tracks = ["bprobs", "loss"]

        # Color mapping for softmasked sequences (upper case = unmasked)
        self.unmasked_colors = {
            "A": "#64AD66",  # green
            "C": "#4C70BA",  # blue
            "G": "#F2BE4A",  # yellow/orange
            "T": "#D9534F",  # red
            "N": "#999999",  # gray
        }

        self.softmasked_colors = {
            "a": "#A8D5AB",  # light green
            "c": "#8AA8D2",  # light blue
            "g": "#F6D98B",  # light yellow
            "t": "#E7A19F",  # light red
            "n": "#C8C8C8",  # light gray
        }

    def _plot_sequence_track(self, ax, seq, start, end, box_height=1.0):
        xs = np.arange(start, end)

        # Draw colored rectangles
        for x, nt in zip(xs, seq):
            if nt in self.softmasked_colors:
                color = self.softmasked_colors[nt]
            else:
                color = self.unmasked_colors.get(nt.upper(), "#777")

            ax.add_patch(plt.Rectangle((x, 0), 1, box_height, color=color, alpha=0.95))

        # Text overlay (always uppercase)
        for x, nt in zip(xs, seq):
            ax.text(
                x + 0.5,
                box_height / 2,
                nt,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

        ax.set_xlim(start, end)
        ax.set_ylim(0, box_height)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("SEQ", rotation=0, labelpad=25)

    def _render_track(self, ax, xs, values, name):
        """Render based on hardcoded type."""
        if name in self.binary_tracks:
            # Binary: rectangle blocks
            for x, v in zip(xs, values):
                if v > 0.5:
                    ax.add_patch(
                        plt.Rectangle((x, 0), 1, 1, facecolor="#444444", alpha=0.9)
                    )
            ax.set_ylim(0, 1)
        else:
            # Continuous: line + filled area
            ax.plot(xs, values, color="dodgerblue", linewidth=1)
            ax.fill_between(xs, 0, values, color="dodgerblue", alpha=0.3)
            ax.set_ylim(0, max(values) * 1.1 + 1e-6)

        ax.set_xlim(xs[0], xs[-1] + 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(name, rotation=0, labelpad=20)

    def _render_igv_with_sequence(self, regions, tracks_dict, seq_str, figsize=(32, 4)):
        """Render a matplotlib figure for a single sequence."""
        num_tracks = len(tracks_dict) + 1  # +1 for sequence
        fig, axes = plt.subplots(
            num_tracks,
            1,
            figsize=(figsize[0], figsize[1] * num_tracks),
            sharex=True,
            gridspec_kw={"height_ratios": [1] * len(tracks_dict) + [0.4]},
        )

        if num_tracks == 1:
            axes = [axes]

        chrom, start, end = regions["chrom"], regions["start"], regions["end"]
        xs = np.arange(start, end - 1)

        # Numeric tracks
        for ax, (name, values) in zip(axes[:-1], tracks_dict.items()):
            self._render_track(ax, xs, values, name)

        # Sequence track
        seq_ax = axes[-1]
        self._plot_sequence_track(seq_ax, seq_str, start, end)

        axes[-1].set_xlabel(f"{chrom}:{start}-{end}")
        fig.tight_layout()

        return fig

    def eval_batch_end(self, state: State, logger: Logger):
        # print(f"Dataloader Label {state.dataloader_label}")
        # print(state.batch)
        if state.dataloader_label != self.target_eval_label:
            return
        chrom = state.batch_get_item("chrom")
        start = state.batch_get_item("start")
        # Hacky work around for disparte keys in datasets, TODO: reprocess everything to have consitent naming
        try:
            end = state.batch_get_item("end")
        except KeyError as _:
            end = state.batch_get_item("stop")
        annot_mask = state.batch_get_item("annotation_mask")
        loss_weights = state.batch_get_item("loss_weights")  # [B, L]
        min_weight = loss_weights.min()
        max_weight = loss_weights.max()
        is_softmasked = loss_weights == min_weight  # [B,L]
        orig_seqs = state.model.tokenizer.batch_decode(
            state.batch_get_item("input_ids")
        )
        # TODO: Visualize the original seq, softmask, and annot_mask annotations overlaye with bpreds

        B, L, _ = state.outputs.logits.shape
        bpred_out = state.outputs.bpred_output  # list of bpred outputs
        # TODO: remove reshape, it's slow AF but using it to visualize indiv batches
        mask = bpred_out[0].boundary_mask.reshape(B, L)  # [B, L]
        # print(f"Mask shape : {mask.shape}")
        probs = bpred_out[0].boundary_prob[..., 1].reshape(B, L).float()  # [B, L]
        # print(f"Probs shape : {probs.shape}")
        losses = state.outputs.unreduced_loss.reshape(B, L)  # [B, L]
        # print(f"Losses shape : {losses.shape}")

        for batch_idx in range(B):
            if self.log_only_N is not None and self.logged_count >= self.log_only_N:
                return
            seq = orig_seqs[batch_idx]
            region = {
                "chrom": chrom[batch_idx].item(),
                "start": start[batch_idx].item(),
                "end": end[batch_idx].item(),
            }
            tracks = {
                "softmask": is_softmasked[batch_idx].cpu().detach().numpy(),
                "annot": annot_mask[batch_idx].cpu().detach().numpy(),
                "bmask": mask[batch_idx].cpu().detach().numpy(),
                "bprobs": probs[batch_idx].cpu().detach().numpy(),
                "loss": losses[batch_idx].cpu().detach().numpy(),
            }
            fig = self._render_igv_with_sequence(
                regions=region, tracks_dict=tracks, seq_str=seq
            )
            table_row = [
                region["chrom"],
                region["start"],
                region["end"],
                wandb.Image(fig),
            ]
            self.rows.append(table_row)
            plt.close(fig)
            self.logged_count += 1

    def eval_end(self, state: State, logger: Logger):
        if state.dataloader_label != self.target_eval_label:
            return
        logger.log_table(
            columns=self.columns,
            rows=self.rows,
            name="Boundary_Viz",
            step=int(state.timestamp.batch),
        )
        self.logged_count = 0
        self.rows = []


###############################################################
######## Text specific versions from orig H-Net, left as ref ##
###############################################################


def decode_utf8_bytes(byte_list):
    """
    Given a set of bytes, decode them to utf8 chars
    """
    decoded_chars = []
    ch_idx = 0
    byte_idx = 0
    n = len(byte_list)
    stop_decoding = False
    while byte_idx < n:
        b = byte_list[byte_idx]
        if b == 254:
            ch, num_bytes = "<BOS>", 1
        elif b == 255:
            ch, num_bytes = "<EOS>", 1
        else:
            for num_bytes in range(1, 5):  # utf8 chars can be between 1 and 8 bytes
                if byte_idx + num_bytes > n:  # out of range
                    stop_decoding = True
                    break
                chunk = bytearray(byte_list[byte_idx : byte_idx + num_bytes])
                try:  # see if this chunk is valid
                    ch = chunk.decode("utf-8")
                    stop_decoding = False
                    break
                except UnicodeDecodeError:  # chunk is invalid, try another chunk size or error out if out of range
                    stop_decoding = True
            else:
                # Invalid UTF-8
                stop_decoding = True

        if stop_decoding:
            break

        decoded_chars.append(
            (ch_idx, ch, num_bytes)
        )  # break up the bytes into (index in origina byte string, decoded char, num bytes in the decoded char)
        byte_idx += (
            num_bytes  # move forward in byte stre depending on bytes in decoded char
        )
        ch_idx += 1

    return decoded_chars


def match_chars_with_booleans(router_output, decoded_list, loss_tensor, batch_idx):
    """
    Match the UTF-8 deoced characters with appropriate probabilities from the corresponding router output
    router output: [bpred_out:[B,L] for _ in range(num_stages)]
    """
    result = []
    byte_idx = 0

    device = router_output[0].token_mask.device
    use_cu_seqlens = "cu_seqlens" in router_output[0]._fields
    if use_cu_seqlens:
        L_init = router_output[0].cu_seqlens[1]
        token_indices_dicts = []
        token_range = torch.arange(L_init, device=device)  # (L,)
        for depth in range(len(router_output)):
            token_indices_dicts.append(
                {k: v for k, v in zip(token_range.tolist(), range(len(token_range)))}
            )
            token_range = token_range[
                router_output[depth].token_mask[
                    router_output[depth].cu_seqlens[batch_idx] : router_output[
                        depth
                    ].cu_seqlens[batch_idx]
                    + len(token_range)
                ]
            ]
    else:
        token_indices_dicts = []  # map token (byte) idxs into the decoded chars
        token_range = torch.arange(
            router_output[0].token_mask.shape[1], device=device
        )  # (L,)
        for depth in range(len(router_output)):
            token_indices_dicts.append(
                {k: v for k, v in zip(token_range.tolist(), range(len(token_range)))}
            )
            token_range = token_range[
                router_output[depth].token_mask[batch_idx, : len(token_range)]
            ]

    for ch_idx, ch, num_bytes in decoded_list:
        for ch_byte_idx in range(num_bytes):
            if num_bytes == 1:
                bracket = "-"
            else:
                if ch_byte_idx == 0:
                    bracket = "┌"
                elif ch_byte_idx == num_bytes - 1:
                    bracket = "└"
                    ch = " "
                else:
                    bracket = "|"
                    ch = " "

            byte_bools, byte_probs = [], []
            token_index = byte_idx + ch_byte_idx
            for depth in range(len(router_output)):
                if use_cu_seqlens:
                    is_tokenized = (
                        router_output[depth]
                        .token_mask[
                            router_output[depth].cu_seqlens[batch_idx]
                            + token_indices_dicts[depth][token_index]
                        ]
                        .item()
                    )
                    token_prob = (
                        router_output[depth]
                        .selected_probs[
                            router_output[depth].cu_seqlens[batch_idx]
                            + token_indices_dicts[depth][token_index]
                        ]
                        .item()
                    )
                else:
                    is_tokenized = (
                        router_output[depth]
                        .token_mask[batch_idx, token_indices_dicts[depth][token_index]]
                        .item()
                    )
                    token_prob = (
                        router_output[depth]
                        .selected_probs[
                            batch_idx, token_indices_dicts[depth][token_index]
                        ]
                        .item()
                    )

                byte_bools.append(is_tokenized)
                byte_probs.append(token_prob)

                if not is_tokenized:
                    break

            result.append(
                (
                    ch,
                    bracket,
                    byte_bools,
                    byte_probs,
                    loss_tensor[batch_idx, token_index],
                )
            )

        byte_idx += num_bytes

    return result


def render_table_in_columns(
    table_cells,
    rows_per_column=200,
    font_path="DejaVuSansMono.ttf",
    font_size=16,
    column_spacing=50,
    margin=20,
):
    # 1) Split into chunks
    chunks = []
    for i in range(0, len(table_cells), rows_per_column):
        chunk = table_cells[i : i + rows_per_column]
        chunks.append(chunk)

    font = ImageFont.truetype(font_path, font_size)

    # measure_column figures out how wide each column is, and how tall the chunk is
    def measure_column(chunk):
        """
        Given a chunk (list of row-tuples), measure:
          - per-column maximum width
          - per-row height
          - total chunk height
        """
        # chunk[i] = (col0_text, col1_text, col2_text, ...)
        num_cols = len(chunk[0]) if chunk else 0
        col_widths = [0] * num_cols
        row_height = 0

        # measure
        for row in chunk:
            # row is e.g. ("<BOS>", "1", "▖")
            for col_index, text in enumerate(row):
                w, h = font.getsize(text)
                col_widths[col_index] = max(col_widths[col_index], w)
                row_height = max(row_height, h)

        # total chunk height = row_height * number_of_rows
        chunk_height = row_height * len(chunk)
        return col_widths, row_height, chunk_height

    # draw_column draws one chunk at a given x_offset
    def draw_column(draw, chunk, col_widths, row_height, x_offset):
        """
        Draws the chunk (mini-table) at x_offset.
        Each row is drawn in a new line.
        col_widths is a list of the measured widths for each column.
        row_height is the line height.
        """
        y = margin
        for row in chunk:
            x = x_offset
            for col_index, text in enumerate(row):
                draw.text((x, y), text, fill="black", font=font)
                x += (
                    col_widths[col_index] + 10
                )  # 10 px gap between columns in the *same* chunk
            y += row_height

    # 2) measure each chunk
    all_chunks_info = []
    total_width = 0
    max_height = 0

    for chunk_idx, chunk in enumerate(chunks):
        col_widths, row_h, chunk_h = measure_column(chunk)
        all_chunks_info.append((chunk, col_widths, row_h, chunk_h))
        total_width += sum(col_widths) + 10 * (
            len(col_widths) - 1
        )  # sum col widths + internal col gaps
        if chunk_idx < len(chunks) - 1:
            total_width += column_spacing  # spacing to next chunk
        if chunk_h > max_height:
            max_height = chunk_h

    # 3) Build final image
    # Height: max among columns (plus top/bottom margin)
    # Width: sum of chunk widths (plus margins + spacing)
    img_width = total_width + 2 * margin
    img_height = max_height + 2 * margin

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    # 4) Draw each chunk side by side
    x_offset = margin
    for chunk_idx, (chunk, col_widths, row_height, chunk_height) in enumerate(
        all_chunks_info
    ):
        draw_column(draw, chunk, col_widths, row_height, x_offset)
        # move x_offset to the right: width of that chunk + spacing
        chunk_width = sum(col_widths) + 10 * (len(col_widths) - 1)
        x_offset += chunk_width
        if chunk_idx < len(all_chunks_info) - 1:
            x_offset += column_spacing

    return img


def visualize_bytes(
    byte_tensor, router_output, loss_tensor, max_batch_idx=10, max_seq_len=512
):
    images = []
    for b_idx in range(max_batch_idx):
        # List[(ch_idx, ch, num_bytes)]
        # decode bytes into utf8 chars
        decoded_list = decode_utf8_bytes(byte_tensor[b_idx, :max_seq_len].tolist())

        # List[(ch, bracket, byte_bools, byte_probs)]
        # pair up each char with it's associated probs
        matched = match_chars_with_booleans(
            router_output, decoded_list, loss_tensor, b_idx
        )

        table_cells = []
        for ch, bracket, byte_bools, byte_probs, byte_loss in matched:
            byte_masks_str = " ".join(
                "■" if byte_bool else "□" for byte_bool in byte_bools
            )
            byte_probs_str = "|" + " ".join(
                str(round(byte_prob, 2)) for byte_prob in byte_probs
            )
            byte_loss_str = "|" + str(round(byte_loss.item(), 2))

            table_cells.append(
                (ch, bracket, byte_masks_str, byte_probs_str, byte_loss_str)
            )

        if os.path.exists("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
            # needed for Chinese characters, only installed for Chinese runs
            font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        else:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

        img = render_table_in_columns(
            table_cells,
            rows_per_column=32,
            font_path=font_path,
            font_size=16,
            column_spacing=50,
            margin=20,
        )
        images.append(img)

    return images
