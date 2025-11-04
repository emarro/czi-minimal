import torch

from collections import namedtuple
from composer.models import HuggingFaceModel
from torchmetrics import PearsonCorrCoef
from torchmetrics.aggregation import MeanMetric, RunningMean
from torchmetrics.classification import MulticlassAccuracy


class ComposerWrapper(HuggingFaceModel):
    def __init__(self, *args, mlm=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_pcc = PearsonCorrCoef()
        self.val_loss = MeanMetric()
        self.val_loss.tag = ""
        self.train_ar_loss = RunningMean()
        self.train_ar_loss.tag = "ar"
        self.val_ar_loss = MeanMetric()
        self.val_ar_loss.tag = "ar"
        self.train_ratio_loss = RunningMean()
        self.train_ratio_loss.tag = "ratio"
        self.val_ratio_loss = MeanMetric()
        self.val_ratio_loss.tag = "ratio"

        self.train_acc = MulticlassAccuracy(average="micro", ignore_index=-100)
        self.train_acc.tag = "acc"

        self.val_acc = MulticlassAccuracy(average="micro", ignore_index=-100)
        self.val_acc.tag = "acc"

        self.mlm = mlm

    def eval_forward(self, batch, outputs=None):
        alt_outputs = None  # hacky placeholder for second fwd pass for ALT seqs in VEP
        if outputs:
            return outputs
        ref_ids = batch["input_ids"].detach().clone()
        if "ref_id" in batch and not self.mlm:
            # if we're doning a VEP task, it makes life much easier to do the ALT fwd pass here and shove it into the output tuple
            ref_bp_id = batch["ref_id"]  # [B]
            ref_ids = batch["input_ids"].detach().clone()
            B, seq_len = ref_ids.shape
            var_idx = seq_len // 2
            # update (possibly masked) token with ref
            ref_ids[:, var_idx] = ref_bp_id

            assert torch.all(ref_ids[:, var_idx] == ref_bp_id), (
                f"REF bps from batch do not match the input ids, IDS: {ref_ids[:, var_idx - 1 : var_idx + 2]}, ref_bps: {ref_bp_id}"
            )
            alt_bp_id = batch["alt_id"]  # [B]
            alt_ids = batch["input_ids"].detach().clone()
            B, seq_len = alt_ids.shape
            # update (possibly masked) token with alt
            alt_ids[:, var_idx] = alt_bp_id
            assert torch.all(ref_bp_id != alt_bp_id), (
                f"REF and ALT bps are the same bp for some batches. REF: {ref_bp_id[ref_bp_id == alt_bp_id]} ALT: {alt_bp_id[ref_bp_id == alt_bp_id]}"
            )
            assert torch.all(ref_ids[:, :var_idx] == alt_ids[:, :var_idx]), (
                "Not all ids before the variant site match"
            )
            assert torch.all(ref_ids[:, var_idx + 1 :] == alt_ids[:, var_idx + 1 :]), (
                "Not all ids after the variant site match"
            )
            assert torch.all(ref_ids[:, var_idx] != alt_ids[:, var_idx]), (
                f"Some variants have the same ALT and REF bp, REF: {ref_ids[:, var_idx][ref_ids[:, var_idx] == alt_ids[:, var_idx]]} ALT: {alt_ids[:, var_idx][ref_ids[:, var_idx] == alt_ids[:, var_idx]]}"
            )

            alt_outputs = self.model(input_ids=alt_ids)

        # Sometimes batches have extra info for eval, some downstream models don't handle unparsed kwargs well
        new_batch = {
            "input_ids": ref_ids,
            "labels": batch["labels"],
            "loss_weights": batch["loss_weights"],
        }

        if "target_ratio" in batch:
            new_batch["target_ratio"] = batch["target_ratio"]

        batch = new_batch

        outputs = self.model(**batch)

        if alt_outputs is not None:
            # hacky way to add the alternate input to the output tuple posthoc
            outputs = outputs._asdict()
            outputs["alt_outputs"] = alt_outputs
            new_namedtuple = namedtuple("CausalLMOutputsforZS", outputs.keys())
            outputs = new_namedtuple(**outputs)
        return outputs

    def update_metric(self, batch, outputs, metric) -> None:
        """
        Update metric by returning as a socre the (log) ref/alt probabilities
        Args:
            batch: dict[str, Tensor] the input batch.
            outputs: MaskedLMOutput['logits': Tensor(batch, seq_len, vocab_len), 'loss': float]
            metric: torchmetrics.Metric the metric we're updating
        """
        # TODO: Redo by shoving all the evals for each split into a collection class?
        if (
            len(batch.keys()) == 5 or "ref_id" not in batch
        ):  # not in the zero-shot eval task
            val = None
            if metric.tag == "ar":
                val = outputs.ar_loss if not self.mlm else None
            elif metric.tag == "ratio":
                val = outputs.ratio_loss if not self.mlm else None
            elif metric.tag == "acc":
                B, L, V = outputs.logits.shape
                preds = outputs.logits.softmax(dim=-1).argmax(dim=-1)
                labels = batch["labels"]
                if self.mlm:
                    # if MLM only count
                    preds[labels == -100] = 0
                    labels[labels == -100] = 0
                metric.update(preds.view(-1), labels.view(-1))
                return
            else:
                val = outputs.loss
            if val is not None:
                metric.update(value=val)
            return
        probs = outputs.logits.softmax(dim=-1)
        batch_size, seq_len, vocab_len = outputs.logits.shape

        if self.mlm:
            ref_bp = batch["ref_id"]  # [batch_size]
            ref_prob = torch.gather(
                probs[:, (seq_len // 2) - 1, :], dim=1, index=ref_bp.unsqueeze(1)
            ).squeeze(1)

            alt_bp = batch["alt_id"]  # [batch_size]
            alt_prob = torch.gather(
                probs[:, (seq_len // 2) - 1, :], dim=1, index=alt_bp.unsqueeze(1)
            ).squeeze(1)

            assert len(probs.shape) == 3, (
                f"Expected probs of shape [batch, seq_len, vocba_len], found {probs.shape}"
            )
            assert (probs < 0).sum() == 0, (
                f"Found probabilities less than 0 in probs: {probs[probs < 0]}"
            )
            assert (probs > 1).sum() == 0, (
                f"Found probabilities greater than 1 in probs: {probs[probs > 1]}"
            )
            assert ((torch.abs(probs.sum(dim=-1)) - 1) > 1e-5).sum() == 0, (
                f"Probabilities in probs do not normalize to 1: {probs[(torch.abs(probs.sum(dim=-1)) - 1) > 1e-5]}"
            )
            score = torch.log(alt_prob / ref_prob)
        else:
            alt_probs = outputs.alt_outputs.logits.softmax(dim=-1)
            ref_bp = batch["ref_id"]  # [B]
            alt_bp = batch["alt_id"]  # [B]
            input_ids = batch["input_ids"]  # [B, L]
            var_idx = seq_len // 2

            assert torch.all(input_ids[:, var_idx] == ref_bp), (
                f"REF bps from batch do not match the input ids, IDS: {input_ids[:, var_idx - 1 : var_idx + 2]}, ref_bps: {ref_bp}"
            )
            assert torch.all(ref_bp != alt_bp), (
                "Not all REF and  ALT bps are different, error in pre-processing"
            )

            ref_log_probs = torch.log(
                torch.gather(probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(2)
            )
            alt_ids = input_ids.detach().clone()  # [B, L]
            alt_ids[:, var_idx] = alt_bp
            assert torch.all(input_ids[:, :var_idx] == alt_ids[:, :var_idx]), (
                "Not all ids before the variant site match"
            )
            assert torch.all(
                input_ids[:, var_idx + 1 :] == alt_ids[:, var_idx + 1 :]
            ), "Not all ids after the variant site match"
            ref_check = input_ids[:, var_idx]
            alt_check = alt_ids[:, var_idx]
            not_match = ref_check != alt_check

            assert torch.all(not_match), (
                f"Some REF and ALT sequences have the same BP at the variant site, REFs: {ref_check[~not_match]} ({ref_check[~not_match].shape}), ALTs: {alt_check[~not_match]} ({alt_check[~not_match].shape})"
            )

            alt_log_probs = torch.log(
                torch.gather(alt_probs, dim=-1, index=alt_ids.unsqueeze(-1)).squeeze(2)
            )
            ref_pll = ref_log_probs.mean(dim=-1)
            alt_pll = alt_log_probs.mean(dim=-1)
            score = alt_pll - ref_pll

        maf = batch["MAF"]  # the 'label' [batch_size]

        metric.update(preds=score, target=maf)

    def get_metrics(self, is_train=False):
        if is_train:
            return {
                "ARLoss": self.train_ar_loss,
                "RatioLoss": self.train_ratio_loss,
                "Accuracy": self.train_acc,
            }
        return {
            "PearsonCorrCoef": self.val_pcc,
            "EvalLoss": self.val_loss,
            "ARLoss": self.val_ar_loss,
            "RatioLoss": self.val_ratio_loss,
            "Accuracy": self.val_acc,
        }
