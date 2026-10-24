"""Exact completion-only causal cross entropy and measured Trainer windows.

PyTorch is imported only when a tensor operation is requested.
"""

from __future__ import annotations


def completion_projection_inputs(labels):
    """Select exactly the causal logits whose targets are supervised tokens.

    The backbone still processes the complete context. With prompt length P,
    the first completion label at P is predicted by the logit at P-1. Explicit
    shifted targets prevent Qwen's loss from shifting the shortened array again.
    """
    import torch
    from torch.nn import functional as functional

    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("Completion-only projection requires microbatch size one")
    if labels.shape[1] < 2 or bool(labels[:, 0].ne(-100).any()):
        raise ValueError("A masked prompt before every completion is required")
    shifted = functional.pad(labels, (0, 1), value=-100)[:, 1:]
    positions = torch.nonzero(shifted[0].ne(-100), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("Completion-only projection requires supervised assistant tokens")
    return {"logits_to_keep": positions, "shift_labels": shifted.index_select(1, positions)}


def completion_only_forward_loss(model, inputs, *, num_items_in_batch=None, return_outputs=False):
    """Use Qwen3's stock CE, projecting no masked prompt/history positions."""
    selected = completion_projection_inputs(inputs["labels"])
    kwargs = {**inputs, **selected}
    if num_items_in_batch is not None:
        kwargs["num_items_in_batch"] = num_items_in_batch
    outputs = model(**kwargs)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    return (loss, outputs) if return_outputs else loss


class CompletionOnlyLossMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.args.per_device_train_batch_size != 1
            or self.args.n_gpu > 1
            or self.accelerator.num_processes != 1
        ):
            raise ValueError(
                "Completion-only trainer is validated for a single device and microbatch one"
            )
        # Trainer must count all unmasked labels across accumulated batches and
        # pass that divisor. The stock Qwen CE consumes it; Trainer consequently
        # must not divide the loss by gradient_accumulation_steps a second time.
        self.model_accepts_loss_kwargs = True
        self._observed_update_work = {}

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Count only successfully processed microbatches. These measurements do
        # not alter loss scaling, batch scheduling, or the optimizer operation.
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        step = int(self.state.global_step)
        window = self._observed_update_work.setdefault(
            step,
            {
                "optimizer_step_start": step,
                "microbatches": 0,
                "example_presentations": 0,
                "supervised_token_presentations": 0,
            },
        )
        window["microbatches"] += 1
        window["example_presentations"] += int(inputs["labels"].shape[0])
        window["supervised_token_presentations"] += int(inputs["labels"].ne(-100).sum().item())
        return loss

    def observed_training_work(self):
        windows = [
            dict(self._observed_update_work[key]) for key in sorted(self._observed_update_work)
        ]
        return {
            "scope": "current_train_invocation",
            "configured_gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "configured_per_device_train_batch_size": self.args.per_device_train_batch_size,
            "actual_microbatches": sum(row["microbatches"] for row in windows),
            "actual_example_presentations": sum(row["example_presentations"] for row in windows),
            "actual_supervised_token_presentations": sum(
                row["supervised_token_presentations"] for row in windows
            ),
            "actual_microbatches_per_optimizer_update": sorted(
                {row["microbatches"] for row in windows}
            ),
            "optimizer_update_windows": windows,
            "global_step_after_training": int(self.state.global_step),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return completion_only_forward_loss(
            model, inputs, num_items_in_batch=num_items_in_batch, return_outputs=return_outputs
        )
