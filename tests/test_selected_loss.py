"""Tiny random CPU Qwen tests: selective projection preserves the RSFT loss."""

# Optional dependency gates must run before ML imports.
# ruff: noqa: E402
from __future__ import annotations

import copy
import tempfile
import unittest

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments

from qwen_posttrain import losses as TRAIN
from qwen_posttrain.modeling import AssistantOnlyCollator


def tiny_qwen():
    config = Qwen3Config(
        vocab_size=29,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=0,
        eos_token_id=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    # Explicit CPU device context: no real checkpoint or accelerator allocation.
    with torch.device("cpu"):
        return Qwen3ForCausalLM(config)


def first_example():
    return {
        "input_ids": torch.tensor([[3, 5, 7, 11, 13, 2, 0, 0]], device="cpu"),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], device="cpu"),
        "labels": torch.tensor([[-100, -100, -100, 11, 13, 2, -100, -100]], device="cpu"),
    }


def second_example():
    return {
        "input_ids": torch.tensor([[4, 9, 2, 0]], device="cpu"),
        "attention_mask": torch.tensor([[1, 1, 1, 0]], device="cpu"),
        "labels": torch.tensor([[-100, -100, 2, -100]], device="cpu"),
    }


class CompletionProjectionTests(unittest.TestCase):
    def assert_same_gradients(self, dense, selected):
        for (name, parameter), (other_name, other_parameter) in zip(
            dense.named_parameters(), selected.named_parameters()
        ):
            self.assertEqual(name, other_name)
            if parameter.grad is None:
                self.assertIsNone(other_parameter.grad, name)
            else:
                torch.testing.assert_close(
                    parameter.grad,
                    other_parameter.grad,
                    atol=2e-6,
                    rtol=2e-5,
                    msg=lambda message: f"{name}: {message}",
                )

    def test_positions_include_first_completion_and_eos_exclude_prompt_and_padding(self):
        selection = TRAIN.completion_projection_inputs(first_example()["labels"])
        self.assertEqual(selection["logits_to_keep"].tolist(), [2, 3, 4])
        self.assertEqual(selection["shift_labels"].tolist(), [[11, 13, 2]])

    def test_dense_and_selected_loss_and_all_parameter_gradients_match(self):
        dense = tiny_qwen()
        selected = copy.deepcopy(dense)
        batch = first_example()
        dense_loss = dense(**batch).loss
        selected_loss, outputs = TRAIN.completion_only_forward_loss(
            selected, batch, return_outputs=True
        )
        self.assertEqual(tuple(outputs.logits.shape), (1, 3, 29))
        torch.testing.assert_close(dense_loss, selected_loss, atol=1e-6, rtol=1e-6)
        dense_loss.backward()
        selected_loss.backward()
        self.assert_same_gradients(dense, selected)

    def test_accumulated_token_divisor_matches_dense_weighted_objective(self):
        dense = tiny_qwen()
        selected = copy.deepcopy(dense)
        batches = [first_example(), second_example()]
        # Trainer counts 3 + 1 supervised tokens over this accumulation window.
        divisor = sum(batch["labels"].ne(-100).sum() for batch in batches)
        self.assertEqual(divisor.item(), 4)
        dense_total = sum(dense(**batch, num_items_in_batch=divisor).loss for batch in batches)
        selected_losses = [
            TRAIN.completion_only_forward_loss(selected, batch, num_items_in_batch=divisor)
            for batch in batches
        ]
        selected_total = sum(selected_losses)
        torch.testing.assert_close(dense_total, selected_total, atol=1e-6, rtol=1e-6)
        dense_total.backward()
        for loss in selected_losses:
            loss.backward()
        self.assert_same_gradients(dense, selected)

    def test_trainer_counts_tokens_and_preserves_loss_kwargs_contract(self):
        class SelectiveTrainer(TRAIN.CompletionOnlyLossMixin, Trainer):
            pass

        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                use_cpu=True,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                report_to=[],
                disable_tqdm=True,
            )
            trainer = SelectiveTrainer(model=tiny_qwen(), args=arguments)
            self.assertTrue(trainer.model_accepts_loss_kwargs)
            batches = [first_example(), second_example()]
            count = trainer._get_num_items_in_batch(batches, torch.device("cpu"))
            self.assertEqual(count.item(), 4)
            actual = trainer.compute_loss(trainer.model, batches[0], num_items_in_batch=count)
            expected = trainer.model(**batches[0], num_items_in_batch=count).loss
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_rejects_multiple_examples_or_no_supervised_tokens(self):
        labels = first_example()["labels"]
        for invalid in (labels.repeat(2, 1), torch.full_like(labels, -100)):
            with self.assertRaises(ValueError):
                TRAIN.completion_projection_inputs(invalid)

    def test_observed_work_counts_short_epoch_windows_and_supervised_tokens(self):
        class SelectiveTrainer(TRAIN.CompletionOnlyLossMixin, Trainer):
            pass

        # A synthetic two-example dataset ends its epoch before the configured
        # accumulation window. Direct counters must reflect actual short windows.
        examples = []
        for token_count in (65, 67):
            targets = [11] * (token_count - 1) + [2]
            examples.append(
                {
                    "input_ids": [3, 5] + targets,
                    "attention_mask": [1] * (token_count + 2),
                    "labels": [-100, -100] + targets,
                }
            )
        model = tiny_qwen()
        model.config.max_position_embeddings = 128
        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                use_cpu=True,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                max_steps=2,
                report_to=[],
                disable_tqdm=True,
                save_strategy="no",
                logging_strategy="no",
            )
            trainer = SelectiveTrainer(
                model=model,
                args=arguments,
                train_dataset=examples,
                data_collator=AssistantOnlyCollator(0),
            )
            trainer.train()
            observed = trainer.observed_training_work()
            self.assertEqual(observed["configured_gradient_accumulation_steps"], 4)
            self.assertEqual(observed["actual_microbatches"], 4)
            self.assertEqual(observed["actual_example_presentations"], 4)
            self.assertEqual(observed["actual_supervised_token_presentations"], 264)
            self.assertEqual(observed["actual_microbatches_per_optimizer_update"], [2])
            self.assertEqual(observed["global_step_after_training"], 2)
            self.assertEqual(
                [
                    row["supervised_token_presentations"]
                    for row in observed["optimizer_update_windows"]
                ],
                [132, 132],
            )


if __name__ == "__main__":
    unittest.main()
