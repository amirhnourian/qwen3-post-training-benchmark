"""Independent CPU checks of the optional group-relative policy objective.

No production checkpoint, sampled reward, or CUDA device is used here. Reference
values below are analytic, so these checks can detect sign, shift, masking,
normalization, clipping, and temperature errors in the actual objective.
"""

# Optional dependency gates must run before ML imports.
# ruff: noqa: E402
from __future__ import annotations

import copy
import math
import unittest
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from qwen_posttrain.grpo import group_advantages, grpo_loss, log_probs_for_step


class GroupRelativeObjectiveTests(unittest.TestCase):
    def test_group_advantages_have_population_unit_variance(self):
        rewards = [0.0, 0.6, 1.0, 0.8]
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
        actual = torch.as_tensor(group_advantages(rewards), dtype=torch.float64)
        expected = torch.tensor([(r - mean) / (std + 1e-8) for r in rewards], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-7)
        self.assertAlmostEqual(float(actual.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(actual.square().mean()), 1.0, places=6)

    def test_constant_reward_groups_have_no_artificial_advantage(self):
        # This includes both saturation and a wholly unsuccessful group.
        for rewards in ([1.0] * 4, [0.0] * 4, [0.6] * 4):
            actual = torch.as_tensor(group_advantages(rewards), dtype=torch.float64)
            self.assertEqual(actual.shape, (4,))
            self.assertTrue(bool(torch.isfinite(actual).all()))
            torch.testing.assert_close(actual, torch.zeros(4, dtype=torch.float64), atol=0, rtol=0)

    def test_zero_valued_on_policy_group_loss_still_has_policy_gradient(self):
        advantages = group_advantages([0.0, 1.0])
        bad = torch.tensor([-0.4, -0.8], dtype=torch.float64, requires_grad=True)
        good = torch.tensor([-0.2, -0.3, -0.7, -0.5], dtype=torch.float64, requires_grad=True)
        # Original GRPO: mean within trajectory, then mean over trajectories.
        loss = (
            grpo_loss(bad, bad.detach().clone(), advantages[0])
            + grpo_loss(good, good.detach().clone(), advantages[1])
        ) / 2
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=12)
        loss.backward()
        magnitude = 1 / (1 + 2e-8)  # Population std of [0, 1] is 1/2.
        torch.testing.assert_close(
            bad.grad, torch.full_like(bad, magnitude / 4), atol=1e-8, rtol=1e-7
        )
        torch.testing.assert_close(
            good.grad, torch.full_like(good, -magnitude / 8), atol=1e-8, rtol=1e-7
        )
        self.assertGreater(float(bad.grad.abs().sum() + good.grad.abs().sum()), 0)

    def test_positive_advantage_clips_only_excessive_probability_increase(self):
        ratio = torch.tensor([1.5, 0.5], dtype=torch.float64)
        logp = ratio.log().requires_grad_()
        loss = grpo_loss(logp, torch.zeros_like(logp), 1.0, clip_epsilon=0.2)
        self.assertAlmostEqual(float(loss.detach()), -0.85, places=12)
        loss.backward()
        torch.testing.assert_close(
            logp.grad, torch.tensor([0.0, -0.25], dtype=torch.float64), atol=1e-12, rtol=1e-12
        )

    def test_negative_advantage_clips_only_excessive_probability_decrease(self):
        ratio = torch.tensor([1.5, 0.5], dtype=torch.float64)
        logp = ratio.log().requires_grad_()
        loss = grpo_loss(logp, torch.zeros_like(logp), -1.0, clip_epsilon=0.2)
        self.assertAlmostEqual(float(loss.detach()), 1.15, places=12)
        loss.backward()
        torch.testing.assert_close(
            logp.grad, torch.tensor([0.75, 0.0], dtype=torch.float64), atol=1e-12, rtol=1e-12
        )

    def test_group_weighting_does_not_treat_each_tool_turn_as_a_trajectory(self):
        # A one-token turn and a three-token turn share one episode reward.
        first = torch.tensor([-0.3], dtype=torch.float64, requires_grad=True)
        second = torch.tensor([-0.7, -0.5, -0.2], dtype=torch.float64, requires_grad=True)
        advantage = 0.75
        combined = torch.cat((first, second))
        dense = grpo_loss(combined, combined.detach(), advantage)
        split = grpo_loss(first, first.detach(), advantage) * (len(first) / 4) + grpo_loss(
            second, second.detach(), advantage
        ) * (len(second) / 4)
        torch.testing.assert_close(split, dense, atol=1e-12, rtol=1e-12)
        dense_grad = torch.autograd.grad(dense, (first, second), retain_graph=True)
        split_grad = torch.autograd.grad(split, (first, second))
        for expected, actual in zip(dense_grad, split_grad):
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            split_grad[0], torch.full_like(first, -advantage / 4), atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            split_grad[1], torch.full_like(second, -advantage / 4), atol=1e-12, rtol=1e-12
        )

    def test_old_policy_is_detached_even_if_caller_provides_a_grad_tensor(self):
        new = torch.tensor([-0.2, -0.4], dtype=torch.float64, requires_grad=True)
        old = torch.tensor([-0.3, -0.5], dtype=torch.float64, requires_grad=True)
        grpo_loss(new, old, 0.5).backward()
        self.assertIsNone(old.grad)
        self.assertIsNotNone(new.grad)
        self.assertGreater(float(new.grad.abs().sum()), 0)

    def test_fixed_behavior_probability_enables_clipping_on_second_inner_update(self):
        # An analytic one-parameter example: first update moves the new/old
        # ratio beyond 1.2; the second update must still compare to that SAME
        # original behavior policy, so its clipped policy gradient is zero.
        parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        old = parameter.detach().clone()
        optimizer = torch.optim.SGD([parameter], lr=0.3)
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = grpo_loss(parameter, old, 1.0, clip_epsilon=0.2)
            loss.backward()
            if iteration == 0:
                self.assertAlmostEqual(float(parameter.grad), -1.0, places=12)
            else:
                self.assertAlmostEqual(float(parameter.grad), 0.0, places=12)
            optimizer.step()
        self.assertAlmostEqual(float(parameter.detach()), 0.3, places=12)
        self.assertAlmostEqual(float(old), 0.0, places=12)

    def test_invalid_nonfinite_or_too_small_reward_groups_are_rejected(self):
        for rewards in ([], [1.0], [0.0, float("nan")], [float("inf"), 1.0]):
            with self.subTest(rewards=rewards):
                with self.assertRaises(ValueError):
                    group_advantages(rewards)

    def test_zero_advantage_contributes_no_reward_gradient(self):
        new = torch.tensor([-0.2, -0.8, -1.5], dtype=torch.float64, requires_grad=True)
        old = torch.tensor([-0.4, -0.9, -1.2], dtype=torch.float64)
        loss = grpo_loss(new, old, 0.0)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        torch.testing.assert_close(new.grad, torch.zeros_like(new), atol=0, rtol=0)


class PositionLogitModel(torch.nn.Module):
    """Tiny deterministic logits; records the actual causal projection indices."""

    def __init__(self):
        super().__init__()
        values = torch.arange(32 * 17, dtype=torch.float32, device="cpu")
        self.table = torch.nn.Parameter(values.sin().reshape(32, 17))
        self.positions = None

    @property
    def device(self):
        return self.table.device

    def forward(self, input_ids, attention_mask, logits_to_keep, use_cache):
        self.positions = logits_to_keep.detach().cpu().tolist()
        assert not use_cache
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        logits = self.table[input_ids]
        return SimpleNamespace(logits=logits.index_select(1, logits_to_keep))


def tiny_cpu_qwen():
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM

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
    with torch.device("cpu"):
        return Qwen3ForCausalLM(config)


class ExactRolloutLogProbabilityTests(unittest.TestCase):
    def test_exact_causal_positions_include_eos_and_mask_every_prompt_token(self):
        model = PositionLogitModel()
        prompt, generated = [3, 5, 7], [11, 13, 2]
        actual = log_probs_for_step(model, prompt, generated, temperature=1.0)
        # Predict generated token 11 from the preceding prompt token 7,
        # token 13 from generated token 11, and EOS 2 from token 13.
        expected = torch.stack(
            [
                model.table[7].log_softmax(-1)[11],
                model.table[11].log_softmax(-1)[13],
                model.table[13].log_softmax(-1)[2],
            ]
        )
        self.assertEqual(model.positions, [2, 3, 4])
        self.assertEqual(actual.shape, (len(generated),))
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=1e-6)
        actual.sum().backward()
        # Rows for prompt tokens 3 and 5 and for the final EOS input are not
        # separate prediction targets; only each causal predecessor contributes.
        for row in (3, 5, 2):
            torch.testing.assert_close(model.table.grad[row], torch.zeros(17), atol=0, rtol=0)
        self.assertGreater(float(model.table.grad[7].abs().sum()), 0)

    def test_temperature_is_applied_to_full_support_logits_before_normalizing(self):
        model = PositionLogitModel()
        prompt, generated = [3, 5], [7, 2]
        temperature = 2.0
        actual = log_probs_for_step(model, prompt, generated, temperature)
        reference = torch.stack(
            [
                (model.table[5] / temperature).log_softmax(-1)[7],
                (model.table[7] / temperature).log_softmax(-1)[2],
            ]
        )
        wrong_temperature = torch.stack(
            [model.table[5].log_softmax(-1)[7], model.table[7].log_softmax(-1)[2]]
        )
        torch.testing.assert_close(actual, reference, atol=5e-7, rtol=1e-6)
        self.assertFalse(torch.allclose(actual, wrong_temperature, atol=1e-4, rtol=1e-4))

    def test_selected_qwen_log_probabilities_and_all_gradients_match_dense_cpu_reference(self):
        dense = tiny_cpu_qwen()
        selected = copy.deepcopy(dense)
        prompt, generated, temperature = [3, 5, 7], [11, 13, 2], 2.0
        ids = torch.tensor([prompt + generated], dtype=torch.long, device="cpu")
        logits = dense(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        causal = logits[0, len(prompt) - 1 : -1, :].float() / temperature
        target = torch.tensor(generated, dtype=torch.long, device="cpu")
        expected = causal.log_softmax(-1).gather(1, target[:, None]).squeeze(1)
        actual = log_probs_for_step(selected, prompt, generated, temperature)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        (-expected.mean()).backward()
        (-actual.mean()).backward()
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

    def test_invalid_empty_prompts_or_completions_and_temperature_are_rejected(self):
        model = PositionLogitModel()
        for prompt, generated, temperature in (
            ([], [2], 1),
            ([3], [], 1),
            ([3], [2], 0),
            ([3], [2], -1),
        ):
            with self.subTest(prompt=prompt, generated=generated, temperature=temperature):
                with self.assertRaises(ValueError):
                    log_probs_for_step(model, prompt, generated, temperature)


if __name__ == "__main__":
    unittest.main()
