# A practical tutorial: verified rollouts, RSFT and GRPO

This tutorial has two tracks. The public arithmetic demo teaches the interfaces
without a GPU. The measured study used a separate private task; its aggregate
results illustrate how to assess a training experiment. Running the demo does
not reproduce those results.

Start with the [report](../reports/qwen3_post_training_report.pdf), then follow
the steps below. [Methods](methods.md) gives the experiment specification;
[reproducibility](reproducibility.md) distinguishes public checks from rerunning
the private task. Each harder concept has a primary reading link.

## 1. Follow one complete attempt

Install the CPU development environment described in the [README](../README.md),
then run:

```bash
qwen-posttrain demo --task examples/arithmetic/task.json --output runs/demo.jsonl
```

The synthetic generator reads an expression, calls a bounded calculator,
writes an answer, and submits it. The environment owns the input and checker;
the generator chooses actions. Each call and response becomes part of the
next observation. One complete sequence is a **rollout**, or episode.

Open [`environment.py`](../src/qwen_posttrain/environment.py) and
[`rollouts.py`](../src/qwen_posttrain/rollouts.py). Notice that each attempt
starts with fresh environment state. The same interfaces support a neural
generator, but the CPU demo is deterministic and explicitly labels itself as
such. Its illustrative character IDs are not Qwen token IDs.

**Try:** inspect the saved JSONL row. Find `steps`, `tool_results`, `answer`,
`submitted` and `score`. Count complete attempts separately from tool calls.
Explain which parts came from the environment and which came from the generator.

## 2. Define the metric before sampling

A strict pass requires every declared task check to pass. Fractional reward
can describe partial correctness. An attempt with high partial reward can
still fail strictly. Keep wrong answers, invalid tool calls and exhausted
budgets in the task-performance denominator; infrastructure exceptions need
their own explicit accounting policy.

For the private measured task, the baseline produced **0 strict passes in 100
attempts**. Historical or pilot outcomes were not mixed into that cohort.
The environment interface was repaired before the primary comparison and
held fixed afterward. That avoids treating a setup repair as a training effect.

**Try:** explain why counting each tool call as a new sample would exaggerate
the sample size. Read [`analysis.py`](../src/qwen_posttrain/analysis.py) alongside
the [methods](methods.md#evaluation-and-uncertainty).

## 3. Treat prompts and sampling as part of the policy

Qwen receives tokens, not Python message dictionaries. Its chat template turns
roles, messages and tool definitions into the token sequence the model sees.
Changing a template, thinking mode, temperature, top-p or token budget can
change success without changing a single weight.

The reported before/after cohorts used matching inputs, tools, quantization,
template, decoding settings and budgets. Their seeds were separate. The
trained adapter was the intended model change. The public refactor records
configuration and model identities to make such comparisons inspectable.

**Try:** find `GenerationConfig` and `parse_assistant_output` in
[`modeling.py`](../src/qwen_posttrain/modeling.py). Identify which settings
affect generation and which delimit tool calls.

**Read:** the [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) and
[Transformers chat templates](https://huggingface.co/docs/transformers/v4.57.1/chat_templating).

## 4. Collect successes without inventing targets

RSFT means sampling attempts, verifying them, rejecting failures from the
supervised set, and learning from accepted model-generated targets. Its
usefulness depends on finding at least one correct attempt.

The measured independent collection found none. A subsequent assisted search
highlighted relevant source information already present in the task. That
task-specific guidance helped the same base model produce successful actions.
After strict filtering and exact-action deduplication, the accepted set held
**one distinct trajectory, two assistant-turn examples and 132 supervised
tokens per dataset pass**.

There are two separate questions: **who generated the target, and what help
did that generation receive?** Exact own-model tokens answer the first. They
do not make assisted collection unassisted. Removing the guidance from the
student prompt does not erase its influence on the target.

**Try:** read `build_dataset` in [`dataset.py`](../src/qwen_posttrain/dataset.py).
Find checks for strict success, unadapted source identity and duplicate targets.
The deterministic demo is not a source of own-Qwen benchmark demonstrations.

**Read:** [ReST](https://arxiv.org/abs/2308.08998) for related
generate–filter–train ideas. Its full method differs from this study.

## 5. Understand exactly which tokens are trained

A causal language model predicts the next token from previous context. In an
assistant-only example, system instructions, user input and tool replies remain
visible but receive label `-100`. Their positions do not contribute directly
to cross-entropy loss. The selected assistant targets do.

The measured action-only transformation kept exact generated action suffixes,
including completion boundaries, and reconstructed the frozen student prompt.
It excluded source reasoning and collection guidance. It did not rewrite
answers. [`dataset.py`](../src/qwen_posttrain/dataset.py) contains separate
full-trajectory and action-only paths; `action_suffix` checks token boundaries.

```text
input:  [system / user / tool context] [assistant action tokens]
labels: [-100, -100, ...             ] [the actual target IDs]
```

The final context token predicts the first action token. Losing that causal
shift is an easy implementation mistake. [`losses.py`](../src/qwen_posttrain/losses.py)
projects only positions needed to predict supervised tokens while retaining
the complete transformer context; small-model tests check loss and gradient
equivalence with the dense calculation.

**Try:** explain why masking the prompt does not make the source information
invisible. Then explain why masking alone need not prevent allocation of a
large full-vocabulary logit tensor.

**Read:** [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html)
for `ignore_index`, and the pinned
[Qwen modeling source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py)
for `logits_to_keep`.

## 6. See why QLoRA fits this workstation

LoRA freezes a layer's original matrix and learns a low-rank update:

```text
W_effective = W_frozen + (alpha / rank) * B * A
```

For an illustrative 4,096-by-4,096 layer, rank 16 uses
`16 × (4096 + 4096) = 131072` adapter parameters instead of changing all
16,777,216 original entries. This example describes one layer, not the whole
adapter. The measured study used rank 16 and alpha 32 across selected projections.

QLoRA adds quantized frozen base weights. NF4 is a four-bit storage format;
the model's arithmetic and activations can still use BF16. A saved adapter
therefore requires the matching base model. It is not a standalone 8B model.

**Try:** trace where [`rsft.py`](../src/qwen_posttrain/rsft.py) freezes the base
and attaches trainable adapter matrices. Identify the parameters passed to the
optimizer.

**Read:** [LoRA](https://arxiv.org/abs/2106.09685),
[QLoRA](https://arxiv.org/abs/2305.14314), and the practical
[PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization).

## 7. Separate examples, microbatches, updates and checkpoints

A microbatch supplies one forward/backward computation. Gradient accumulation
combines microbatches before an optimizer update. An epoch is a pass through
the dataset. None of these creates a new independent successful trajectory.
Tiny datasets can revisit the same target many times.

The measured RSFT run completed **40 cumulative updates**, including recovery
from checkpoint **26 to 40**. Recovery restored adapter, optimizer, scheduler
and RNG state under the same data and optimization settings. Loading adapter
weights alone is a different operation. The checkpoint was not selected using
the final evaluation.

Activation checkpointing is another distinct concept: it recomputes activations
during backpropagation to reduce memory. It is unrelated to saving a recovery
checkpoint on disk.

**Try:** explain why finishing steps 27–40 does not mean starting a second
40-update experiment. Read resume validation in
[`modeling.py`](../src/qwen_posttrain/modeling.py) and the trainer configuration.

**Read:** [Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.1/trainer)
and [PyTorch activation checkpointing](https://docs.pytorch.org/docs/2.7/checkpoint.html).

## 8. Understand how GRPO changes the learning rule

The optional continuation generated fresh groups of attempts from the current
policy. Each attempt received the original task reward. Relative advantage
compares its reward with the rest of its group:

```text
advantage_i = (reward_i - group_mean) / (group_std + epsilon)
ratio_t = exp(current_log_probability_t - saved_old_log_probability_t)
surrogate_t = min(ratio_t * advantage_i,
                  clip(ratio_t, 1-epsilon_clip, 1+epsilon_clip) * advantage_i)
```

The clipped objective limits incentives for large policy changes on reused
samples. It does not guarantee every probability ratio stays inside the clip
interval. Unlike RSFT, unsuccessful model actions can receive a negative
advantage rather than being excluded from the objective.

The measured run used **12 groups of four**, two inner updates for each
nonconstant group, and KL coefficient **0**. Four groups had equal rewards,
so they supplied zero relative signal and skipped optimization: **16 actual
updates**, not the 24-update upper bound. Old probabilities stayed fixed
through both updates. Loss was averaged over model tokens within a trajectory,
then across trajectories, following the original trajectory-mean form.

Collection used **temperature 2 with full support**, and replay calculated
likelihoods under the same temperature policy. Tool observations stayed in
context but were not model action targets. The implementation is custom; the
TRL documentation is a reference, not an implementation claim.

**Try:** compute the signs of advantages for `[0, 0, 1, 1]`. Then try
`[1, 1, 1, 1]`. Follow `group_advantages` and `grpo_loss` in
[`grpo.py`](../src/qwen_posttrain/grpo.py), then the update loop in
[`grpo_training.py`](../src/qwen_posttrain/grpo_training.py).

**Read:** [DeepSeekMath §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1)
and [TRL's GRPO guide](https://huggingface.co/docs/trl/grpo_trainer).

## 9. Evaluate and interpret the charts

The measured fresh cohorts were **0/100 → 100/100 → 100/100** for base,
RSFT and RSFT-plus-GRPO. Both trained stages reached the observed score ceiling.
GRPO maintained that score; it did not show a further pass-rate increase.

A training-loss curve asks how probable supervised targets became under
teacher forcing. A GRPO reward curve describes attempts sampled during online
training. A fresh benchmark cohort tests the final policy under fixed
evaluation conditions. These are different measurements.

Wilson intervals retain uncertainty at zero or all successes: 0/100 gives
approximately **0–3.70%**, and 100/100 gives **96.30–100%**. These intervals
describe stochastic behavior on this fixed task, not performance on unseen
tasks or variability between independently trained adapters.

**Try:** run `qwen-posttrain verify-results --help`, then verify the published
aggregate files. Explain why fresh seeds cannot establish transfer. Identify
what a genuinely held-out task split would need to change.

**Read:** [NIST's proportion intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
and [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/2.7/notes/randomness.html).
Use the [reference map](references.md) when you need a deeper derivation.
