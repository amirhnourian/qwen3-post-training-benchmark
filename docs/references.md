# Reading references by concept

Start with the [tutorial](tutorial.md), then use this map for difficult concepts.
These are official documentation or original research. A citation supplies
background; it does not mean the project reproduces every part of that source.

| Concept | Read first | What to look for |
| --- | --- | --- |
| Qwen thinking and sampling | [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) | Thinking-mode controls, decoding recommendations and model loading |
| Roles, tools and token sequences | [Transformers chat templates, v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/chat_templating) | Rendering model-specific control tokens and generation prompts |
| Causal next-token prediction | [Hugging Face causal-language-modeling course](https://huggingface.co/learn/llm-course/en/chapter7/6) | Predicting continuations from previous tokens; its full example is not this training run |
| Masked supervised loss | [PyTorch CrossEntropyLoss, 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html) | Class-index targets and `ignore_index=-100` |
| Generate, filter, train | [Reinforced Self-Training](https://arxiv.org/abs/2308.08998) | Related generate–filter–train ideas; assisted action distillation here is separately specified |
| Low-rank adapters | [LoRA paper](https://arxiv.org/abs/2106.09685) | Frozen base weights and trainable low-rank matrix updates |
| Four-bit adapter training | [QLoRA paper](https://arxiv.org/abs/2305.14314) | NF4 and double quantization; storage format differs from computation precision |
| Practical quantized training | [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) | Preparing a quantized model for adapter training |
| Optimizer updates | [PyTorch AdamW, 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.optim.AdamW.html) | Momentum state and weight-decay behavior; this GRPO implementation uses zero weight decay |
| Activation memory | [PyTorch checkpointing, 2.7](https://docs.pytorch.org/docs/2.7/checkpoint.html) | Recompute activations; distinguish this from saving recovery state |
| Saving and recovery | [Transformers Trainer, v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/trainer) | Training configuration and checkpoint continuation |
| Wilson intervals | [NIST intervals for a proportion](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) | Uncertainty when zero or all attempts succeed |
| Two-rate differences | [statsmodels independent-proportion intervals](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html) | Newcombe difference interval; this package implements the calculation directly |
| GRPO foundation | [DeepSeekMath, §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1) | Group-relative advantages, clipping and original trajectory normalization |
| GRPO implementation variants | [TRL GRPO guide](https://huggingface.co/docs/trl/grpo_trainer) | KL coefficient, loss normalization and sampling settings; this project uses a custom trainer |
| Seeds and nondeterminism | [PyTorch reproducibility, 2.7](https://docs.pytorch.org/docs/2.7/notes/randomness.html) | Limits of reproducibility across versions, devices and operations |

## Advanced implementation reading

The measured stack used Transformers 4.57.6. Its version-family manuals above
are complemented by pinned source when exact behavior matters:

- [Qwen3 modeling source, v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py):
  search `logits_to_keep` for vocabulary-projection selection.
- [Causal loss source, v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/loss/loss_utils.py):
  search `ForCausalLMLoss`, `shift_labels` and `num_items_in_batch` for token
  alignment and loss normalization.
- [Trainer source, v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/trainer.py):
  search `get_batch_samples` and `resume_from_checkpoint` for accumulation and
  recovery details.

Live documentation can evolve. Use the pinned dependency groups and local
regression tests rather than assuming a newer example has the same behavior.
Reading a paper does not establish that the public synthetic demo reproduced
the private measured task or that a same-task gain transfers to new tasks.
