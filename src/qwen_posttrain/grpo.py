"""Small, explicit GRPO math, independent of the environment's gold answers."""

from __future__ import annotations

import math


def group_advantages(rewards):
    if len(rewards) < 2 or not all(math.isfinite(float(x)) for x in rewards):
        raise ValueError("A finite reward group of at least two members is required")
    values = [float(x) for x in rewards]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    if std == 0:
        return [0.0] * len(values)
    return [(x - mean) / (std + 1e-8) for x in values]


def grpo_loss(new_logps, old_logps, advantage, clip_epsilon=0.2):
    import torch

    if new_logps.ndim != 1 or new_logps.shape != old_logps.shape or not new_logps.numel():
        raise ValueError("Log probabilities must be matching nonempty token vectors")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must lie in (0,1)")
    ratio = torch.exp(new_logps - old_logps.detach())
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage
    return -torch.minimum(unclipped, clipped).mean()


def log_probs_for_step(model, prompt_ids, generated_ids, temperature):
    """Exact sampled-token causal log probabilities; prompts have no RL targets.

    This is a full-support temperature policy. Callers must sample with top_p=1
    and top_k=0, and pass the same temperature used at generation time.
    Qwen's selected-position projection avoids a vocabulary matrix for prompts.
    """
    import torch

    if not prompt_ids or not generated_ids or temperature <= 0:
        raise ValueError("Nonempty prompt/completion and positive temperature required")
    device = model.device
    ids = torch.tensor([prompt_ids + generated_ids], dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_ids) - 1, ids.shape[1] - 1, device=device)
    outputs = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        logits_to_keep=positions,
        use_cache=False,
    )
    logits = outputs.logits[0].float() / temperature
    targets = torch.tensor(generated_ids, dtype=torch.long, device=device)
    # logsumexp form avoids retaining a second vocabulary-sized log-softmax.
    return logits.gather(1, targets[:, None]).squeeze(1) - logits.logsumexp(dim=-1)
