"""Online multi-turn QLoRA GRPO against a caller-supplied environment.

This explicit beta-zero implementation preserves full-support temperature
sampling, fixed behavior log probabilities over inner iterations, trajectory-
then-group normalization, and complete skips for constant-reward groups.
No private task, prompt, label, target, or reference solution is bundled here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

from .grpo import group_advantages, grpo_loss, log_probs_for_step
from .modeling import (
    GenerationConfig,
    QwenGenerator,
    base_model_identity,
    file_sha256,
    fingerprint,
    package_versions,
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_record(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def task_provenance(configuration: dict, task) -> dict:
    """Bind visible inputs separately from the evaluator, without revealing it."""
    task_hash = fingerprint(task.agent_view())
    evaluator_hash = task.evaluator_hash
    if not isinstance(evaluator_hash, str) or not evaluator_hash.strip():
        raise ValueError("A nonempty actual evaluator hash is required")
    for field, actual in (("task_hash", task_hash), ("evaluator_hash", evaluator_hash)):
        if configuration.get(field) and configuration[field] != actual:
            raise ValueError(f"{field} changed after declaration")
    return {"task_hash": task_hash, "evaluator_hash": evaluator_hash}


def load_grpo_base_model(generation, *, model_factory, quantization_config, compute_dtype):
    """Forward the declared attention backend to the actual model loader."""
    return model_factory.from_pretrained(
        generation.model,
        revision=generation.revision,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation=generation.attention,
    )


def train_grpo(
    configuration: dict, task, *, config_path: str | Path | None = None, rollout_runner=None
) -> dict:
    """Continue an existing LoRA adapter on original environment rewards.

    No fallback or success-based stopping changes the declared group budget.
    Timing starts after model/tokenizer loading and includes rollout generation,
    old-policy scoring, optimization, rolling checkpoints and final saves.
    An interrupted run is preserved and cannot silently restart over its files.
    """
    config = dict(configuration)
    defaults = {
        "beta": 0.0,
        "weight_decay": 0.0,
        "clip_epsilon": 0.2,
        "max_grad_norm": 1.0,
        "optimizer_seed": 42,
        "iterations_per_group": 2,
    }
    config = {**defaults, **config}
    if (
        config.get("groups", 0) < 1
        or config.get("group_size", 0) < 2
        or config["iterations_per_group"] < 1
    ):
        raise ValueError("Positive groups/iterations and a group size of at least two are required")
    if config.get("learning_rate", 0) <= 0 or not 0 < config["clip_epsilon"] < 1:
        raise ValueError("Positive learning rate and clipping epsilon in (0,1) are required")
    if config["beta"] != 0 or config["weight_decay"] != 0:
        raise ValueError("This implementation declares beta=0 and zero weight decay")
    generation = GenerationConfig(**config["generation"])
    generation.validate()
    if (
        generation.temperature <= 0
        or generation.top_p != 1
        or generation.top_k != 0
        or not generation.adapter
    ):
        raise ValueError(
            "GRPO requires an initial adapter and full-support positive-temperature sampling"
        )
    provenance = task_provenance(config, task)
    task_hash, evaluator_hash = provenance["task_hash"], provenance["evaluator_hash"]
    if config_path and json.loads(Path(config_path).read_text()) != configuration:
        raise ValueError("Supplied GRPO configuration differs from its recorded file")
    config_sha256 = file_sha256(config_path) if config_path else fingerprint(config)
    if rollout_runner is None:
        from .rollouts import run_attempt

        rollout_runner = run_attempt

    def set_phase(value):
        if config.get("phase_file"):
            phase_path = Path(config["phase_file"])
            phase_path.parent.mkdir(parents=True, exist_ok=True)
            phase_path.write_text(value, encoding="utf-8")

    root = Path(config["evidence_dir"])
    output = Path(config["output_adapter"])
    if (root / "training_started.json").exists() or output.exists():
        raise FileExistsError(
            "Refuse to overwrite or silently restart GRPO; preserve evidence and declare a new run"
        )
    root.mkdir(parents=True, exist_ok=True)
    initial = Path(config["generation"]["adapter"])
    initial_adapter_sha256 = file_sha256(initial / "adapter_model.safetensors")
    if (
        config.get("initial_adapter_sha256")
        and initial_adapter_sha256 != config["initial_adapter_sha256"]
    ):
        raise ValueError("Initial adapter changed after declaration")
    save_json(
        root / "training_started.json",
        {
            "started_at": now(),
            "config_sha256": config_sha256,
            "configuration": config,
            "evaluator_hash": evaluator_hash,
            "packages": package_versions(),
        },
    )
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("QLoRA GRPO requires a BF16-capable CUDA GPU")
    set_seed(config["optimizer_seed"])
    identity = base_model_identity(
        config["generation"]["model"], config["generation"].get("revision")
    )
    if identity["fingerprint"] != config.get("base_model_fingerprint", identity["fingerprint"]):
        raise ValueError("Base model changed after declaration")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = load_grpo_base_model(
        generation,
        model_factory=AutoModelForCausalLM,
        quantization_config=quant,
        compute_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(model, initial, is_trainable=True)
    if any(p.requires_grad for name, p in model.named_parameters() if "lora_" not in name):
        raise AssertionError("Only LoRA may be updated")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("The initial adapter contains no trainable LoRA parameters")
    if any(module.p > 0 for module in model.modules() if isinstance(module, torch.nn.Dropout)):
        raise ValueError("Likelihood replay requires zero model/adapter dropout")
    tokenizer = AutoTokenizer.from_pretrained(
        config["generation"]["model"], revision=config["generation"].get("revision")
    )
    tokenizer.pad_token = tokenizer.eos_token

    class AutocastGenerator(QwenGenerator):
        def generate(self, *values, **kwargs):
            # Match the arithmetic policy of old/current likelihood replay.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return super().generate(*values, **kwargs)

    generator = AutocastGenerator.__new__(AutocastGenerator)
    generator.config = GenerationConfig(**config["generation"])
    generator.config.validate()
    generator.model, generator.tokenizer, generator.torch = model, tokenizer, torch
    generator.metadata = {
        "packages": package_versions(),
        "base_model_fingerprint": identity["fingerprint"],
        "evaluator_hash": evaluator_hash,
        "quantization_policy": generator.config.quantization,
        "chat_template_sha256": fingerprint(tokenizer.chat_template),
        "initial_adapter_sha256": initial_adapter_sha256,
        "policy": "online-grpo-full-support-temperature",
        "config_sha256": config_sha256,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
    }
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    update_count = 0
    zero_variance_groups = 0
    updates = []
    groups = []
    for group_index in range(config["groups"]):
        set_phase("grpo_rollout")
        model.eval()
        model.config.use_cache = True
        rows = []
        for member in range(config["group_size"]):
            seed = config["start_seed"] + group_index * config["group_size"] + member
            generator.metadata = {
                **generator.metadata,
                "group": group_index + 1,
                "optimizer_updates": update_count,
            }
            row = rollout_runner(generator, task, seed=seed, phase="grpo_train")
            row["group"] = group_index + 1
            row["optimizer_updates_at_generation"] = update_count
            append_record(root / "train_rollouts.jsonl", row)
            if row["status"] == "runtime_error":
                raise RuntimeError(row["error"])
            if row.get("task_hash") != task_hash or row.get("evaluator_hash") != evaluator_hash:
                raise ValueError("GRPO rollout changed the declared task or evaluator")
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "rollout",
                        "group": group_index + 1,
                        "seed": seed,
                        "reward": row["score"]["reward"],
                        "success": row["success"],
                        "tokens": row["generated_tokens"],
                        "status": row["status"],
                    }
                ),
                flush=True,
            )
        rewards = [float(row["score"]["reward"]) for row in rows]
        advantages = group_advantages(rewards)
        group = {
            "group": group_index + 1,
            "rewards": rewards,
            "advantages": advantages,
            "mean_reward": sum(rewards) / len(rewards),
            "strict_pass_rate": sum(r["success"] for r in rows) / len(rows),
            "seeds": [r["seed"] for r in rows],
            "generated_tokens": sum(r["generated_tokens"] for r in rows),
            "optimizer_updates_before": update_count,
            "zero_variance": not any(advantages),
        }
        if group["zero_variance"]:
            zero_variance_groups += 1
            group["optimizer_updates_after"] = update_count
            append_record(root / "reward_curve.jsonl", group)
            groups.append(group)
            print(json.dumps({"event": "group", **group}), flush=True)
            continue
        if any(not row["generated_tokens"] for row in rows):
            raise ValueError("Nonconstant reward group contains no sampled actions")
        set_phase("grpo_update")
        model.config.use_cache = False
        # The behavior policy is fixed for this entire group, including mu>1.
        old_logps = []
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                old_logps.append(
                    [
                        log_probs_for_step(
                            model,
                            s["prompt_token_ids"],
                            s["generated_token_ids"],
                            generator.config.temperature,
                        )
                        .detach()
                        .cpu()
                        for s in row["steps"]
                        if s["generated_token_ids"]
                    ]
                )
        append_record(
            root / "old_policy_logprobs.jsonl",
            {
                "group": group_index + 1,
                "optimizer_updates": update_count,
                "temperature": generator.config.temperature,
                "seeds": [r["seed"] for r in rows],
                "per_episode_per_step_logprobs": [
                    [v.tolist() for v in episode] for episode in old_logps
                ],
            },
        )
        for iteration in range(config["iterations_per_group"]):
            optimizer.zero_grad(set_to_none=True)
            model.train()
            prior = [p.detach().clone() for p in parameters]
            loss_value, clip_tokens, total_tokens = 0.0, 0, 0
            for row, advantage, old_steps in zip(rows, advantages, old_logps):
                token_total = row["generated_tokens"]
                sampled_steps = [s for s in row["steps"] if s["generated_token_ids"]]
                assert sum(len(s["generated_token_ids"]) for s in sampled_steps) == token_total
                for step, old in zip(sampled_steps, old_steps):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        new = log_probs_for_step(
                            model,
                            step["prompt_token_ids"],
                            step["generated_token_ids"],
                            generator.config.temperature,
                        )
                        old = old.to(new.device)
                        ratio = (new.detach() - old).exp()
                        clip_tokens += int(
                            (
                                (ratio < 1 - config["clip_epsilon"])
                                | (ratio > 1 + config["clip_epsilon"])
                            ).sum()
                        )
                        total_tokens += new.numel()
                        # Original GRPO: mean within each trajectory, then mean across G.
                        contribution = (
                            grpo_loss(new, old, advantage, config["clip_epsilon"])
                            * new.numel()
                            / token_total
                            / len(rows)
                        )
                    if not torch.isfinite(contribution):
                        raise ValueError("Nonfinite GRPO objective")
                    contribution.backward()
                    loss_value += float(contribution.detach())
                    del contribution, new, old, ratio
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"])
            if not torch.isfinite(grad_norm):
                raise ValueError("Nonfinite GRPO gradient")
            optimizer.step()
            delta_sq = sum(
                float((p.detach().float() - before.float()).square().sum())
                for p, before in zip(parameters, prior)
            )
            del prior
            update_count += 1
            update = {
                "optimizer_update": update_count,
                "group": group_index + 1,
                "iteration": iteration + 1,
                "loss": loss_value,
                "gradient_norm_before_clip": float(grad_norm),
                "parameter_delta_l2": delta_sq**0.5,
                "clip_fraction": clip_tokens / total_tokens,
                "model_token_presentations": total_tokens,
                "mean_rollout_reward": group["mean_reward"],
                "timestamp": now(),
            }
            updates.append(update)
            append_record(root / "updates.jsonl", update)
            print(json.dumps({"event": "update", **update}), flush=True)
        group["optimizer_updates_after"] = update_count
        append_record(root / "reward_curve.jsonl", group)
        groups.append(group)
        print(json.dumps({"event": "group", **group}), flush=True)
        checkpoint = output.parent / "grpo_checkpoint_latest"
        model.save_pretrained(checkpoint)
        save_json(
            checkpoint / "progress.json",
            {"completed_group": group_index + 1, "optimizer_updates": update_count},
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "config_sha256": config_sha256,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            checkpoint / "optimizer_state.pt",
        )
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    summary = {
        "status": "completed",
        "method": "online_multi_turn_original_grpo_qlora",
        "started_at": json.loads((root / "training_started.json").read_text())["started_at"],
        "finished_at": now(),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": config_sha256,
        "configuration": config,
        "base_model_fingerprint": identity["fingerprint"],
        "task_hash": task_hash,
        "evaluator_hash": evaluator_hash,
        "initial_adapter_sha256": initial_adapter_sha256,
        "final_adapter_path": str(output),
        "final_adapter_sha256": file_sha256(output / "adapter_model.safetensors"),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "actual_optimizer_updates": update_count,
        "zero_variance_groups": zero_variance_groups,
        "groups_completed": len(groups),
        "rollouts_completed": len(groups) * config["group_size"],
        "generated_tokens": sum(g["generated_tokens"] for g in groups),
        "model_token_presentations": sum(u["model_token_presentations"] for u in updates),
        "wall_seconds": time.monotonic() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "reward_curve_path": str(root / "reward_curve.jsonl"),
        "reward_curve_sha256": file_sha256(root / "reward_curve.jsonl"),
        "updates_path": str(root / "updates.jsonl"),
        "train_rollouts_path": str(root / "train_rollouts.jsonl"),
        "train_rollouts_sha256": file_sha256(root / "train_rollouts.jsonl"),
        "packages": package_versions(),
        "gpu": torch.cuda.get_device_name(0),
        "objective": {
            "beta": 0,
            "clip_epsilon": config["clip_epsilon"],
            "normalization": "mean_tokens_per_trajectory_then_mean_group",
            "mask": "all_sampled_model_tokens_including_reasoning_failures_eos; prompts_tools_excluded",
            "behavior_policy": "softmax_logits_div_temperature_full_support",
            "reference_model": None,
            "equal_reward_groups": "skip_optimizer_entirely",
            "old_policy": "fixed_logprobs_per_group_across_inner_iterations",
        },
    }
    save_json(root / "training_summary.json", summary)
    set_phase("grpo_training_complete")
    print(
        json.dumps(
            {
                "event": "training_complete",
                "updates": update_count,
                "seconds": summary["wall_seconds"],
            }
        ),
        flush=True,
    )

    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", type=Path, help="Environment JSON; otherwise configuration.task")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    from .environment import load_task

    task_path = args.task or config.get("task")
    if task_path is None:
        parser.error("--task or configuration.task is required")
    train_grpo(config, load_task(Path(task_path)), config_path=args.config)


if __name__ == "__main__":
    main()
