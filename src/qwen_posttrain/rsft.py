"""Single-device QLoRA supervised fine-tuning on exact own-success tokens.

The module and CLI help import without PyTorch. Model allocation happens only in
train_rsft, after data/manifest/base-model identity checks succeed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .losses import (
    CompletionOnlyLossMixin,
)
from .modeling import (
    QUANTIZATION_POLICY,
    AssistantOnlyCollator,
    base_model_identity,
    file_sha256,
    fingerprint,
    iter_jsonl,
    package_versions,
    validate_resume_configuration,
    validate_training_example,
)


@dataclass
class RSFTConfig:
    dataset: str | Path
    output: str | Path
    manifest: str | Path | None = None
    model: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    max_length: int = 8192
    epochs: float = 2
    max_steps: int = -1
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    lr: float = 1e-4
    accumulation: int = 4
    seed: int = 424242
    attention: str = "sdpa"
    resume_from_checkpoint: str | Path | None = None


def train_rsft(configuration: RSFTConfig | dict) -> dict:
    """Train once, or explicitly continue the same checkpoint/configuration.

    Returned wall time, allocator peaks and direct work counters cover only this
    invocation; the final optimizer step remains cumulative on checkpoint resume.
    """
    if isinstance(configuration, dict):
        configuration = RSFTConfig(**configuration)
    args = argparse.Namespace(**asdict(configuration))
    for key in ("dataset", "output", "manifest", "resume_from_checkpoint"):
        if getattr(args, key) is not None:
            setattr(args, key, Path(getattr(args, key)))
    if (
        args.epochs <= 0
        or args.rank <= 0
        or args.alpha <= 0
        or args.accumulation <= 0
        or args.lr <= 0
    ):
        raise ValueError("Training sizes, LR and epochs must be positive")
    manifest_path = args.manifest or args.dataset.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_sha256 = file_sha256(args.dataset)
    if manifest["dataset_sha256"] != dataset_sha256:
        raise ValueError("Training dataset hash differs from the accepted-source manifest")
    if manifest.get("invariants", {}).get("quantization_policy") != QUANTIZATION_POLICY:
        raise ValueError("Training pool quantization differs from this NF4 double/BF16 trainer")
    identity = base_model_identity(args.model, args.revision)
    if identity["fingerprint"] != manifest.get("invariants", {}).get("base_model_fingerprint"):
        raise ValueError(
            "Training base model weights/config differ from the own-success rollout base"
        )
    rows = list(iter_jsonl(args.dataset))
    if not rows:
        raise ValueError("Training dataset is empty")
    for row in rows:
        validate_training_example(row, args.max_length)
    if (args.output / "adapter_config.json").exists() and args.resume_from_checkpoint is None:
        raise FileExistsError("Completed adapter exists; use a new output or explicitly resume")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires a BF16-capable CUDA GPU")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if fingerprint(tokenizer.chat_template) != manifest["invariants"]["chat_template_sha256"]:
        raise ValueError("Training tokenizer template differs from the rollout template")
    if max(max(row["input_ids"]) for row in rows) >= len(tokenizer):
        raise ValueError("A training token ID is outside the tokenizer vocabulary")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=args.attention,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()

    class TokenDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            return {name: row[name] for name in ("input_ids", "labels", "attention_mask")}

    class DurableLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.is_world_process_zero:
                with (Path(args.output_dir) / "training_log.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps({"step": state.global_step, **logs}) + "\n")

    class CompletionOnlyTrainer(CompletionOnlyLossMixin, Trainer):
        pass

    args.output.mkdir(parents=True, exist_ok=True)
    public_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    metadata = {
        "config": public_config,
        "packages": package_versions(),
        "dataset_sha256": dataset_sha256,
        "training_manifest_sha256": file_sha256(manifest_path),
        "base_model_fingerprint": identity["fingerprint"],
        "base_model_identity": identity,
        "quantization_policy": QUANTIZATION_POLICY,
        "completion_only_logit_projection": True,
        "training_policy": {
            "target_modules": target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "optimizer": "adamw_torch",
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.05,
            "per_device_train_batch_size": 1,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "bf16": True,
            "tf32": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "logit_projection": "supervised_causal_positions_v1",
            "loss_normalization": "all_supervised_tokens_per_accumulation_window",
        },
        "trainable_parameters": trainable,
        "total_parameters": total,
        "examples": len(rows),
        "accepted_trajectories": manifest["accepted_trajectories"],
        "supervised_tokens_per_epoch": manifest["supervised_tokens"],
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "chat_template_sha256": fingerprint(tokenizer.chat_template),
    }
    configuration_path = args.output / "training_config.json"
    if args.resume_from_checkpoint:
        previous_path = configuration_path
        if not previous_path.is_file():
            previous_path = args.resume_from_checkpoint.parent / "training_config.json"
        if not previous_path.is_file():
            raise ValueError(
                "Resume requires the original training_config.json; refusing an unverifiable resume"
            )
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        validate_resume_configuration(previous, metadata)
    elif configuration_path.exists():
        raise FileExistsError(
            "Training configuration exists; use a new output or explicitly resume"
        )
    configuration_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    arguments = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_strategy="steps",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
    )
    trainer = CompletionOnlyTrainer(
        model=model,
        args=arguments,
        train_dataset=TokenDataset(),
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=[DurableLog()],
    )
    print(
        json.dumps(
            {
                "trainable_parameters": trainable,
                "total_parameters": total,
                "examples": len(rows),
                "supervised_tokens": manifest["supervised_tokens"],
            }
        ),
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    trainer.save_state()
    metadata.update(
        {
            "wall_seconds": time.monotonic() - started,
            "train_result": result.metrics,
            "log_history": trainer.state.log_history,
            "global_steps": trainer.state.global_step,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "adapter_model_sha256": file_sha256(args.output / "adapter_model.safetensors"),
            "observed_training_work": trainer.observed_training_work(),
        }
    )
    (args.output / "training_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "trained",
                "steps": trainer.state.global_step,
                "wall_seconds": metadata["wall_seconds"],
                "peak_gpu_allocated_bytes": metadata["peak_gpu_allocated_bytes"],
            }
        ),
        flush=True,
    )

    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="JSON object of RSFTConfig fields; explicit flags override it"
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument(
        "--attention", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa"
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args(argv)
    if known.config:
        configuration = json.loads(known.config.read_text(encoding="utf-8"))
        if not isinstance(configuration, dict) or set(configuration) - set(
            RSFTConfig.__dataclass_fields__
        ):
            parser.error("--config must contain only RSFTConfig fields")
        parser.set_defaults(**configuration)
    args = parser.parse_args(argv)
    if args.dataset is None or args.output is None:
        parser.error("--dataset and --output are required, either as flags or in --config")
    values = vars(args)
    values.pop("config")
    train_rsft(RSFTConfig(**values))


if __name__ == "__main__":
    main()
