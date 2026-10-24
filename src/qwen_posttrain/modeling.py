"""Qwen3 generation and exact, assistant-only rejection-sampling examples.

Heavy ML imports are deliberately lazy: inspecting logs and building training data
does not allocate a GPU, download weights, or require PyTorch.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
QUANTIZATION_POLICY = "nf4-double-bf16"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_model_identity(model: str, revision: str | None = None) -> dict[str, Any]:
    """Identify actual local weights, or an immutable remote repository commit.

    A tokenizer/template match cannot establish base-model identity. Local
    fingerprints cover weights, configuration, and tokenizer artifacts; moving
    an unchanged checkpoint does not change identity.
    """
    directory = Path(model)
    if directory.is_dir():
        config = directory / "config.json"
        weights = sorted(directory.glob("*.safetensors"))
        if not config.is_file() or not weights:
            raise ValueError("Local base model must contain config.json and safetensors weights")
        paths = [config, *weights]
        index = directory / "model.safetensors.index.json"
        if index.is_file():
            paths.append(index)
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "tokenizer.model",
            "chat_template.jinja",
        ):
            artifact = directory / name
            if artifact.is_file():
                paths.append(artifact)
        files = []
        for path in sorted(paths, key=lambda item: item.name):
            before = path.stat()
            digest = file_sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Base model artifact changed while hashing: {path}")
            files.append({"name": path.name, "bytes": after.st_size, "sha256": digest})
        identity = {"kind": "local-safetensors-v1", "files": files}
    else:
        if directory.exists():
            raise ValueError("The local base model must be a checkpoint directory")
        if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("Remote base models require an immutable 40-character commit revision")
        identity = {"kind": "remote-commit-v1", "repository": model, "revision": revision.lower()}
    return {**identity, "fingerprint": fingerprint(identity)}


def validate_resume_configuration(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject training resumes that change the data, model or optimization run."""
    for key in (
        "dataset_sha256",
        "base_model_fingerprint",
        "quantization_policy",
        "training_policy",
    ):
        if key not in saved or saved[key] != current.get(key):
            raise ValueError(f"Refusing incompatible training resume: {key} changed or is missing")
    for key in (
        "rank",
        "alpha",
        "dropout",
        "lr",
        "epochs",
        "max_steps",
        "accumulation",
        "seed",
        "max_length",
        "attention",
    ):
        if key not in saved.get("config", {}) or saved["config"][key] != current.get(
            "config", {}
        ).get(key):
            raise ValueError(
                f"Refusing incompatible training resume: config.{key} changed or is missing"
            )


def package_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    adapter: str | None = None
    max_context: int = 12288
    max_new_tokens: int = 2048
    max_total_new_tokens: int = 32768
    max_steps: int = 40
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    thinking: bool = True
    # Inputs are read through the explicit environment tool; implicit preload is unsupported.
    preload_inputs: bool = False
    attention: str = "sdpa"
    quantization: str = QUANTIZATION_POLICY

    def validate(self) -> None:
        if self.preload_inputs:
            raise ValueError(
                "preload_inputs=True is unsupported; task inputs are accessed through read_input"
            )
        if self.max_context < 128 or self.max_new_tokens < 1:
            raise ValueError("Context must be >=128 and max_new_tokens >=1")
        if self.max_new_tokens >= self.max_context:
            raise ValueError("max_new_tokens must be smaller than max_context")
        if self.max_steps < 1 or self.max_total_new_tokens < 1:
            raise ValueError("Generation budgets must be positive")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("Invalid sampling configuration")
        if self.quantization != QUANTIZATION_POLICY:
            raise ValueError(
                "The supported quantization policy is NF4 double quantization with BF16 compute"
            )

    @property
    def identifier(self) -> str:
        return fingerprint(asdict(self))


def parse_assistant_output(raw_text: str, *, thinking: bool, step: int = 0) -> dict[str, Any]:
    """Parse Qwen's native tool tags without executing arbitrary text.

    The raw text and token IDs remain the source of truth for training. Parsed
    messages are only the history used for the next generation.
    """
    text = raw_text
    for suffix in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    reasoning = ""
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip("\n")
    elif thinking:
        # An unfinished thought is not a valid tool action.
        return {
            "message": {"role": "assistant", "content": "", "reasoning_content": text},
            "calls": [],
            "error": "unterminated_thinking",
        }
    text = text.strip("\n")
    matches = list(TOOL_CALL_RE.finditer(text))
    if ("<tool_call>" in text or "</tool_call>" in text) and not matches:
        return {
            "message": {"role": "assistant", "content": text, "reasoning_content": reasoning},
            "calls": [],
            "error": "malformed_tool_call_tags",
        }
    calls = []
    try:
        for index, match in enumerate(matches):
            payload = json.loads(match.group(1))
            name = payload["name"]
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("Tool call requires a string name and an argument object")
            calls.append(
                {
                    "id": f"call_{step}_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "message": {"role": "assistant", "content": text, "reasoning_content": reasoning},
            "calls": [],
            "error": f"invalid_tool_call_json:{type(exc).__name__}",
        }
    content = TOOL_CALL_RE.sub("", text).strip("\n")
    # Any unmatched tags are malformed even if an earlier call was valid.
    if "<tool_call>" in content or "</tool_call>" in content:
        return {
            "message": {"role": "assistant", "content": text, "reasoning_content": reasoning},
            "calls": [],
            "error": "unmatched_tool_call_tags",
        }
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return {"message": message, "calls": calls, "error": None}


class QwenGenerator:
    """Single-GPU, batch-size-one HF generation with a fixed quantization policy."""

    def __init__(self, config: GenerationConfig):
        config.validate()
        identity = base_model_identity(config.model, config.revision)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.config = config
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("NF4/BF16 inference requires a CUDA GPU")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            revision=config.revision,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation=config.attention,
        )
        if config.adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, config.adapter, is_trainable=False)
        self.model.eval()
        self.model.config.use_cache = True
        self.metadata = {
            "packages": package_versions(),
            "base_model_fingerprint": identity["fingerprint"],
            "base_model_identity": identity,
            "quantization_policy": config.quantization,
            "model_commit": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_vocab_size": len(self.tokenizer),
            "chat_template_sha256": fingerprint(self.tokenizer.chat_template),
            "gpu": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "adapter_config_sha256": file_sha256(Path(config.adapter) / "adapter_config.json")
            if config.adapter
            else None,
            "adapter_model_sha256": file_sha256(Path(config.adapter) / "adapter_model.safetensors")
            if config.adapter
            else None,
        }

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)

    def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, remaining_budget: int
    ) -> dict[str, Any]:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.config.thinking,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_ids = encoded["input_ids"][0].tolist()
        room = self.config.max_context - len(prompt_ids)
        limit = min(self.config.max_new_tokens, remaining_budget, room)
        if limit < 1:
            return {
                "error": "context_limit" if room < 1 else "generation_budget",
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": [],
                "raw_text": "",
                "effective_max_new_tokens": 0,
            }
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        kwargs: dict[str, Any] = {
            "max_new_tokens": limit,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.model.generation_config.eos_token_id,
        }
        if self.config.temperature > 0:
            kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **kwargs)
        generated_ids = output[0, len(prompt_ids) :].tolist()
        return {
            "error": None,
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": generated_ids,
            "raw_text": self.tokenizer.decode(generated_ids, skip_special_tokens=False),
            "effective_max_new_tokens": limit,
            "hit_generation_limit": len(generated_ids) >= limit,
        }


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def training_examples_from_rollout(
    rollout: dict[str, Any],
    *,
    max_length: int,
    accepted_phases: tuple[str, ...] = ("collection", "train_pool"),
) -> tuple[list[dict[str, Any]], str | None]:
    """Accept only complete, submitted successes from a separate training pool.

    An entire trajectory is rejected if a turn does not fit. We never crop a
    prefix containing the system prompt or earlier tool turns, and we never
    infer assistant masks from decoded strings.
    """
    if rollout.get("phase") not in accepted_phases:
        return [], "not_training_pool"
    if rollout.get("score", {}).get("strict_pass") is not True or not rollout.get("submitted"):
        return [], "not_successful_submission"
    if rollout.get("status") != "submitted":
        return [], "incomplete_trajectory"
    if rollout.get("schema_version") != SCHEMA_VERSION:
        return [], "unsupported_schema"
    steps = rollout.get("steps", [])
    if not steps:
        return [], "no_assistant_turns"
    examples = []
    for step_index, step in enumerate(steps):
        prompt = step.get("prompt_token_ids")
        generated = step.get("generated_token_ids")
        if (
            not isinstance(prompt, list)
            or not isinstance(generated, list)
            or not prompt
            or not generated
        ):
            return [], "missing_exact_tokens"
        if not all(
            isinstance(token, int) and not isinstance(token, bool) and token >= 0
            for token in prompt + generated
        ):
            return [], "invalid_token_ids"
        if step.get("error") or step.get("parse_error"):
            return [], "failed_generation_or_parse"
        if len(prompt) + len(generated) > max_length:
            return [], "trajectory_exceeds_max_length"
        examples.append(
            {
                "input_ids": prompt + generated,
                "labels": [-100] * len(prompt) + generated,
                "attention_mask": [1] * (len(prompt) + len(generated)),
                "prompt_length": len(prompt),
                "completion_length": len(generated),
                "rollout_id": rollout["run_id"],
                "step_index": step_index,
                "seed": rollout["seed"],
                "config_hash": rollout["config_hash"],
                "task_hash": rollout["task_hash"],
                "chat_template_sha256": rollout["metadata"]["chat_template_sha256"],
            }
        )
    return examples, None


def validate_training_example(example: dict[str, Any], max_length: int) -> None:
    ids, labels = example["input_ids"], example["labels"]
    if not isinstance(ids, list) or not isinstance(labels, list):
        raise ValueError("Input IDs and labels must be token arrays")
    if not all(
        isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in ids
    ):
        raise ValueError("Invalid input token ID")
    prompt_length = example["prompt_length"]
    if (
        isinstance(prompt_length, bool)
        or not isinstance(prompt_length, int)
        or not 0 < prompt_length < len(ids)
    ):
        raise ValueError("A complete prompt and nonempty assistant completion are required")
    if not len(ids) == len(labels) == len(example["attention_mask"]):
        raise ValueError("Input, label and attention lengths differ")
    if not all(
        isinstance(label, int) and not isinstance(label, bool) and (label == -100 or label >= 0)
        for label in labels
    ):
        raise ValueError("Labels must be integer token IDs or the -100 mask")
    if example.get("completion_length", len(ids) - prompt_length) != len(ids) - prompt_length:
        raise ValueError("Completion length disagrees with exact token arrays")
    if len(ids) > max_length:
        raise ValueError("An example exceeds max_length; truncation is not permitted")
    if any(label != -100 for label in labels[:prompt_length]):
        raise ValueError("Prompt/tool tokens must be masked")
    if labels[prompt_length:] != ids[prompt_length:]:
        raise ValueError("Every generated assistant token must retain its exact label")
    if any(mask != 1 for mask in example["attention_mask"]):
        raise ValueError("Unpadded examples must have a complete attention mask")


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        length = max(len(item["input_ids"]) for item in features)
        result: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
        for item in features:
            padding = length - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            result["labels"].append(item["labels"] + [-100] * padding)
            result["attention_mask"].append(item["attention_mask"] + [0] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in result.items()}
