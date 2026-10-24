"""Pure-Python masking and provenance tests on synthetic tool trajectories."""

from __future__ import annotations

import copy
import json

import pytest

from qwen_posttrain import dataset
from qwen_posttrain.modeling import (
    QUANTIZATION_POLICY,
    canonical_json,
    fingerprint,
    parse_assistant_output,
    validate_training_example,
)


class CharacterTokenizer:
    chat_template = "synthetic-native-template"
    eos_token_id = 2
    specials = {"<|im_end|>": 2, "</think>": 3, "<think>": 4, "<tool_call>": 5, "</tool_call>": 6}

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            for token, identifier in self.specials.items():
                if text.startswith(token):
                    result.append(identifier)
                    text = text[len(token) :]
                    break
            else:
                result.append(ord(text[0]) + 100)
                text = text[1:]
        return result

    def decode(self, ids, **kwargs):
        reverse = {value: key for key, value in self.specials.items()}
        return "".join(reverse[value] if value in reverse else chr(value - 100) for value in ids)

    def apply_chat_template(
        self, messages, *, tools, tokenize, add_generation_prompt, enable_thinking
    ):
        text = canonical_json({"messages": messages, "tools": tools}) + "\nassistant\n"
        if not enable_thinking:
            text += "<think>\n\n</think>\n\n"
        return self.encode(text)


def example_source(tokenizer, *, seed=10):
    task = {
        "system_prompt": "Use the provided tools.",
        "user_prompt": "Compute the toy quantity.",
        "tools": [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("write_answer", "submit")
        ],
    }
    config = {"adapter": None, "thinking": True, "quantization": QUANTIZATION_POLICY}
    record = {
        "schema_version": 1,
        "run_id": f"collection-{seed}",
        "seed": seed,
        "phase": "collection",
        "config": config,
        "config_hash": fingerprint(config),
        "task_hash": "synthetic-source",
        "evaluator_hash": "synthetic-evaluator",
        "metadata": {
            "base_model_fingerprint": "synthetic-base",
            "quantization_policy": QUANTIZATION_POLICY,
            "chat_template_sha256": fingerprint(tokenizer.chat_template),
        },
        "success": True,
        "score": {"strict_pass": True},
        "submitted": True,
        "status": "submitted",
        "steps": [],
    }
    messages = [
        {"role": "system", "content": "Collector workflow."},
        {"role": "user", "content": "Collector-only hint."},
    ]
    for index, (name, arguments, result) in enumerate(
        (
            ("write_answer", {"content": "toy_output=7"}, {"written": True}),
            ("submit", {}, {"submitted": True}),
        )
    ):
        prompt = dataset.render_prefix(tokenizer, messages, task["tools"], thinking=True)
        raw = (
            "<think>Synthetic reasoning.</think>\n\n<tool_call>"
            + json.dumps({"name": name, "arguments": arguments})
            + "</tool_call><|im_end|>"
        )
        parsed = parse_assistant_output(raw, thinking=True, step=index)
        record["steps"].append(
            {
                "step_index": index,
                "prompt_token_ids": prompt,
                "generated_token_ids": tokenizer.encode(raw),
                "raw_text": raw,
                "error": None,
                "parse_error": None,
                "assistant_message": parsed["message"],
                "tool_results": [{"name": name, "arguments": arguments, "result": result}],
            }
        )
        messages.append(copy.deepcopy(parsed["message"]))
        messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": parsed["calls"][0]["id"],
                "content": canonical_json(result),
            }
        )
    record["messages"] = messages
    return task, record


def test_action_suffix_is_literal_source_tokens_with_original_eos():
    tokenizer = CharacterTokenizer()
    _, source = example_source(tokenizer)
    step = source["steps"][0]
    ids, text, provenance = dataset.action_suffix(step, tokenizer, {2})
    start, end = provenance["source_generated_token_range"]
    assert ids == step["generated_token_ids"][start:end]
    assert tokenizer.decode(ids) == text
    assert ids[-1] == 2 and "Synthetic reasoning" not in text
    assert provenance["text_normalization"]["retokenized"] is False


def test_action_student_prefix_removes_guidance_and_reasoning_and_masks_prior_actions():
    tokenizer = CharacterTokenizer()
    task, source = example_source(tokenizer)
    examples, reason = dataset.action_examples(
        source,
        tokenizer=tokenizer,
        task=task,
        student_task_hash="student-task",
        student_config_hash="student-policy",
        max_length=10000,
        eos_ids={2},
    )
    assert reason is None and len(examples) == 2
    for example in examples:
        validate_training_example(example, 10000)
        prompt = tokenizer.decode(example["input_ids"][: example["prompt_length"]])
        assert "Collector-only hint" not in prompt and "Synthetic reasoning" not in prompt
        assert example["labels"][: example["prompt_length"]] == [-100] * example["prompt_length"]
    assert "toy_output=7" in tokenizer.decode(
        examples[1]["input_ids"][: examples[1]["prompt_length"]]
    )


@pytest.mark.parametrize("mutation", ["tokens", "prompt", "history", "eos", "strict"])
def test_action_rejects_unverifiable_or_unsuccessful_source(mutation):
    tokenizer = CharacterTokenizer()
    task, source = example_source(tokenizer)
    if mutation == "tokens":
        source["steps"][0]["generated_token_ids"][0] = 200
    if mutation == "prompt":
        source["steps"][0]["prompt_token_ids"][0] = 200
    if mutation == "history":
        source["messages"][3]["content"] = '{"changed":true}'
    if mutation == "eos":
        source["steps"][0]["generated_token_ids"].pop()
    if mutation == "strict":
        source["score"]["strict_pass"] = False
    examples, reason = dataset.action_examples(
        source,
        tokenizer=tokenizer,
        task=task,
        student_task_hash="task",
        student_config_hash="policy",
        max_length=10000,
        eos_ids={2},
    )
    assert examples == [] and reason


def test_builder_deduplicates_exact_targets_and_records_real_counts(tmp_path):
    tokenizer = CharacterTokenizer()
    _, source = example_source(tokenizer)
    duplicate = copy.deepcopy(source)
    duplicate.update(run_id="duplicate", seed=11)
    failed = copy.deepcopy(source)
    failed.update(run_id="failed", seed=12)
    failed["score"]["strict_pass"] = False
    path = tmp_path / "collection.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in (source, duplicate, failed)))
    result = dataset.build_dataset(path, tmp_path / "dataset.jsonl", max_length=10000)
    assert result["pool_attempts"] == 3 and result["pool_successes"] == 2
    assert result["accepted_trajectories"] == 1 and result["examples"] == 2
    assert result["source_evaluator_hash"] == "synthetic-evaluator"
    assert result["invariants"]["evaluator_hash"] == "synthetic-evaluator"
    assert result["rejections"] == {
        "duplicate_success_trajectory": 1,
        "not_successful_submission": 1,
    }
    rows = list(dataset.iter_jsonl(tmp_path / "dataset.jsonl"))
    assert result["supervised_tokens"] == sum(row["completion_length"] for row in rows)
    for row in rows:
        validate_training_example(row, 10000)


def test_oversized_trajectory_is_rejected_whole_without_output(tmp_path):
    tokenizer = CharacterTokenizer()
    _, source = example_source(tokenizer)
    path = tmp_path / "collection.jsonl"
    path.write_text(json.dumps(source) + "\n")
    with pytest.raises(ValueError, match="No usable own successes"):
        dataset.build_dataset(path, tmp_path / "dataset.jsonl", max_length=10)
    assert not (tmp_path / "dataset.jsonl").exists()


def test_rsft_config_cli_flags_override_json_without_ml(tmp_path, monkeypatch):
    from qwen_posttrain import rsft

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"dataset": "sample.jsonl", "output": "adapter", "max_steps": 2}))
    captured = []
    monkeypatch.setattr(rsft, "train_rsft", captured.append)
    rsft.main(["--config", str(path), "--max-steps", "3"])
    assert captured[0].max_steps == 3 and str(captured[0].dataset) == "sample.jsonl"


def test_group_advantages_work_without_optional_ml_dependencies():
    from qwen_posttrain.grpo import group_advantages

    assert group_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    advantages = group_advantages([0.0, 1.0])
    assert abs(sum(advantages)) < 1e-12
    assert advantages[0] < 0 < advantages[1]


def test_training_module_imports_are_lazy_in_a_fresh_interpreter():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = """
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','peft','accelerate','bitsandbytes'}:
        raise RuntimeError('Unexpected eager ML dependency: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import qwen_posttrain.modeling
import qwen_posttrain.losses
import qwen_posttrain.dataset
import qwen_posttrain.rsft
import qwen_posttrain.grpo
import qwen_posttrain.grpo_training
"""
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_invalid_grpo_policy_fails_before_loading_any_model(tmp_path):
    from qwen_posttrain.grpo_training import train_grpo

    config = {
        "groups": 1,
        "group_size": 2,
        "iterations_per_group": 2,
        "learning_rate": 1e-5,
        "generation": {
            "model": "synthetic",
            "adapter": "synthetic-adapter",
            "top_p": 0.8,
            "top_k": 20,
        },
        "output_adapter": str(tmp_path / "adapter"),
        "evidence_dir": str(tmp_path / "evidence"),
    }
    with pytest.raises(ValueError, match="full-support"):
        train_grpo(config, object())
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("value", [None, "", "   ", "another-evaluator"])
def test_collection_cannot_mix_or_omit_evaluator_identity(tmp_path, value):
    tokenizer = CharacterTokenizer()
    _, source = example_source(tokenizer)
    second = copy.deepcopy(source)
    second.update(run_id="second", seed=11, evaluator_hash=value)
    path = tmp_path / "collection.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in (source, second)))
    with pytest.raises(ValueError, match="evaluator_hash"):
        dataset.build_dataset(path, tmp_path / "dataset.jsonl", max_length=10000)
    assert not (tmp_path / "dataset.jsonl").exists()


def test_task_provenance_distinguishes_changed_gold_from_unchanged_prompt():
    from qwen_posttrain.environment import TaskSpec
    from qwen_posttrain.grpo_training import task_provenance

    task = TaskSpec(
        task_id="toy",
        system_prompt="Use tools.",
        user_prompt="Return a toy value.",
        tools=[],
        scoring={"kind": "exact_text", "expected": "first"},
    )
    changed = TaskSpec(
        task_id="toy",
        system_prompt=task.system_prompt,
        user_prompt=task.user_prompt,
        tools=[],
        scoring={"kind": "exact_text", "expected": "second"},
    )
    assert task.fingerprint() == changed.fingerprint()
    assert task.evaluator_hash != changed.evaluator_hash
    actual = task_provenance({}, task)
    assert actual == {"task_hash": task.fingerprint(), "evaluator_hash": task.evaluator_hash}
    with pytest.raises(ValueError, match="evaluator_hash"):
        task_provenance(actual, changed)
    assert "first" not in json.dumps(actual)


def test_declared_attention_backend_reaches_the_real_loader_interface():
    from qwen_posttrain.grpo_training import load_grpo_base_model
    from qwen_posttrain.modeling import GenerationConfig

    calls = []

    class RecordingFactory:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append((model, kwargs))
            return "loaded-policy"

    generation = GenerationConfig(model="synthetic-model", attention="eager", revision="abc")
    loaded = load_grpo_base_model(
        generation,
        model_factory=RecordingFactory,
        quantization_config="quantizer",
        compute_dtype="bf16",
    )
    assert loaded == "loaded-policy"
    assert calls[0][1]["attn_implementation"] == "eager"
    assert calls[0][0] == generation.model and calls[0][1]["revision"] == generation.revision


def test_implicit_input_preloading_is_explicitly_rejected_before_model_load():
    from qwen_posttrain.modeling import GenerationConfig

    GenerationConfig(preload_inputs=False).validate()
    with pytest.raises(ValueError, match="read_input"):
        GenerationConfig(preload_inputs=True).validate()
