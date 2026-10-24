"""Build traceable own-success datasets without rewriting generated targets.

Full-trajectory examples supervise exact sampled tokens. Action distillation can
remove closed reasoning only at an existing token boundary, then reconstruct a
student prompt from a caller-supplied public or private environment specification.
No task, target answer, or reference solution is embedded in this module.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

from .modeling import (
    QUANTIZATION_POLICY,
    SCHEMA_VERSION,
    file_sha256,
    fingerprint,
    iter_jsonl,
    parse_assistant_output,
    training_examples_from_rollout,
    validate_training_example,
)


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def render_prefix(
    tokenizer, messages: list[dict], tools: list[dict], *, thinking: bool
) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
    )
    if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
        raise ValueError("Expected an unbatched exact token array from the chat template")
    return ids


def action_suffix(step: dict, tokenizer, eos_ids: set[int]) -> tuple[list[int], str, dict]:
    """Return a literal source-token suffix; reject a partial-token trim."""
    ids = step.get("generated_token_ids")
    raw = step.get("raw_text")
    if (
        not isinstance(ids, list)
        or not ids
        or not isinstance(raw, str)
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in ids
        )
    ):
        raise ValueError("missing_exact_source_generation")
    if decode(tokenizer, ids) != raw:
        raise ValueError("source_text_token_mismatch")
    if ids[-1] not in eos_ids:
        raise ValueError("source_action_missing_eos")
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    if not closing:
        raise ValueError("missing_thinking_delimiter_tokens")
    boundary = None
    for index in range(len(ids) - len(closing) + 1):
        if ids[index : index + len(closing)] == closing:
            boundary = index + len(closing)
            break
    if boundary is None or "</think>" not in raw:
        raise ValueError("source_thinking_not_closed")
    after_thought = raw.split("</think>", 1)[1]
    if decode(tokenizer, ids[boundary:]) != after_thought:
        raise ValueError("source_thinking_boundary_mismatch")
    target = after_thought.lstrip()
    removed = len(after_thought) - len(target)
    # Removing whitespace must never rewrite/re-tokenize an answer/action. Find
    # an exact original token boundary whose entire decoded suffix is target.
    start = None
    for index in range(boundary, len(ids)):
        prefix = decode(tokenizer, ids[boundary:index])
        if prefix and not prefix.isspace():
            break
        if len(prefix) > removed:
            break
        if len(prefix) == removed and decode(tokenizer, ids[index:]) == target:
            start = index
            break
    if start is None:
        raise ValueError("leading_whitespace_not_token_aligned")
    target_ids = ids[start:]
    if (
        not target
        or not target_ids
        or not raw.endswith(target)
        or decode(tokenizer, target_ids) != target
    ):
        raise ValueError("target_is_not_exact_source_suffix")
    parsed = parse_assistant_output(target, thinking=False, step=step["step_index"])
    if parsed["error"] or not parsed["calls"]:
        raise ValueError("source_suffix_has_no_valid_action")
    provenance = {
        "source_generated_token_range": [start, len(ids)],
        "source_generated_token_ids_sha256": fingerprint(ids),
        "source_action_token_ids_sha256": fingerprint(target_ids),
        "source_action_text_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
        "source_action_text": target,
        "text_normalization": {
            "removed_thinking": True,
            "leading_whitespace_characters_removed": removed,
            "retokenized": False,
        },
    }
    return target_ids, target, provenance


def action_examples(
    record: dict,
    *,
    tokenizer,
    task: dict,
    student_task_hash: str,
    student_config_hash: str,
    max_length: int,
    eos_ids: set[int],
) -> tuple[list[dict], str | None]:
    if (
        record.get("success") is not True
        or record.get("score", {}).get("strict_pass") is not True
        or record.get("submitted") is not True
    ):
        return [], "not_submitted_strict_success"
    if record.get("status") != "submitted" or record.get("schema_version") != SCHEMA_VERSION:
        return [], "incomplete_or_unsupported_trajectory"
    if record.get("config", {}).get("thinking") is not True:
        return [], "source_is_not_thinking_collection"
    source_messages = record.get("messages", [])
    if (
        len(source_messages) < 3
        or source_messages[0].get("role") != "system"
        or source_messages[1].get("role") != "user"
    ):
        return [], "invalid_source_initial_messages"
    history = [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": task["user_prompt"]},
    ]
    allowed_tools = {tool["function"]["name"] for tool in task["tools"]}
    cursor = 2
    examples = []
    try:
        for index, step in enumerate(record.get("steps", [])):
            if step.get("step_index") != index or step.get("error") or step.get("parse_error"):
                raise ValueError("source_generation_or_parse_error")
            if cursor >= len(source_messages) or source_messages[cursor].get("role") != "assistant":
                raise ValueError("source_history_step_mismatch")
            assistant = source_messages[cursor]
            if assistant != step.get("assistant_message"):
                raise ValueError("source_assistant_message_mismatch")
            parsed = parse_assistant_output(step["raw_text"], thinking=True, step=index)
            if parsed["error"] or parsed["message"] != assistant:
                raise ValueError("source_action_parse_mismatch")
            if any(call["function"]["name"] not in allowed_tools for call in parsed["calls"]):
                raise ValueError("source_tool_unavailable_to_student")
            expected_source = render_prefix(
                tokenizer, source_messages[:cursor], task["tools"], thinking=True
            )
            if expected_source != step.get("prompt_token_ids"):
                raise ValueError("source_prompt_token_mismatch")
            prompt = render_prefix(tokenizer, history, task["tools"], thinking=False)
            target_ids, _, provenance = action_suffix(step, tokenizer, eos_ids)
            if len(prompt) + len(target_ids) > max_length:
                raise ValueError("trajectory_exceeds_student_max_length")
            example = {
                "input_ids": prompt + target_ids,
                "labels": [-100] * len(prompt) + target_ids,
                "attention_mask": [1] * (len(prompt) + len(target_ids)),
                "prompt_length": len(prompt),
                "completion_length": len(target_ids),
                "rollout_id": record["run_id"],
                "step_index": index,
                "seed": record["seed"],
                "config_hash": student_config_hash,
                "task_hash": student_task_hash,
                "chat_template_sha256": fingerprint(tokenizer.chat_template),
                "source_config_hash": record["config_hash"],
                "source_task_hash": record["task_hash"],
                **provenance,
            }
            validate_training_example(example, max_length)
            examples.append(example)
            action_history = copy.deepcopy(assistant)
            action_history.pop("reasoning_content", None)
            history.append(action_history)
            cursor += 1
            for result in step.get("tool_results", []):
                if cursor >= len(source_messages):
                    raise ValueError("missing_source_tool_response")
                response = source_messages[cursor]
                if (
                    response.get("role") != "tool"
                    or response.get("name") != result["name"]
                    or json.loads(response["content"]) != result["result"]
                ):
                    raise ValueError("source_tool_response_mismatch")
                history.append(copy.deepcopy(response))
                cursor += 1
        if not examples or cursor != len(source_messages):
            raise ValueError("source_history_has_unmatched_messages")
    except (KeyError, TypeError, ValueError) as exc:
        return [], str(exc)
    return examples, None


def build_dataset(
    source: str | Path,
    output: str | Path,
    manifest_path: str | Path | None = None,
    *,
    max_length: int = 8192,
    action_only: bool = False,
    student_task=None,
    student_generation: dict | None = None,
    model: str | None = None,
    revision: str | None = None,
    tokenizer=None,
    eos_ids: set[int] | None = None,
    accepted_phases: tuple[str, ...] = ("collection", "train_pool"),
    assistance_disclosure: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Filter complete strict successes, deduplicate targets, and preserve hashes.

    ``student_task`` is a TaskSpec or its agent-view mapping. Action distillation
    requires it and a frozen non-thinking student generation configuration.
    The source tokenizer/base fingerprints must agree throughout the pool.
    """
    source, output = Path(source), Path(output)
    manifest_path = Path(manifest_path) if manifest_path else output.with_suffix(".manifest.json")
    if max_length < 2 or not accepted_phases:
        raise ValueError("A positive context bound and explicit collection phases are required")
    if len({source.resolve(), output.resolve(), manifest_path.resolve()}) != 3:
        raise ValueError("Source, dataset and manifest must be distinct files")
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError("Refuse to overwrite dataset or manifest")
    records = list(iter_jsonl(source))
    if not records:
        raise ValueError("Collection is empty")
    seen_ids, invariants = set(), {}
    required = ("base_model_fingerprint", "chat_template_sha256", "quantization_policy")
    for record in records:
        if not record.get("run_id") or record["run_id"] in seen_ids:
            raise ValueError("Every source rollout requires a unique run_id")
        seen_ids.add(record["run_id"])
        if record.get("phase") not in accepted_phases or record.get("config", {}).get("adapter"):
            raise ValueError(
                "RSFT sources must be unadapted-model collection records from declared phases"
            )
        if record.get("config_hash") != fingerprint(record.get("config")):
            raise ValueError("Source generation configuration fingerprint is invalid")
        metadata = record.get("metadata", {})
        current = {field: metadata.get(field) for field in required}
        if not all(current.values()) or current["quantization_policy"] != QUANTIZATION_POLICY:
            raise ValueError("Source base, tokenizer and NF4/BF16 provenance are required")
        if not invariants:
            invariants = current
        elif current != invariants:
            raise ValueError("Collection mixes base-model, tokenizer or quantization identities")
    source_configs = {record["config_hash"] for record in records}
    source_tasks = {record.get("task_hash") for record in records}
    if len(source_configs) != 1 or len(source_tasks) != 1 or None in source_tasks:
        raise ValueError("One declared collection configuration and task are required")
    evaluator_hashes = [record.get("evaluator_hash") for record in records]
    if (
        not all(isinstance(value, str) and value.strip() for value in evaluator_hashes)
        or len(set(evaluator_hashes)) != 1
    ):
        raise ValueError("Collection requires one consistent nonempty evaluator_hash")
    source_evaluator_hash = evaluator_hashes[0]
    if (
        hasattr(student_task, "evaluator_hash")
        and student_task.evaluator_hash != source_evaluator_hash
    ):
        raise ValueError("Source and student evaluators differ")
    task = student_task.agent_view() if hasattr(student_task, "agent_view") else student_task
    if action_only:
        if not task or not student_generation or student_generation.get("thinking") is not False:
            raise ValueError(
                "Action distillation requires a frozen non-thinking student task/configuration"
            )
        if student_generation.get("adapter"):
            raise ValueError("Own-success RSFT starts from the unadapted student base")
        if student_generation.get("preload_inputs"):
            raise ValueError("preload_inputs=True is unsupported; task inputs use read_input")
        if student_generation.get("quantization") != QUANTIZATION_POLICY:
            raise ValueError("Source and student quantization policies must match")
        if tokenizer is None:
            if not model:
                raise ValueError("Action distillation requires a tokenizer or model identifier")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        if fingerprint(tokenizer.chat_template) != invariants["chat_template_sha256"]:
            raise ValueError("Source and student tokenizer templates differ")
        eos_ids = set(eos_ids or {tokenizer.eos_token_id}) - {None}
        if not eos_ids:
            raise ValueError("At least one explicit EOS token is required")
        student_task_hash, student_config_hash = fingerprint(task), fingerprint(student_generation)
    else:
        student_task_hash, student_config_hash = (
            next(iter(source_tasks)),
            next(iter(source_configs)),
        )
    accepted, rejected, examples, counts, seen_targets = [], [], [], Counter(), {}
    for record in records:
        if action_only:
            turns, reason = action_examples(
                record,
                tokenizer=tokenizer,
                task=task,
                student_task_hash=student_task_hash,
                student_config_hash=student_config_hash,
                max_length=max_length,
                eos_ids=eos_ids,
            )
        else:
            turns, reason = training_examples_from_rollout(
                record, max_length=max_length, accepted_phases=accepted_phases
            )
        target_hash = (
            fingerprint([turn["labels"][turn["prompt_length"] :] for turn in turns])
            if not reason
            else None
        )
        if target_hash in seen_targets and not reason:
            reason = "duplicate_success_trajectory"
        if reason:
            counts[reason] += 1
            rejection = {"run_id": record["run_id"], "reason": reason}
            if reason == "duplicate_success_trajectory":
                rejection["duplicate_of"] = seen_targets[target_hash]
            rejected.append(rejection)
            continue
        seen_targets[target_hash] = record["run_id"]
        accepted.append(
            {
                "run_id": record["run_id"],
                "seed": record["seed"],
                "assistant_turns": len(turns),
                "assistant_tokens_sha256": target_hash,
                "generated_tokens": sum(turn["completion_length"] for turn in turns),
            }
        )
        examples.extend(turns)
    if not examples:
        raise ValueError(f"No usable own successes; rejection counts: {dict(counts)}")
    for example in examples:
        validate_training_example(example, max_length)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8"
    )
    temporary.replace(output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "own_success_action_distillation_rsft"
        if action_only
        else "own_success_full_trajectory_rsft",
        "assistance_disclosure": assistance_disclosure,
        "sources": [{"path": str(source), "sha256": file_sha256(source)}],
        "source_config": records[0]["config"],
        "source_config_hash": next(iter(source_configs)),
        "source_task_hash": next(iter(source_tasks)),
        "source_evaluator_hash": source_evaluator_hash,
        "student_task_hash": student_task_hash,
        "pool_attempts": len(records),
        "pool_successes": sum(row.get("score", {}).get("strict_pass") is True for row in records),
        "accepted_trajectories": len(accepted),
        "examples": len(examples),
        "input_tokens": sum(len(row["input_ids"]) for row in examples),
        "supervised_tokens": sum(row["completion_length"] for row in examples),
        "max_length": max_length,
        "accepted": accepted,
        "rejected": rejected,
        "rejections": dict(counts),
        "invariants": {
            **invariants,
            "config_hash": student_config_hash,
            "task_hash": student_task_hash,
            "evaluator_hash": source_evaluator_hash,
        },
        "dataset_sha256": file_sha256(output),
        "loss_mask": "Prompt/history/tool tokens -100; exact sampled assistant tokens only.",
        "length_policy": "Reject a whole trajectory if any turn exceeds max_length; never truncate.",
        "deduplication": "Exact target token sequences including turn boundaries.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--phase", action="append", help="Allowed collection phase; may be repeated"
    )
    parser.add_argument("--action-only", action="store_true")
    parser.add_argument("--student-task", type=Path)
    parser.add_argument("--student-config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--assistance-disclosure")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    task = None
    if args.student_task:
        from .environment import load_task

        task = load_task(args.student_task)
    student = json.loads(args.student_config.read_text()) if args.student_config else None
    if student and "generation" in student:
        student = student["generation"]
    manifest = build_dataset(
        args.input,
        args.output,
        args.manifest,
        max_length=args.max_length,
        action_only=args.action_only,
        student_task=task,
        student_generation=student,
        model=args.model,
        revision=args.revision,
        assistance_disclosure=args.assistance_disclosure,
        accepted_phases=tuple(args.phase or ("collection", "train_pool")),
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("accepted_trajectories", "examples", "supervised_tokens", "rejections")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
