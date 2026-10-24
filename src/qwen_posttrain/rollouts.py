"""Multi-turn episodes with exact generated tokens and isolated tool state."""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
import time
from typing import Any

from .environment import TaskEnvironment, TaskSpec
from .modeling import fingerprint, parse_assistant_output


def run_attempt(
    generator: Any,
    task: TaskSpec,
    *,
    seed: int,
    phase: str = "evaluation",
    run_id: str | None = None,
    environment_factory=TaskEnvironment,
) -> dict[str, Any]:
    """Run one fresh episode; infrastructure errors remain explicit records.

    ``generator`` supplies config, metadata, set_seed and generate. The Qwen
    backend and the CPU demonstration implement this same small protocol.
    """
    config = generator.config
    public_config = (
        dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(vars(config))
    )
    started = time.monotonic()
    messages = task.initial_messages()
    record = {
        "schema_version": 1,
        "run_id": run_id or f"{phase}-{seed}",
        "phase": phase,
        "seed": seed,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": public_config,
        "config_hash": fingerprint(public_config),
        "task_hash": task.fingerprint(),
        "evaluator_hash": task.evaluator_hash,
        "metadata": copy.deepcopy(generator.metadata),
        "steps": [],
        "generated_tokens": 0,
        "status": "max_steps",
        "submitted": False,
        "success": False,
    }
    generator.set_seed(seed)
    with environment_factory(task, max_tool_calls=config.max_steps * 4) as environment:
        try:
            for index in range(config.max_steps):
                remaining = config.max_total_new_tokens - record["generated_tokens"]
                if remaining <= 0:
                    record["status"] = "generation_budget"
                    break
                generated = generator.generate(messages, task.tools, remaining_budget=remaining)
                step = {"step_index": index, **generated, "tool_results": []}
                record["steps"].append(step)
                record["generated_tokens"] += len(generated["generated_token_ids"])
                if generated.get("error"):
                    record["status"] = generated["error"]
                    break
                parsed = parse_assistant_output(
                    generated["raw_text"], thinking=config.thinking, step=index
                )
                step.update(parse_error=parsed["error"], assistant_message=parsed["message"])
                messages.append(parsed["message"])
                if parsed["error"]:
                    record["status"] = parsed["error"]
                elif not parsed["calls"]:
                    record["status"] = "finished_without_submit"
                else:
                    for call in parsed["calls"]:
                        function = call["function"]
                        result = environment.run_tool(function["name"], function["arguments"])
                        step["tool_results"].append(
                            {
                                "name": function["name"],
                                "arguments": function["arguments"],
                                "result": result,
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "name": function["name"],
                                "tool_call_id": call["id"],
                                "content": json.dumps(
                                    result, sort_keys=True, separators=(",", ":")
                                ),
                            }
                        )
                        if environment.submitted:
                            record["status"] = "submitted"
                            break
                if environment.submitted or parsed["error"] or not parsed["calls"]:
                    break
        except Exception as exc:
            record.update(status="runtime_error", error=f"{type(exc).__name__}: {exc}")
        record.update(
            answer=environment.answer, submitted=environment.submitted, score=environment.score
        )
        record["success"] = record["score"]["strict_pass"]
    record.update(
        messages=messages,
        finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        duration_seconds=time.monotonic() - started,
    )
    return record


@dataclasses.dataclass(frozen=True)
class DemoConfig:
    model: str = "deterministic-synthetic-demo"
    max_steps: int = 4
    max_context: int = 16384
    max_new_tokens: int = 1024
    max_total_new_tokens: int = 4096
    thinking: bool = False


class DemoGenerator:
    """A deterministic tool walkthrough, explicitly not a neural-model result."""

    config = DemoConfig()
    metadata = {
        "backend": "deterministic-demo",
        "is_model_benchmark": False,
        "token_encoding": "Unicode code points for demonstration only",
    }

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def generate(self, messages, tools, *, remaining_budget):
        del tools
        previous = messages[-1]
        if previous["role"] != "tool":
            name, args = "read_input", {"name": "expression"}
        else:
            observation = json.loads(previous["content"])
            if previous["name"] == "read_input":
                name, args = "calculate", {"expression": observation["content"]}
            elif previous["name"] == "calculate":
                name, args = "write_answer", {"content": str(observation["result"])}
            else:
                name, args = "submit", {}
        raw = (
            "<tool_call>"
            + json.dumps({"name": name, "arguments": args}, sort_keys=True)
            + "</tool_call>"
        )
        tokens = [ord(c) for c in raw][: min(remaining_budget, self.config.max_new_tokens)]
        return {
            "error": None,
            "prompt_token_ids": [ord(c) for c in json.dumps(messages, sort_keys=True)],
            "generated_token_ids": tokens,
            "raw_text": "".join(chr(c) for c in tokens),
            "effective_max_new_tokens": min(remaining_budget, self.config.max_new_tokens),
            "hit_generation_limit": len(tokens) < len(raw),
        }
