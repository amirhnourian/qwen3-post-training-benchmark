"""An explicit task contract and a small, bounded tool environment.

Task scoring is evaluator-only. Model-visible messages contain no scoring
configuration, and tools never execute Python, shell commands, or host reads.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


_ALLOWED_TOOLS = {"read_input", "calculate", "write_answer", "submit"}


@dataclass(frozen=True)
class TaskSpec:
    """An explicit input task; ``scoring`` is never part of its agent view."""

    task_id: str
    system_prompt: str
    user_prompt: str
    tools: list[dict[str, Any]]
    inputs: dict[str, str] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.task_id or not isinstance(self.task_id, str):
            raise ValueError("A nonempty task_id is required")
        if not all(
            isinstance(v, str) for v in (self.system_prompt, self.user_prompt, self.description)
        ):
            raise ValueError("Task prompts and description must be strings")
        if not isinstance(self.inputs, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.inputs.items()
        ):
            raise ValueError("Inputs must be a mapping of names to literal strings")
        names = []
        for tool in self.tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError("Tools must use the function-tool schema")
            name = tool.get("function", {}).get("name")
            if name not in _ALLOWED_TOOLS:
                raise ValueError(f"Unsupported tool: {name}")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("Duplicate tool names")
        kind = self.scoring.get("kind")
        if kind not in {"exact_text", "numeric"} or "expected" not in self.scoring:
            raise ValueError("Scoring requires exact_text/numeric kind and expected value")
        if kind == "exact_text" and not isinstance(self.scoring["expected"], str):
            raise ValueError("exact_text expected value must be a string")
        if kind == "numeric":
            expected, tolerance = (
                self.scoring["expected"],
                self.scoring.get("absolute_tolerance", 0),
            )
            if (
                any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in (expected, tolerance)
                )
                or tolerance < 0
            ):
                raise ValueError(
                    "Numeric scoring values must be finite; tolerance must be nonnegative"
                )

    def agent_view(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "task_id": self.task_id,
                "system_prompt": self.system_prompt,
                "user_prompt": self.user_prompt,
                "tools": self.tools,
                "inputs": self.inputs,
                "description": self.description,
            }
        )

    def initial_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]

    def fingerprint(self) -> str:
        return _fingerprint(self.agent_view())

    @property
    def evaluator_hash(self) -> str:
        return _fingerprint(self.scoring)


def load_task(path: str | Path) -> TaskSpec:
    """Read one explicit JSON task, with no discovery of surrounding files."""
    source = Path(path)
    if source.is_dir():
        source = source / "task.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Task JSON must be an object")
    allowed = {
        "task_id",
        "system_prompt",
        "user_prompt",
        "tools",
        "inputs",
        "scoring",
        "description",
    }
    if set(payload) - allowed:
        raise ValueError(f"Unexpected task fields: {sorted(set(payload) - allowed)}")
    return TaskSpec(**payload)


def calculate(expression: str) -> int | float:
    """Evaluate bounded arithmetic syntax rather than model-supplied code."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 256:
        raise ValueError("An arithmetic expression of at most 256 characters is required")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression is too complex")
    binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def bounded(value: Any) -> int | float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or abs(value) > 2**128:
            raise ValueError("Arithmetic magnitude exceeds the configured bound")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Arithmetic result must be finite")
        return value

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            return bounded(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return bounded(unary[type(node.op)](visit(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 32:
                raise ValueError("Exponent exceeds the configured bound")
            return bounded(binary[type(node.op)](left, right))
        raise ValueError("Only finite arithmetic literals and operators are allowed")

    return visit(tree.body)


class TaskEnvironment:
    """Fresh in-memory answer state; tools can access only declared inputs."""

    def __init__(self, task: TaskSpec, *, max_tool_calls: int = 32):
        if max_tool_calls < 1:
            raise ValueError("Tool-call budget must be positive")
        self.task = task
        self.max_tool_calls = max_tool_calls
        self.calls = 0
        self.answer: str | None = None
        self.submitted = False
        self._tools = {t["function"]["name"] for t in task.tools}

    def __enter__(self) -> TaskEnvironment:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.submitted:
            return {"error": "The answer has already been submitted"}
        if name not in self._tools or not isinstance(args, dict):
            return {"error": "Unsupported tool or invalid arguments"}
        if self.calls >= self.max_tool_calls:
            return {"error": "Tool-call budget exhausted"}
        self.calls += 1
        try:
            if name == "read_input":
                if set(args) != {"name"} or args["name"] not in self.task.inputs:
                    raise ValueError("Only explicitly declared input names may be read")
                return {"content": self.task.inputs[args["name"]]}
            if name == "calculate":
                if set(args) != {"expression"}:
                    raise ValueError("calculate requires only expression")
                return {"result": calculate(args["expression"])}
            if name == "write_answer":
                content = args.get("content")
                if (
                    set(args) != {"content"}
                    or not isinstance(content, str)
                    or len(content.encode()) > 65536
                ):
                    raise ValueError("write_answer requires bounded text content")
                self.answer = content
                return {"written": True}
            if args:
                raise ValueError("submit takes no arguments")
            self.submitted = True
            return {"submitted": True}
        except (ValueError, TypeError, SyntaxError, ArithmeticError) as exc:
            return {"error": str(exc)}

    @property
    def score(self) -> dict[str, Any]:
        correct = False
        if self.answer is not None:
            if self.task.scoring["kind"] == "exact_text":
                correct = self.answer.strip() == self.task.scoring["expected"].strip()
            else:
                try:
                    observed = float(self.answer.strip())
                    correct = math.isfinite(observed) and abs(
                        observed - self.task.scoring["expected"]
                    ) <= self.task.scoring.get("absolute_tolerance", 0)
                except ValueError:
                    correct = False
        return {
            "reward": float(correct),
            "strict_pass": correct,
            "failure_category": None
            if correct
            else "missing_answer"
            if self.answer is None
            else "wrong_answer",
        }
