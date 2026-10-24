"""Task isolation and bounded tools using a clearly synthetic public task."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_posttrain.environment import TaskEnvironment, calculate, load_task

ROOT = Path(__file__).resolve().parents[1]


def demo_task():
    return load_task(ROOT / "examples/arithmetic/task.json")


def test_evaluator_configuration_is_not_in_agent_messages_or_tool_outputs():
    task = replace(demo_task(), scoring={"kind": "exact_text", "expected": "CHECKER_ONLY_SENTINEL"})
    assert "scoring" not in task.agent_view()
    assert "CHECKER_ONLY_SENTINEL" not in json.dumps(task.initial_messages())
    view = task.agent_view()
    view["inputs"]["expression"] = "changed"
    assert task.inputs["expression"] != "changed"
    with TaskEnvironment(task) as environment:
        replies = [
            environment.run_tool("write_answer", {"content": "wrong"}),
            environment.run_tool("submit", {}),
        ]
        assert "CHECKER_ONLY_SENTINEL" not in json.dumps(replies)
        assert "reward" not in json.dumps(replies)
        assert not environment.score["strict_pass"]


def test_agent_and_evaluator_identities_are_bound_separately():
    task = demo_task()
    changed_checker = replace(task, scoring={"kind": "numeric", "expected": -47})
    assert task.fingerprint() == changed_checker.fingerprint()
    assert task.evaluator_hash != changed_checker.evaluator_hash
    assert replace(task, inputs={"expression": "13 + 29"}).fingerprint() != task.fingerprint()


def test_allowlisted_inputs_and_tools_do_not_read_host_files():
    with TaskEnvironment(demo_task()) as environment:
        for name, args in (
            ("read_input", {"name": "../secret"}),
            ("shell", {"command": "id"}),
            ("calculate", {"expression": "open('secret').read()"}),
        ):
            assert "error" in environment.run_tool(name, args)
        assert environment.run_tool("read_input", {"name": "expression"}) == {
            "content": "(23 * 11) - 19"
        }


def test_fresh_answer_state_and_budget_are_enforced():
    task = demo_task()
    with TaskEnvironment(task, max_tool_calls=1) as first:
        assert first.run_tool("write_answer", {"content": "234"}) == {"written": True}
        assert first.score["strict_pass"]  # Artifact score is distinct from submit protocol.
        assert "error" in first.run_tool("submit", {})
        assert not first.submitted
    with TaskEnvironment(task) as second:
        assert second.answer is None and not second.score["strict_pass"]
        second.run_tool("write_answer", {"content": "234"})
        second.run_tool("submit", {})
        assert "error" in second.run_tool("write_answer", {"content": "different"})


@pytest.mark.parametrize(
    "expression,expected",
    [("(23 * 11) - 19", 234), ("17 / 4", 4.25), ("-19 % 6", 5), ("2 ** -3", 0.125)],
)
def test_bounded_arithmetic(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "True",
        "name + 1",
        "9 ** 99",
        "1 / 0",
        "[1, 2]",
        "(1).__class__",
        "1e9999",
        "9" * 257,
    ],
)
def test_executable_nonfinite_and_unbounded_expressions_are_rejected(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        calculate(expression)


def test_explicit_task_schema_rejects_unexpected_fields(tmp_path):
    payload = json.loads((ROOT / "examples/arithmetic/task.json").read_text())
    payload["automatic_host_directory"] = "not-allowed"
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unexpected task fields"):
        load_task(path)


def test_bundled_example_is_exactly_the_documented_synthetic_task():
    assert (ROOT / "src/qwen_posttrain/demo_task.json").read_bytes() == (
        ROOT / "examples/arithmetic/task.json"
    ).read_bytes()
