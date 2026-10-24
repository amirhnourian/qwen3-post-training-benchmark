"""Offline demo and exact episode recording; no model downloads or GPUs."""

import builtins
import contextlib
import io
import json
from pathlib import Path

import pytest

from qwen_posttrain.analysis import summarize_records
from qwen_posttrain.cli import main
from qwen_posttrain.environment import load_task
from qwen_posttrain.rollouts import DemoGenerator, run_attempt

ROOT = Path(__file__).resolve().parents[1]


def task():
    return load_task(ROOT / "examples/arithmetic/task.json")


def test_demo_replays_exact_model_tokens_and_tool_observations():
    row = run_attempt(DemoGenerator(), task(), seed=13, phase="synthetic_demo")
    assert row["success"] and row["submitted"] and row["answer"] == "234"
    assert row["metadata"]["is_model_benchmark"] is False
    assert [s["tool_results"][0]["name"] for s in row["steps"]] == [
        "read_input",
        "calculate",
        "write_answer",
        "submit",
    ]
    assert row["generated_tokens"] == sum(len(s["generated_token_ids"]) for s in row["steps"])
    for step in row["steps"]:
        assert step["raw_text"] == "".join(chr(token) for token in step["generated_token_ids"])
        assert step["prompt_token_ids"]
    assert "scoring" not in json.dumps(row["messages"][:2])


def test_malformed_action_remains_a_failed_observation():
    class Malformed(DemoGenerator):
        def generate(self, messages, tools, *, remaining_budget):
            output = super().generate(messages, tools, remaining_budget=remaining_budget)
            output["raw_text"] = '<tool_call>{"name":'
            output["generated_token_ids"] = list(map(ord, output["raw_text"]))
            return output

    row = run_attempt(Malformed(), task(), seed=14)
    assert not row["success"] and row["status"] == "malformed_tool_call_tags"
    summary = summarize_records([row])
    assert summary["trials"] == 1 and summary["successes"] == 0
    assert summary["infrastructure_execution_events"] == 0


def test_infrastructure_errors_are_separate_from_completed_observations():
    class Broken(DemoGenerator):
        def generate(self, *args, **kwargs):
            raise RuntimeError("synthetic infrastructure exception")

    failure = run_attempt(Broken(), task(), seed=15)
    completed = run_attempt(DemoGenerator(), task(), seed=16)
    assert failure["status"] == "runtime_error" and "error" in failure
    summary = summarize_records([failure, completed])
    assert summary["trials"] == 1 and summary["successes"] == 1
    assert summary["infrastructure_execution_events"] == 1


def test_demo_and_help_do_not_import_machine_learning_dependencies(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "peft", "bitsandbytes"}:
            raise AssertionError("The offline demo must not import model dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    output = tmp_path / "demo.jsonl"
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        assert main(["demo", "--output", str(output)]) == 0
    assert "not a model measurement" in captured.getvalue()
    assert json.loads(output.read_text())["success"]
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0


def test_analyze_cli_reads_saved_observations(tmp_path):
    source, output = tmp_path / "episodes.jsonl", tmp_path / "summary.json"
    source.write_text(json.dumps(run_attempt(DemoGenerator(), task(), seed=17)) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["analyze", "--input", str(source), "--output", str(output)]) == 0
    summary = json.loads(output.read_text())
    assert summary["successes"] == summary["trials"] == 1
    assert summary["ci95_low"] < 1 and summary["ci95_high"] == 1


def test_demo_configuration_cannot_silently_select_a_model_backend(tmp_path):
    config = tmp_path / "demo.json"
    config.write_text(json.dumps({"backend": "deterministic-demo", "seed": 31}))
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["demo", "--config", str(config)]) == 0
    config.write_text(json.dumps({"backend": "gpu-model"}))
    with pytest.raises(ValueError, match="deterministic-demo"):
        main(["demo", "--config", str(config)])


def test_analyze_cli_includes_adjacent_infrastructure_events(tmp_path):
    source, output = tmp_path / "attempts.jsonl", tmp_path / "summary.json"
    row = run_attempt(DemoGenerator(), task(), seed=37)
    source.write_text(json.dumps(row) + "\n")
    source.with_name("attempts.errors.jsonl").write_text(
        json.dumps(
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "status": "runtime_error",
                "error": "synthetic earlier execution",
            }
        )
        + "\n"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["analyze", "--input", str(source), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["trials"] == result["successes"] == 1
    assert result["infrastructure_execution_events"] == 1
    assert result["provenance"]["evaluator_hash"] == row["evaluator_hash"]
