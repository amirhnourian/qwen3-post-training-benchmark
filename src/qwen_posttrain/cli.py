"""Small command-line entry point; model imports occur only when requested."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib import resources
from pathlib import Path

from .analysis import read_jsonl, summarize_records, verify_results
from .environment import load_task
from .rollouts import DemoGenerator, run_attempt


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _summarize_saved(path: Path, phase: str) -> dict:
    errors = path.with_name(path.stem + ".errors.jsonl")
    return summarize_records(
        read_jsonl(path), phase=phase, error_records=read_jsonl(errors) if errors.exists() else []
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    delegated = {"build-rsft": "dataset", "train-rsft": "rsft", "train-grpo": "grpo_training"}
    if argv and argv[0] in delegated:
        importlib.import_module(f".{delegated[argv[0]]}", __package__).main(argv[1:])
        return 0
    parser = argparse.ArgumentParser(
        description="Post-training tools and an offline synthetic demonstration"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="CPU-only deterministic walkthrough, not a model benchmark"
    )
    demo.add_argument(
        "--task", type=Path, help="Explicit task JSON; defaults to the bundled synthetic task"
    )
    demo.add_argument("--output", type=Path)
    demo.add_argument("--config", type=Path, help="Optional deterministic-demo configuration")
    demo.add_argument("--seed", type=int)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate an explicitly configured Qwen policy on a supplied task"
    )
    evaluate.add_argument("--task", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--count", type=int, default=10)
    evaluate.add_argument("--start-seed", type=int, default=100)
    evaluate.add_argument("--phase", default="evaluation")
    analyze = commands.add_parser(
        "analyze", help="Compute strict rates and Wilson intervals from saved episodes"
    )
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--phase", default="evaluation")
    verify = commands.add_parser(
        "verify-results", help="Verify sanitized summary statistics and reward-curve arithmetic"
    )
    verify.add_argument("--summary", type=Path, default=Path("results/benchmark.json"))
    verify.add_argument("--curve", type=Path, default=Path("results/grpo_reward_curve.csv"))
    verify.add_argument("--provenance", type=Path)
    for name in delegated:
        commands.add_parser(
            name, help=f"Run {name}; use '{name} --help' for its options", add_help=False
        )
    args = parser.parse_args(argv)
    if args.command == "demo":
        settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if settings.get("backend", "deterministic-demo") != "deterministic-demo":
            raise ValueError("demo accepts only the deterministic-demo backend")
        task_path = (
            args.task
            or settings.get("task")
            or resources.files("qwen_posttrain").joinpath("demo_task.json")
        )
        seed = args.seed if args.seed is not None else settings.get("seed", 7)
        record = run_attempt(
            DemoGenerator(), load_task(task_path), seed=seed, phase="synthetic_demo"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "kind": "synthetic demonstration only; not a model measurement",
                    "success": record["success"],
                    "answer": record["answer"],
                    "model_turns": len(record["steps"]),
                    "status": record["status"],
                },
                indent=2,
            )
        )
        return 0 if record["success"] else 1
    if args.command == "evaluate":
        from .modeling import GenerationConfig, QwenGenerator

        if args.count < 1:
            parser.error("--count must be positive")
        if args.output.exists():
            raise FileExistsError(
                "Use a fresh output file; existing observations are never overwritten"
            )
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        config = GenerationConfig(**payload.get("generation", payload))
        config.validate()
        task = load_task(args.task)
        generator = QwenGenerator(config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for seed in range(args.start_seed, args.start_seed + args.count):
            record = run_attempt(generator, task, seed=seed, phase=args.phase)
            target = (
                args.output.with_suffix(".errors.jsonl")
                if record["status"] == "runtime_error"
                else args.output
            )
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["status"] == "runtime_error":
                raise RuntimeError(record["error"])
        print(json.dumps(_summarize_saved(args.output, args.phase), indent=2))
        return 0
    if args.command == "analyze":
        summary = _summarize_saved(args.input, args.phase)
        _write_json(args.output, summary)
        print(json.dumps(summary, indent=2))
        return 0
    print(
        json.dumps(
            verify_results(args.summary, args.curve, provenance_path=args.provenance), indent=2
        )
    )
    return 0
