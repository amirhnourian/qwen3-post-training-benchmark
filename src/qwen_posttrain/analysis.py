"""Dependency-free statistics and verification of sanitized result summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .modeling import fingerprint


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if (
        any(isinstance(x, bool) or not isinstance(x, int) for x in (successes, trials))
        or not 0 <= successes <= trials
        or trials < 1
    ):
        raise ValueError("Expected integer counts with 0 <= successes <= trials and trials > 0")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie in (0, 1)")
    z = statistics.NormalDist().inv_cdf((1 + confidence) / 2)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials**2))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_difference(
    before_successes: int, before_trials: int, after_successes: int, after_trials: int
) -> dict[str, float]:
    first_low, first_high = wilson_interval(before_successes, before_trials)
    second_low, second_high = wilson_interval(after_successes, after_trials)
    first, second = before_successes / before_trials, after_successes / after_trials
    difference = second - first
    return {
        "difference": difference,
        "ci95_low": max(
            -1.0, difference - math.sqrt((second - second_low) ** 2 + (first_high - first) ** 2)
        ),
        "ci95_high": min(
            1.0, difference + math.sqrt((second_high - second) ** 2 + (first - first_low) ** 2)
        ),
    }


def fisher_two_sided(
    before_successes: int, before_trials: int, after_successes: int, after_trials: int
) -> float:
    wilson_interval(before_successes, before_trials)
    wilson_interval(after_successes, after_trials)
    total = before_successes + after_successes
    observed = math.comb(before_trials, before_successes) * math.comb(after_trials, after_successes)
    numerator = sum(
        math.comb(before_trials, count) * math.comb(after_trials, total - count)
        for count in range(max(0, total - after_trials), min(before_trials, total) + 1)
        if math.comb(before_trials, count) * math.comb(after_trials, total - count) <= observed
    )
    return min(1.0, numerator / math.comb(before_trials + after_trials, total))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at line {number}")
            result.append(value)
    return result


def summarize_records(
    records: list[dict[str, Any]],
    *,
    phase: str = "evaluation",
    error_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize one homogeneous cohort and preserve infrastructure events.

    Model failures remain in the denominator. Failed executions from an errors
    sidecar are distinct events, including retries of a completed seed.
    """
    observations = [row for row in records if row.get("status") != "runtime_error"]
    if not observations:
        raise ValueError("No completed model observations")
    if len({r["run_id"] for r in observations}) != len(observations) or len(
        {r["seed"] for r in observations}
    ) != len(observations):
        raise ValueError("Duplicate completed episode IDs or seeds")
    identities = []
    for row in observations:
        config, metadata = row.get("config", {}), row.get("metadata", {})
        if not row.get("task_hash") or not row.get("evaluator_hash"):
            raise ValueError("Task and evaluator fingerprints are required")
        if row.get("config_hash") != fingerprint(config):
            raise ValueError("Generation configuration fingerprint is invalid")
        identity = {
            "task_hash": row["task_hash"],
            "evaluator_hash": row["evaluator_hash"],
            "config_hash": row["config_hash"],
            **{
                key: metadata.get(key)
                for key in (
                    "base_model_fingerprint",
                    "quantization_policy",
                    "chat_template_sha256",
                    "adapter_model_sha256",
                    "adapter_config_sha256",
                    "model_commit",
                )
            },
            "backend": metadata.get("backend", "model"),
        }
        if metadata.get("is_model_benchmark") is not False:
            if not all(
                identity[key]
                for key in ("base_model_fingerprint", "quantization_policy", "chat_template_sha256")
            ):
                raise ValueError("Model, tokenizer and quantization identities are required")
            if identity["quantization_policy"] != config.get("quantization"):
                raise ValueError("Recorded quantization differs from generation configuration")
            if config.get("adapter") and not all(
                identity[key] for key in ("adapter_model_sha256", "adapter_config_sha256")
            ):
                raise ValueError("An adapted evaluation requires adapter fingerprints")
        identities.append(identity)
    if len({fingerprint(identity) for identity in identities}) != 1:
        raise ValueError(
            "Cannot combine different task, evaluator, generation, model or adapter identities"
        )
    events = [row for row in records if row.get("status") == "runtime_error"]
    for event in error_records or []:
        if event.get("status") != "runtime_error":
            raise ValueError("An infrastructure sidecar must contain failed executions only")
        events.append(event)
    unique_events = {}
    for event in events:
        detail = {
            key: event.get(key)
            for key in ("run_id", "seed", "status", "error", "started_at", "finished_at")
        }
        unique_events[fingerprint(detail)] = detail
    successes, rewards = 0, []
    for row in observations:
        score = row["score"]
        strict = score.get("strict_pass") is True
        if row["success"] is not strict:
            raise ValueError("Episode success differs from the strict score")
        reward = score["reward"]
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or not 0 <= reward <= 1
        ):
            raise ValueError("Reward must be a finite number in [0, 1]")
        successes += strict
        rewards.append(reward)
    low, high = wilson_interval(successes, len(observations))
    return {
        "phase": phase,
        "successes": successes,
        "trials": len(observations),
        "pass_rate": successes / len(observations),
        "ci95_low": low,
        "ci95_high": high,
        "mean_reward": statistics.mean(rewards),
        "infrastructure_execution_events": len(unique_events),
        "infrastructure_events": list(unique_events.values()),
        "provenance": identities[0],
        "generated_tokens": sum(row.get("generated_tokens", 0) for row in observations),
        "scope": "Attempt-level reliability on the supplied task; no claim of task transfer.",
    }


def verify_results(
    summary_path: str | Path, curve_path: str | Path, *, provenance_path: str | Path | None = None
) -> dict[str, Any]:
    """Recompute public summary arithmetic, reward groups and bound checksums.

    This verifies the published summaries, not unavailable original task traces.
    It performs no model loading, generation, training or network access.
    """
    summary_path, curve_path = Path(summary_path), Path(curve_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def close(left: Any, right: float) -> bool:
        return (
            not isinstance(left, bool)
            and isinstance(left, (int, float))
            and math.isfinite(left)
            and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)
        )

    stages = summary["stages"]
    require(len({s["name"] for s in stages}) == len(stages), "Duplicate stage names")
    for stage in stages:
        successes, trials = stage["successes"], stage["trials"]
        low, high = wilson_interval(successes, trials)
        require(close(stage["pass_rate"], successes / trials), "Incorrect stage pass rate")
        require(
            close(stage["ci95_low"], low) and close(stage["ci95_high"], high),
            "Incorrect Wilson confidence interval",
        )
        require(
            close(stage["mean_reward"], stage["mean_reward"]) and 0 <= stage["mean_reward"] <= 1,
            "Invalid stage mean reward",
        )
    by_name = {s["name"]: s for s in stages}
    for label, before_name, after_name in (
        ("rsft_minus_baseline", "baseline", "rsft"),
        ("grpo_minus_rsft", "rsft", "grpo"),
    ):
        if label not in summary.get("comparisons", {}):
            continue
        before, after = by_name[before_name], by_name[after_name]
        computed = newcombe_difference(
            before["successes"], before["trials"], after["successes"], after["trials"]
        )
        for key, value in computed.items():
            require(
                close(summary["comparisons"][label][key], value),
                "Incorrect stage difference confidence interval",
            )
        expected_p = fisher_two_sided(
            before["successes"], before["trials"], after["successes"], after["trials"]
        )
        recorded_p = summary["comparisons"][label]["fisher_two_sided_p"]
        require(
            isinstance(recorded_p, (int, float))
            and not isinstance(recorded_p, bool)
            and math.isfinite(recorded_p)
            and math.isclose(recorded_p, expected_p, rel_tol=1e-10, abs_tol=0.0),
            "Incorrect exact comparison test",
        )
    with curve_path.open(encoding="utf-8", newline="") as handle:
        curve = list(csv.DictReader(handle))
    grpo = summary["grpo"]
    require(len(curve) == grpo["groups"], "Incorrect reward-group count")
    cursor, zero_groups, tokens, presentations = 0, 0, 0, 0
    for index, group in enumerate(curve, 1):
        require(int(group["group"]) == index, "Reward groups must be unique and ordered")
        rewards = json.loads(group["episode_rewards"])
        require(
            len(rewards) == grpo["group_size"]
            and all(close(r, r) and 0 <= r <= 1 for r in rewards),
            "Invalid episode rewards",
        )
        mean = sum(rewards) / len(rewards)
        constant = all(r == rewards[0] for r in rewards)
        require(close(float(group["mean_reward"]), mean), "Incorrect group mean reward")
        require(
            close(float(group["strict_pass_rate"]), sum(r == 1 for r in rewards) / len(rewards)),
            "Incorrect group strict fraction",
        )
        require(
            group["zero_variance"] in {"true", "false"}
            and (group["zero_variance"] == "true") is constant,
            "Incorrect constant-reward flag",
        )
        require(int(group["optimizer_updates_before"]) == cursor, "Incorrect group optimizer start")
        group_tokens = int(group["generated_tokens"])
        require(group_tokens > 0, "Invalid group generated-token count")
        tokens += group_tokens
        if constant:
            zero_groups += 1
        else:
            cursor += grpo["iterations_per_nonconstant_group"]
            presentations += group_tokens * grpo["iterations_per_nonconstant_group"]
        require(
            int(group["optimizer_updates_after"]) == cursor,
            "Constant groups must skip updates; other groups use the declared iterations",
        )
    require(
        cursor == grpo["optimizer_updates"] and zero_groups == grpo["zero_variance_groups"],
        "Incorrect optimizer or constant-group totals",
    )
    require(
        tokens == grpo["generated_tokens"]
        and presentations == grpo["optimization_token_presentations"],
        "Incorrect sampled/optimization token totals",
    )
    require(grpo["rollouts"] == grpo["groups"] * grpo["group_size"], "Incorrect rollout total")
    provenance_path = (
        Path(provenance_path) if provenance_path else summary_path.parent / "provenance.json"
    )
    checked = 0
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        project = provenance_path.resolve().parent.parent
        for artifact in provenance["public_artifacts"]:
            path = (project / artifact["path"]).resolve()
            require(
                path.is_relative_to(project), "Provenance artifact escapes the project directory"
            )
            require(
                path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"],
                "Public artifact checksum mismatch",
            )
            checked += 1
    return {
        "status": "passed",
        "stages": len(stages),
        "groups": len(curve),
        "optimizer_updates": cursor,
        "zero_variance_groups": zero_groups,
        "provenance_files_checked": checked,
        "scope": "Public summary arithmetic and checksums only; original task traces are retained separately.",
    }
