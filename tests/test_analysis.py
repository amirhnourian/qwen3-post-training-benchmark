"""Independent statistical checks and corruption tests on public summaries."""

import copy
import csv
import json
from pathlib import Path

import pytest

from qwen_posttrain.analysis import (
    fisher_two_sided,
    newcombe_difference,
    verify_results,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]


def test_wilson_handles_boundary_outcomes_without_zero_uncertainty():
    low, high = wilson_interval(0, 25)
    assert low == 0 and 0 < high < 1
    other_low, other_high = wilson_interval(25, 25)
    assert other_high == 1 and other_low == pytest.approx(1 - high)
    for counts in ((-1, 25), (26, 25), (0, 0), (True, 25)):
        with pytest.raises(ValueError):
            wilson_interval(*counts)


def test_identical_ceiling_cohorts_do_not_prove_equivalence():
    interval = newcombe_difference(25, 25, 25, 25)
    assert interval["difference"] == 0
    assert interval["ci95_low"] < 0 < interval["ci95_high"]
    assert fisher_two_sided(25, 25, 25, 25) == 1
    assert fisher_two_sided(0, 12, 12, 12) < 0.001


def test_current_public_results_and_reward_curve_verify():
    checked = verify_results(
        ROOT / "results/benchmark.json", ROOT / "results/grpo_reward_curve.csv"
    )
    assert checked["status"] == "passed"
    assert checked["stages"] == 3 and checked["groups"] == 12
    assert checked["optimizer_updates"] == 16 and checked["zero_variance_groups"] == 4
    assert checked["provenance_files_checked"] >= 2


@pytest.mark.parametrize(
    "corruption",
    [
        "interval",
        "difference",
        "p_value",
        "group_mean",
        "strict_fraction",
        "extra_update",
        "tokens",
    ],
)
def test_corrupted_summary_arithmetic_is_rejected(tmp_path, corruption):
    summary = json.loads((ROOT / "results/benchmark.json").read_text())
    with (ROOT / "results/grpo_reward_curve.csv").open(newline="") as stream:
        curve = list(csv.DictReader(stream))
    if corruption == "interval":
        summary["stages"][0]["ci95_high"] = 0.9
    elif corruption == "difference":
        summary["comparisons"]["grpo_minus_rsft"]["difference"] = 0.1
    elif corruption == "p_value":
        summary["comparisons"]["rsft_minus_baseline"]["fisher_two_sided_p"] = 0.0
    elif corruption == "group_mean":
        curve[0]["mean_reward"] = "0.99"
    elif corruption == "strict_fraction":
        curve[0]["strict_pass_rate"] = "0.99"
    elif corruption == "extra_update":
        next(row for row in curve if row["zero_variance"] == "true")["optimizer_updates_after"] = (
            "99"
        )
    else:
        summary["grpo"]["generated_tokens"] += 1
    path, csv_path = tmp_path / "summary.json", tmp_path / "curve.csv"
    path.write_text(json.dumps(copy.deepcopy(summary)))
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    with pytest.raises(ValueError):
        verify_results(path, csv_path)


def _synthetic_episode(seed):
    from qwen_posttrain.environment import load_task
    from qwen_posttrain.rollouts import DemoGenerator, run_attempt

    return run_attempt(
        DemoGenerator(), load_task(ROOT / "examples/arithmetic/task.json"), seed=seed
    )


@pytest.mark.parametrize(
    "field", ["task", "evaluator", "configuration", "base", "template", "quantization", "adapter"]
)
def test_new_run_summary_rejects_heterogeneous_cohorts(field):
    from qwen_posttrain.analysis import summarize_records
    from qwen_posttrain.modeling import fingerprint

    first, second = _synthetic_episode(81), _synthetic_episode(82)
    if field == "task":
        second["task_hash"] = "changed-task"
    elif field == "evaluator":
        second["evaluator_hash"] = "changed-checker"
    elif field == "configuration":
        second["config"]["max_steps"] += 1
        second["config_hash"] = fingerprint(second["config"])
    else:
        key = {
            "base": "base_model_fingerprint",
            "template": "chat_template_sha256",
            "quantization": "quantization_policy",
            "adapter": "adapter_model_sha256",
        }[field]
        second["metadata"][key] = "changed-identity"
    with pytest.raises(ValueError, match="Cannot combine"):
        summarize_records([first, second])


def test_new_run_summary_rejects_forged_config_hash_and_exposes_provenance():
    from qwen_posttrain.analysis import summarize_records

    row = _synthetic_episode(83)
    summary = summarize_records([row])
    assert summary["provenance"]["task_hash"] == row["task_hash"]
    assert summary["provenance"]["evaluator_hash"] == row["evaluator_hash"]
    assert summary["provenance"]["config_hash"] == row["config_hash"]
    row["config_hash"] = "incorrect"
    with pytest.raises(ValueError, match="configuration fingerprint"):
        summarize_records([row])


def test_new_run_summary_deduplicates_events_without_dropping_completed_seed():
    from qwen_posttrain.analysis import summarize_records

    row = _synthetic_episode(84)
    error = {
        "run_id": row["run_id"],
        "seed": row["seed"],
        "status": "runtime_error",
        "error": "synthetic failure",
        "started_at": "first-attempt",
    }
    summary = summarize_records([error, row], error_records=[copy.deepcopy(error)])
    assert summary["trials"] == summary["successes"] == 1
    assert summary["infrastructure_execution_events"] == 1
    assert summary["infrastructure_events"][0]["seed"] == row["seed"]
