#!/usr/bin/env python3
"""Render the standalone LaTeX report and plots from published measurements.

Run from a source checkout. A TeX compiler is optional when only generating the
editable source and PNGs; pass --tectonic to also export and check the PDF.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--tectonic", help="Path to Tectonic; omit to generate source and PNGs only"
    )
    parser.add_argument("--source-version", default="initial public extraction")
    args = parser.parse_args()
    root = args.root.resolve()
    from qwen_posttrain.analysis import verify_results

    verify_results(
        root / "results/benchmark.json",
        root / "results/grpo_reward_curve.csv",
        provenance_path=root / "results/provenance.json",
    )
    data = json.loads((root / "results/benchmark.json").read_text())
    with (root / "results/grpo_reward_curve.csv").open(newline="") as handle:
        curve = list(csv.DictReader(handle))
    values = {
        "REPO_URL": "https://github.com/amirhnourian/qwen3-post-training-benchmark",
        "SOURCE_VERSION": args.source_version,
    }
    for prefix, row in zip(("BASE", "RSFT", "GRPO"), data["stages"]):
        for key, value in {
            "SUCC": row["successes"],
            "N": row["trials"],
            "RATE": f"{row['pass_rate'] * 100:.1f}",
            "LOW": f"{row['ci95_low'] * 100:.2f}",
            "HIGH": f"{row['ci95_high'] * 100:.2f}",
            "REWARD": f"{row['mean_reward']:.3f}",
        }.items():
            values[f"{prefix}_{key}"] = str(value)
    for prefix, name in (("RSFT", "rsft_minus_baseline"), ("GRPO", "grpo_minus_rsft")):
        for suffix, key in (("LOW", "ci95_low"), ("HIGH", "ci95_high")):
            values[f"{prefix}_DIFF_{suffix}"] = f"{data['comparisons'][name][key] * 100:+.2f}"
    values["PASS_POINTS"] = "\n".join(
        f"{index} {row['pass_rate']} {max(0, row['ci95_high'] - row['pass_rate'])} {max(0, row['pass_rate'] - row['ci95_low'])}"
        for index, row in enumerate(data["stages"])
    )
    values["REWARD_POINTS"] = " ".join(f"({row['group']},{row['mean_reward']})" for row in curve)
    values["SUCCESS_POINTS"] = " ".join(
        f"({row['group']},{row['strict_pass_rate']})" for row in curve
    )
    template = (root / "reports/report_template.tex").read_text()
    for key, value in values.items():
        template = template.replace(f"@{key}@", value)
    if re.search(r"@[A-Z_]+@", template):
        raise ValueError("Unresolved report template placeholder")
    source = root / "reports/qwen3_post_training_report.tex"
    source.write_text(template)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160}
    )
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    for index, row in enumerate(data["stages"]):
        rate = row["pass_rate"]
        ax.errorbar(
            index,
            rate,
            yerr=[[max(0, rate - row["ci95_low"])], [max(0, row["ci95_high"] - rate)]],
            fmt="o",
            capsize=6,
            color="#245B73",
            markersize=8,
        )
        ax.annotate(
            f"{row['successes']}/{row['trials']}",
            (index, rate),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
        )
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["Baseline", "RSFT", "RSFT + GRPO"],
        xlim=(-0.5, 2.5),
        ylim=(-0.03, 1.13),
        ylabel="Strict pass rate",
        title="Three fresh 100-attempt cohorts on one fixed task",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        root / "reports/pass_rates.png", facecolor="white", metadata={"Software": "qwen-posttrain"}
    )
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    x = [int(row["group"]) for row in curve]
    ax.plot(
        x,
        [float(row["mean_reward"]) for row in curve],
        marker="o",
        color="#9B5B35",
        label="Mean verifier reward",
    )
    ax.plot(
        x,
        [float(row["strict_pass_rate"]) for row in curve],
        marker="s",
        ls="--",
        color="#245B73",
        label="Strict success fraction",
    )
    ax.set(
        xticks=x,
        xlabel="On-policy group (four fresh episodes)",
        ylabel="Reward / success fraction",
        ylim=(-0.04, 1.06),
        title="Actual GRPO training groups: full-support T=2 exploration",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        root / "reports/grpo_reward_curve.png",
        facecolor="white",
        metadata={"Software": "qwen-posttrain"},
    )
    plt.close(fig)
    if args.tectonic:
        subprocess.run(
            [args.tectonic, "--keep-logs", "--outdir", str(source.parent), str(source)], check=True
        )
        from pypdf import PdfReader

        pages = len(PdfReader(source.with_suffix(".pdf")).pages)
        if pages != 2:
            raise ValueError(f"Expected two report pages, got {pages}; repair layout")
        print(json.dumps({"pdf": str(source.with_suffix(".pdf")), "pages": pages}))
    else:
        print(json.dumps({"source": str(source), "plots": 2}))


if __name__ == "__main__":
    main()
