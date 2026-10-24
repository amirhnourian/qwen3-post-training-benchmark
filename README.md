# Qwen3-8B post-training benchmark

[![CPU checks](https://github.com/amirhnourian/qwen3-post-training-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/amirhnourian/qwen3-post-training-benchmark/actions/workflows/ci.yml)

An agent-training study combining rejection-sampling fine-tuning (RSFT) and a
short online GRPO continuation on one fixed, privately supplied task.

**[Read the benchmark report](reports/qwen3_post_training_report.pdf)** ·
[LaTeX source](reports/qwen3_post_training_report.tex) ·
[Step-by-step tutorial](docs/tutorial.md) · [Methods](docs/methods.md)

| Stage | Strict passes | 95% Wilson interval |
| --- | ---: | ---: |
| Qwen3-8B baseline | 0/100 | 0.00–3.70% |
| After RSFT | 100/100 | 96.30–100.00% |
| After RSFT + GRPO | 100/100 | 96.30–100.00% |

The RSFT targets came from **one distinct successful Qwen3-8B trajectory**:
two assistant-action examples and 132 supervised tokens per dataset pass.
Collection received disclosed task-specific source guidance. The adapter
completed 40 cumulative updates, including continuation from checkpoint 26.
GRPO then completed 12 groups of four rollouts, with 16 actual optimizer updates;
four equal-reward groups skipped updates.

These results measure repeated stochastic success on the same task. Fresh
seeds do not create unseen tasks. Memorization is plausible, transfer is
untested, and GRPO matched an already saturated RSFT score.

## What is public

This is a **post-experiment refactor** of the reusable pipeline. It includes a
synthetic arithmetic environment, CPU tests, optional model/training modules,
sanitized aggregate measurements and a report. The original task, checker
answers, raw trajectories, training examples and trained adapters are not
published. The synthetic demo is a separate example; it did not produce the
reported benchmark results. See [reproducibility boundaries](docs/reproducibility.md).

## Quick start: CPU only

Use Python 3.12 from a checkout of this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
qwen-posttrain demo --task examples/arithmetic/task.json --output runs/demo.jsonl
qwen-posttrain verify-results --summary results/benchmark.json --curve results/grpo_reward_curve.csv
```

The demo uses a deterministic synthetic generator and bounded tools. It needs
no GPU, model weights, account or model download. The core package has no
third-party runtime dependencies. Tests deny network connections by default.

```bash
python -m pytest -q
python -m ruff check src tests tools
python -m build
```

CI also installs CPU-only PyTorch and Transformers to exercise tiny-model
loss/gradient tests without downloading pretrained weights. Those tests skip in
the minimal development environment when optional ML dependencies are absent.
A passing CPU job is not a GPU-training claim.

## GPU work is explicit

The measured workstation used an **RTX 5090 with 32 GB VRAM**, an **AMD Ryzen 9
9950X3D**, and **60 GiB usable RAM**. Quantized model loading and QLoRA adapter
training were used.

Install the tested CUDA wheel separately, then the pinned optional stack:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[gpu]'
qwen-posttrain evaluate --help
qwen-posttrain train-rsft --help
qwen-posttrain train-grpo --help
```

Model downloads and training require an explicit command and user-provided
configuration. Start with [the reproducibility guide](docs/reproducibility.md);
the public synthetic task is suitable for exercising interfaces, not validating
the original study's effect size.

## Read the project

| Path | Purpose |
| --- | --- |
| [`docs/tutorial.md`](docs/tutorial.md) | Learn the workflow and inspect a safe example |
| [`docs/methods.md`](docs/methods.md) | Evaluation, assisted RSFT, QLoRA and GRPO definitions |
| [`docs/architecture.md`](docs/architecture.md) | Environment, model, data and trainer boundaries |
| [`docs/reproducibility.md`](docs/reproducibility.md) | CPU commands, GPU setup and evidence limits |
| [`docs/references.md`](docs/references.md) | Official documentation and original papers by concept |
| [`src/qwen_posttrain/`](src/qwen_posttrain/) | Reusable implementation |
| [`examples/arithmetic/task.json`](examples/arithmetic/task.json) | Public synthetic demo task |
| [`results/`](results/) | Sanitized aggregate measurements |
| [`reports/`](reports/) | Benchmark PDF, LaTeX and figures |
| [`agent.md`](agent.md) | Short public refactor work log |

Coding assistants supported implementation and review. Target authorship and
collection assistance are disclosed separately in the methods.

The source is available for inspection and review under the
[all-rights-reserved notice](LICENSE). Third-party dependencies and model
weights retain their own licenses.
