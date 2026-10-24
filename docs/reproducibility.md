# Reproducibility guide

## What can be reproduced publicly

The public checkout supports three distinct activities:

1. Run a deterministic synthetic tool-use demonstration and CPU regression tests.
2. Recalculate the released aggregate statistics and GRPO curve checks.
3. Run new model experiments on tasks and demonstrations you provide explicitly.

Exact replay of the reported private-task study requires inputs, checker and
raw model records that are not released. This repository is a post-experiment
refactor. It cannot independently prove private target authorship or regenerate
the original fine-tuned adapters from the public aggregates. Its synthetic
example is not the reported benchmark.

## CPU installation and checks

Use Python 3.12. The core package has no third-party dependencies; the `dev`
extra installs pinned test, lint and build tools.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
qwen-posttrain demo
qwen-posttrain demo --config configs/demo.json --output runs/demo.jsonl
qwen-posttrain analyze --input runs/demo.jsonl --phase synthetic_demo --output runs/demo-summary.json
qwen-posttrain verify-results --summary results/benchmark.json --curve results/grpo_reward_curve.csv
python -m pytest -q
python -m ruff check src tests
python -m build
```

The bundled default demo also works outside the checkout after package
installation. Result-verification paths are relative to the checkout unless
you pass explicit paths. Demo records use illustrative character IDs and must
not be treated as Qwen-generated training examples.

For the additional tiny-model loss and gradient tests, install a CPU-only
PyTorch wheel, then the `test-ml` group:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,test-ml]'
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

These tests initialize small random models locally; they do not download
pretrained weights. The test suite denies network connections. Dependency
installation itself uses package indexes. The CI workflow uses this CPU stack,
runs lint, tests, demo/result checks and source/wheel builds. Actual test counts
are those printed by the current suite, not counts from the earlier experiment.

## Explicit GPU setup

The measured stack used Python 3.12, PyTorch **2.7.1+cu128**, Transformers
**4.57.6**, PEFT **0.18.1**, Accelerate **1.12.0**, and bitsandbytes **0.49.2**.
Pinned optional groups live in [`pyproject.toml`](../pyproject.toml). The GPU
and CPU Torch commands below are alternatives for their respective environments.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[gpu]'
```

Model retrieval is a separate explicit operation. Respect the model's own
license and use an immutable revision:

```bash
hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir models/Qwen3-8B
```

The measured adapter workload ran on an RTX 5090 with 32 GB VRAM, an AMD
Ryzen 9 9950X3D and 60 GiB usable RAM. Other devices and library builds can
behave differently. A fixed seed alone does not guarantee bitwise reproduction;
see [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/2.7/notes/randomness.html).

## Exercise real-model interfaces on the public task

Inspect [`configs/evaluate.example.json`](../configs/evaluate.example.json).
The following recipe consistently uses the downloaded local model so collection
and training have the same identity. Create a local configuration without
changing the published example:

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path('configs/evaluate.example.json').read_text())
config['generation']['model'] = 'models/Qwen3-8B'
Path('runs').mkdir(exist_ok=True)
Path('runs/toy-generation.json').write_text(json.dumps(config, indent=2))
PY
```

A model evaluation is an explicit opt-in operation:

```bash
qwen-posttrain evaluate \
  --task examples/arithmetic/task.json \
  --config runs/toy-generation.json \
  --output runs/toy-evaluation.jsonl --count 4 --start-seed 71000
```

This creates a new synthetic-task experiment. It does not reproduce the
private task's baseline. Existing output files are not overwritten. The
allowlisted environment keeps checker values out of agent observations.

For new RSFT source collection, use the declared `collection` phase with the
unadapted base model and a fresh output path:

```bash
qwen-posttrain evaluate \
  --task examples/arithmetic/task.json \
  --config runs/toy-generation.json \
  --output runs/toy-collection.jsonl --phase collection --count 8 --start-seed 72000
qwen-posttrain build-rsft \
  --input runs/toy-collection.jsonl \
  --output runs/toy-rsft.jsonl --manifest runs/toy-rsft.manifest.json
qwen-posttrain train-rsft \
  --dataset runs/toy-rsft.jsonl --manifest runs/toy-rsft.manifest.json \
  --model models/Qwen3-8B --output adapters/toy-rsft --max-steps 4
```

The source and training model identities must match. The builder fails if
there are no usable verified successes; it does not substitute reference
answers. This simple recipe uses full-trajectory examples. The measured study's
action-only transformation additionally requires `--action-only`, a frozen
`--student-task`, `--student-config` and matching tokenizer via `--model`.
Use `qwen-posttrain build-rsft --help` to inspect the exact flags.

Training configuration can also be supplied through `train-rsft --config`.
`RSFTConfig` in [`rsft.py`](../src/qwen_posttrain/rsft.py) defines its fields.
Recovery requires a matching saved checkpoint and explicit
`--resume-from-checkpoint`; it continues the existing total-update budget.

For a new GRPO experiment, create an explicit configuration for
`qwen-posttrain train-grpo --config YOUR_CONFIG --task YOUR_TASK`. Required
fields include `groups`, `group_size`, `iterations_per_group`, `learning_rate`,
`start_seed`, `generation`, `output_adapter` and `evidence_dir`. Generation
must name an existing initial adapter, use positive temperature, `top_p=1`
and `top_k=0`. This implementation declares `beta=0` and no weight decay.
Read [`grpo_training.py`](../src/qwen_posttrain/grpo_training.py) and the
[GRPO method](methods.md#online-grpo-continuation) before starting. Existing
started/output paths are preserved; this interface does not silently restart
an interrupted GRPO run.

## Rebuild the public report

The public report is generated from sanitized aggregate JSON/CSV and an
editable standalone LaTeX template. Install the report group:

```bash
python -m pip install -e '.[report]'
python tools/build_report.py --source-version YOUR_SOURCE_COMMIT
```

That regenerates LaTeX and PNG figures. To compile the PDF, supply a Tectonic
0.17 executable with its required TeX bundle available:

```bash
python tools/build_report.py \
  --source-version YOUR_SOURCE_COMMIT \
  --tectonic /path/to/tectonic
```

The builder checks that the exported PDF has two pages. Visual review is
still required after layout changes. The report identifies the checked source
version; generated report files can be committed afterward without placing
their own final commit hash inside themselves. Rebuilding editable sources is
supported; identical PDF bytes across toolchains are not promised.
